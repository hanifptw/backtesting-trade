"""Adaptive Trend 24H strategy.

Trend-following preset for a crypto futures bot that may run around the clock.
The strategy uses closed higher-timeframe candles for market regime, then
enters on main-timeframe Donchian breakouts only when volatility is tradable.

Rules:
    HTF BULL  = close > EMA slow, EMA fast > EMA slow, ADX above threshold,
                and Supertrend direction bullish.
    HTF BEAR  = close < EMA slow, EMA fast < EMA slow, ADX above threshold,
                and Supertrend direction bearish.
    LONG      = HTF BULL + close breaks above the prior Donchian high.
    SHORT     = HTF BEAR + close breaks below the prior Donchian low.
    NO TRADE  = warm-up, chop/range, ADX too low, or ATR% outside bounds.

Risk is ATR-sized: initial stop is N x ATR from entry, optional RR take-profit,
and trailing starts after the trade moves at least N x entry ATR in our favor.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from backtesting import Strategy
from backtesting.lib import OHLCV_AGG

from app.core.indicators import adx, atr, ema, supertrend


_REGIME_BULL = 1.0
_REGIME_BEAR = -1.0
_REGIME_NONE = 0.0
_MIN_STOP_BUMP_PCT = 0.1


def donchian_high_prior(high, period: int) -> np.ndarray:
    """Highest high of the previous `period` bars, excluding current bar."""
    values = pd.Series(high).astype(float)
    return np.array(values.rolling(int(period)).max().shift(1), dtype=float)


def donchian_low_prior(low, period: int) -> np.ndarray:
    """Lowest low of the previous `period` bars, excluding current bar."""
    values = pd.Series(low).astype(float)
    return np.array(values.rolling(int(period)).min().shift(1), dtype=float)


def _safe_htf_rule(rule: str) -> str:
    allowed = {"2h", "4h", "6h", "8h", "1d"}
    rule = str(rule).lower()
    return rule if rule in allowed else "4h"


class AdaptiveTrend24H(Strategy):
    # HTF regime filter
    htf_rule: str = "4h"
    ema_fast: int = 50
    ema_slow: int = 200
    adx_period: int = 14
    adx_threshold: float = 25.0

    # Main timeframe entry and volatility filter
    atr_period: int = 14
    donchian_period: int = 20
    min_atr_pct: float = 0.15
    max_atr_pct: float = 6.0

    # Risk
    initial_sl_atr_mult: float = 2.0
    trail_activation_atr_mult: float = 1.0
    trail_distance_atr_mult: float = 3.0
    tp_rr: float = 0.0
    cooldown_bars: int = 3

    # Side and sizing
    long_enabled: bool = True
    short_enabled: bool = True
    trade_amount: float = 100.0
    leverage_enabled: bool = True
    leverage: int = 3
    long_size_mult: float = 1.0
    short_size_mult: float = 0.7

    def init(self) -> None:
        close = np.asarray(self.data.Close, dtype=float)
        high = np.asarray(self.data.High, dtype=float)
        low = np.asarray(self.data.Low, dtype=float)

        self._atr = self.I(
            atr,
            high,
            low,
            close,
            self.atr_period,
            "rma",
            name=f"ATR({self.atr_period})",
            plot=False,
        )
        self._donchian_high = self.I(
            donchian_high_prior,
            high,
            self.donchian_period,
            name=f"Donchian High prev({self.donchian_period})",
            overlay=True,
        )
        self._donchian_low = self.I(
            donchian_low_prior,
            low,
            self.donchian_period,
            name=f"Donchian Low prev({self.donchian_period})",
            overlay=True,
        )

        self._build_htf_regime()

        self._entry_atr: float | None = None
        self._last_exit_bar: int | None = None
        self._closed_trades_seen = 0

    def next(self) -> None:
        self._mark_recent_exit()

        price = float(self.data.Close[-1])
        atr_now = float(self._atr[-1])
        regime = float(self._htf_regime[-1])
        donchian_high = float(self._donchian_high[-1])
        donchian_low = float(self._donchian_low[-1])

        if any(math.isnan(v) for v in (price, atr_now, regime, donchian_high, donchian_low)):
            return

        if self.position:
            self._update_trailing(price)
            return

        if self._cooldown_active():
            return
        if not self._volatility_is_tradable(price, atr_now):
            return

        if (
            self.long_enabled
            and regime == _REGIME_BULL
            and price > donchian_high
        ):
            self._open_position("LONG", price, atr_now)
        elif (
            self.short_enabled
            and regime == _REGIME_BEAR
            and price < donchian_low
        ):
            self._open_position("SHORT", price, atr_now)

    def _build_htf_regime(self) -> None:
        df_main = self.data.df
        rule = _safe_htf_rule(self.htf_rule)
        df_htf = (
            df_main.resample(rule, label="right", closed="right")
            .agg(OHLCV_AGG)
            .dropna()
        )

        htf_close = np.asarray(df_htf["Close"], dtype=float)
        htf_high = np.asarray(df_htf["High"], dtype=float)
        htf_low = np.asarray(df_htf["Low"], dtype=float)

        ema_fast_arr = ema(htf_close, self.ema_fast)
        ema_slow_arr = ema(htf_close, self.ema_slow)
        adx_arr = adx(htf_high, htf_low, htf_close, self.adx_period)
        st_values = supertrend(htf_high, htf_low, htf_close, self.atr_period, 3.0, "rma")
        st_line = st_values[0]
        st_dir = st_values[1]

        valid = (
            np.isfinite(ema_fast_arr)
            & np.isfinite(ema_slow_arr)
            & np.isfinite(adx_arr)
            & np.isfinite(st_dir)
        )
        bull = (
            valid
            & (htf_close > ema_slow_arr)
            & (ema_fast_arr > ema_slow_arr)
            & (adx_arr >= self.adx_threshold)
            & (st_dir > 0)
        )
        bear = (
            valid
            & (htf_close < ema_slow_arr)
            & (ema_fast_arr < ema_slow_arr)
            & (adx_arr >= self.adx_threshold)
            & (st_dir < 0)
        )

        regime = np.full(len(df_htf), _REGIME_NONE, dtype=float)
        regime[bull] = _REGIME_BULL
        regime[bear] = _REGIME_BEAR

        htf_ema_fast = self._align_htf(ema_fast_arr, df_htf.index, df_main.index)
        htf_ema_slow = self._align_htf(ema_slow_arr, df_htf.index, df_main.index)
        htf_regime = self._align_htf(regime, df_htf.index, df_main.index)
        htf_bull_st = self._align_htf(np.where(st_dir > 0, st_line, np.nan), df_htf.index, df_main.index)
        htf_bear_st = self._align_htf(np.where(st_dir < 0, st_line, np.nan), df_htf.index, df_main.index)

        self._htf_regime = self.I(
            lambda a=htf_regime: a,
            name=f"HTF Regime {rule} (1 bull, -1 bear)",
            plot=False,
        )
        self.I(lambda a=htf_ema_fast: a, name=f"HTF EMA{self.ema_fast} {rule}", overlay=True)
        self.I(lambda a=htf_ema_slow: a, name=f"HTF EMA{self.ema_slow} {rule}", overlay=True)
        self.I(lambda a=htf_bull_st: a, name=f"HTF Supertrend {rule} Bullish", overlay=True)
        self.I(lambda a=htf_bear_st: a, name=f"HTF Supertrend {rule} Bearish", overlay=True)

    @staticmethod
    def _align_htf(
        arr_htf: np.ndarray,
        htf_index: pd.DatetimeIndex,
        main_index: pd.DatetimeIndex,
    ) -> np.ndarray:
        series = pd.Series(arr_htf, index=htf_index)
        aligned = (
            series.reindex(main_index.union(htf_index), method="ffill")
            .reindex(main_index)
        )
        return np.ascontiguousarray(aligned.to_numpy(dtype=float, copy=True))

    def _mark_recent_exit(self) -> None:
        n_closed = len(self.closed_trades)
        if n_closed <= self._closed_trades_seen:
            return
        self._closed_trades_seen = n_closed
        self._last_exit_bar = len(self.data) - 1
        self._entry_atr = None

    def _cooldown_active(self) -> bool:
        if self._last_exit_bar is None or self.cooldown_bars <= 0:
            return False
        return (len(self.data) - 1 - self._last_exit_bar) <= int(self.cooldown_bars)

    def _volatility_is_tradable(self, price: float, atr_now: float) -> bool:
        if price <= 0 or atr_now <= 0:
            return False
        atr_pct = atr_now / price * 100.0
        return self.min_atr_pct <= atr_pct <= self.max_atr_pct

    def _position_size(self, size_mult: float) -> float:
        equity = float(self.equity)
        size = self.trade_amount / equity * float(size_mult) if equity > 0 else 0.0
        return max(1e-6, min(0.9999, size))

    def _open_position(self, side: str, price: float, atr_now: float) -> None:
        risk = self.initial_sl_atr_mult * atr_now
        if risk <= 0:
            return

        if side == "LONG":
            sl = price - risk
            tp = price + self.tp_rr * risk if self.tp_rr > 0 else None
            if sl <= 0 or sl >= price:
                return
            self.buy(
                size=self._position_size(self.long_size_mult),
                sl=sl,
                tp=tp,
                tag={"regime": "BULL", "atr": round(atr_now, 6)},
            )
        else:
            sl = price + risk
            tp = price - self.tp_rr * risk if self.tp_rr > 0 else None
            if sl <= price or (tp is not None and tp <= 0):
                return
            self.sell(
                size=self._position_size(self.short_size_mult),
                sl=sl,
                tp=tp,
                tag={"regime": "BEAR", "atr": round(atr_now, 6)},
            )

        self._entry_atr = atr_now

    def _update_trailing(self, price: float) -> None:
        if not self.trades or self._entry_atr is None or self._entry_atr <= 0:
            return

        trade = self.trades[-1]
        entry = float(trade.entry_price)
        entry_atr = float(self._entry_atr)
        activation = self.trail_activation_atr_mult * entry_atr
        distance = self.trail_distance_atr_mult * entry_atr

        if trade.is_long:
            if price - entry < activation:
                return
            desired = price - distance
            current = float(trade.sl) if trade.sl is not None else 0.0
            if desired <= current:
                return
            improvement = (desired - current) / current * 100.0 if current > 0 else 100.0
            if improvement >= _MIN_STOP_BUMP_PCT:
                trade.sl = desired
        else:
            if entry - price < activation:
                return
            desired = price + distance
            current = float(trade.sl) if trade.sl is not None else float("inf")
            if desired >= current:
                return
            improvement = (
                (current - desired) / current * 100.0
                if current != float("inf")
                else 100.0
            )
            if improvement >= _MIN_STOP_BUMP_PCT:
                trade.sl = desired


SCHEMA = {
    "htf_rule": {
        "type": "select",
        "default": "4h",
        "label": "HTF rule",
        "group": "Regime",
        "options": [
            {"value": "2h", "label": "2h"},
            {"value": "4h", "label": "4h"},
            {"value": "6h", "label": "6h"},
            {"value": "8h", "label": "8h"},
            {"value": "1d", "label": "1d"},
        ],
    },
    "ema_fast": {"type": "int", "default": 50, "min": 5, "max": 200, "label": "HTF EMA fast", "group": "Regime"},
    "ema_slow": {"type": "int", "default": 200, "min": 20, "max": 500, "label": "HTF EMA slow", "group": "Regime"},
    "adx_period": {"type": "int", "default": 14, "min": 5, "max": 50, "label": "HTF ADX period", "group": "Regime"},
    "adx_threshold": {"type": "float", "default": 25.0, "min": 5.0, "max": 60.0, "label": "ADX trend threshold", "group": "Regime"},
    "atr_period": {"type": "int", "default": 14, "min": 5, "max": 100, "label": "ATR period", "group": "Entry"},
    "donchian_period": {"type": "int", "default": 20, "min": 5, "max": 200, "label": "Donchian period", "group": "Entry"},
    "min_atr_pct": {"type": "float", "default": 0.15, "min": 0.0, "max": 10.0, "label": "Min ATR %", "group": "Entry"},
    "max_atr_pct": {"type": "float", "default": 6.0, "min": 0.1, "max": 30.0, "label": "Max ATR %", "group": "Entry"},
    "initial_sl_atr_mult": {"type": "float", "default": 2.0, "min": 0.5, "max": 10.0, "label": "Initial SL x ATR", "group": "Risk"},
    "trail_activation_atr_mult": {"type": "float", "default": 1.0, "min": 0.0, "max": 10.0, "label": "Trail activation x ATR", "group": "Risk"},
    "trail_distance_atr_mult": {"type": "float", "default": 3.0, "min": 0.5, "max": 20.0, "label": "Trail distance x ATR", "group": "Risk"},
    "tp_rr": {"type": "float", "default": 0.0, "min": 0.0, "max": 10.0, "label": "Take-profit RR (0=off)", "group": "Risk"},
    "cooldown_bars": {"type": "int", "default": 3, "min": 0, "max": 100, "label": "Cooldown bars", "group": "Risk"},
    "long_enabled": {"type": "bool", "default": True, "label": "Enable LONG", "group": "Side"},
    "short_enabled": {"type": "bool", "default": True, "label": "Enable SHORT", "group": "Side"},
    "trade_amount": {"type": "float", "default": 100.0, "min": 1.0, "max": 1_000_000.0, "label": "Trade amount (USDT margin)", "group": "Sizing"},
    "leverage_enabled": {"type": "bool", "default": True, "label": "Enable leverage", "group": "Sizing"},
    "leverage": {"type": "int", "default": 3, "min": 1, "max": 125, "label": "Leverage (x)", "group": "Sizing"},
    "long_size_mult": {"type": "float", "default": 1.0, "min": 0.1, "max": 3.0, "label": "LONG size multiplier", "group": "Sizing"},
    "short_size_mult": {"type": "float", "default": 0.7, "min": 0.1, "max": 3.0, "label": "SHORT size multiplier", "group": "Sizing"},
}
