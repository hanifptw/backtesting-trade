"""Freqtrade-inspired Futures 24H composite strategy.

Native reimplementation of ideas from freqtrade-strategies futures examples.
No Freqtrade code is imported or copied. The strategy combines several simple
trend/volatility signals into one 24/7 futures preset:

    adx_sma             ADX filter + SMA fast/slow direction
    ema_reinforced      EMA fast/slow direction aligned with HTF regime
    triple_supertrend   three Supertrend directions agree
    trend_obv           price above/below EMA with OBV confirmation
    volatility_breakout close change exceeds closed-HTF ATR threshold
    ott                 adaptive MA crosses its OTT line

Default mode is ``ensemble``. Individual modes use the selected signal only.
All higher-timeframe data is resampled from closed HTF candles only.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from backtesting import Strategy
from backtesting.lib import OHLCV_AGG

from app.core.indicators import adx, atr, ema, sma, supertrend


_REGIME_BULL = 1.0
_REGIME_BEAR = -1.0
_REGIME_NONE = 0.0
_MIN_STOP_BUMP_PCT = 0.1
_SIGNAL_MODES = (
    "adx_sma",
    "ema_reinforced",
    "triple_supertrend",
    "trend_obv",
    "volatility_breakout",
    "ott",
)
_ALL_MODES = ("ensemble",) + _SIGNAL_MODES


def closed_htf_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample open-time indexed candles to fully closed HTF candles.

    With 1h bars indexed by open time, the 4h candle labelled 04:00 contains
    00:00, 01:00, 02:00, and 03:00, excluding the 04:00 candle itself.
    """
    return (
        df.resample(_safe_htf_rule(rule), label="right", closed="left")
        .agg(OHLCV_AGG)
        .dropna()
    )


def crossed_above_prior(left, right) -> np.ndarray:
    """True only on the bar where left crosses above right."""
    left_s = pd.Series(left).astype(float)
    right_s = pd.Series(right).astype(float)
    return np.array((left_s > right_s) & (left_s.shift(1) <= right_s.shift(1)), dtype=float)


def crossed_below_prior(left, right) -> np.ndarray:
    """True only on the bar where left crosses below right."""
    left_s = pd.Series(left).astype(float)
    right_s = pd.Series(right).astype(float)
    return np.array((left_s < right_s) & (left_s.shift(1) >= right_s.shift(1)), dtype=float)


def obv(close, volume) -> np.ndarray:
    close_s = pd.Series(close).astype(float)
    volume_s = pd.Series(volume).astype(float)
    direction = np.sign(close_s.diff()).fillna(0.0)
    return np.array((direction * volume_s).cumsum(), dtype=float)


def ott_values(close, period: int = 2, percent: float = 1.4) -> np.ndarray:
    """Adaptive moving average and OTT line.

    This is a compact native implementation of the OTT idea used by Freqtrade's
    futures example: a Chande-momentum-adjusted EMA with dynamic trailing bands.
    Returns shape (2, N): row 0 = adaptive MA, row 1 = OTT line.
    """
    close_arr = np.asarray(close, dtype=float)
    n = len(close_arr)
    var = np.full(n, np.nan, dtype=float)
    ott_line = np.full(n, np.nan, dtype=float)
    if n == 0:
        return np.array(np.vstack([var, ott_line]), dtype=float)

    period = max(1, int(period))
    percent = max(0.0, float(percent))
    alpha = 2.0 / (period + 1.0)

    close_s = pd.Series(close_arr)
    diff = close_s.diff()
    up_sum = diff.clip(lower=0.0).rolling(9, min_periods=1).sum()
    down_sum = (-diff.clip(upper=0.0)).rolling(9, min_periods=1).sum()
    denom = (up_sum + down_sum).replace(0.0, np.nan)
    cmo = ((up_sum - down_sum) / denom).abs().fillna(0.0).to_numpy(dtype=float)

    first = int(np.where(np.isfinite(close_arr))[0][0]) if np.isfinite(close_arr).any() else 0
    var[first] = close_arr[first]
    for i in range(first + 1, n):
        if not np.isfinite(close_arr[i]):
            continue
        weight = min(1.0, max(0.0, alpha * cmo[i]))
        prev = var[i - 1] if np.isfinite(var[i - 1]) else close_arr[i - 1]
        var[i] = weight * close_arr[i] + (1.0 - weight) * prev

    long_stop = np.full(n, np.nan, dtype=float)
    short_stop = np.full(n, np.nan, dtype=float)
    direction = np.full(n, 1.0, dtype=float)
    raw_ott = np.full(n, np.nan, dtype=float)

    for i in range(first, n):
        if not np.isfinite(var[i]):
            continue
        band = var[i] * percent * 0.01
        new_long = var[i] - band
        new_short = var[i] + band

        if i == first or not np.isfinite(long_stop[i - 1]) or not np.isfinite(short_stop[i - 1]):
            long_stop[i] = new_long
            short_stop[i] = new_short
            direction[i] = 1.0
        else:
            long_stop[i] = max(new_long, long_stop[i - 1]) if var[i] > long_stop[i - 1] else new_long
            short_stop[i] = min(new_short, short_stop[i - 1]) if var[i] < short_stop[i - 1] else new_short

            if var[i] > short_stop[i - 1]:
                direction[i] = 1.0
            elif var[i] < long_stop[i - 1]:
                direction[i] = -1.0
            else:
                direction[i] = direction[i - 1]

        mt = long_stop[i] if direction[i] > 0 else short_stop[i]
        raw_ott[i] = (
            mt * (200.0 + percent) / 200.0
            if var[i] > mt
            else mt * (200.0 - percent) / 200.0
        )

    ott_line = pd.Series(raw_ott).shift(2).to_numpy(dtype=float)
    return np.array(np.vstack([var, ott_line]), dtype=float)


def ott_var(close, period: int = 2, percent: float = 1.4) -> np.ndarray:
    return np.array(ott_values(close, period, percent)[0], dtype=float)


def ott_line(close, period: int = 2, percent: float = 1.4) -> np.ndarray:
    return np.array(ott_values(close, period, percent)[1], dtype=float)


def _safe_htf_rule(rule: str) -> str:
    allowed = {"2h", "4h", "6h", "8h", "1d"}
    rule = str(rule).lower()
    return rule if rule in allowed else "4h"


def _safe_mode(mode: str) -> str:
    mode = str(mode).lower()
    return mode if mode in _ALL_MODES else "ensemble"


def _finite(*values: float) -> bool:
    return all(math.isfinite(float(v)) for v in values)


class FreqtradeFutures24H(Strategy):
    # Signal selection
    mode: str = "ensemble"
    htf_rule: str = "4h"
    regime_filter_enabled: bool = True
    min_confirmations: int = 2

    # Trend and momentum
    ema_fast: int = 20
    ema_slow: int = 50
    sma_fast: int = 12
    sma_slow: int = 48
    adx_period: int = 14
    adx_threshold: float = 25.0
    atr_period: int = 14
    min_atr_pct: float = 0.15
    max_atr_pct: float = 6.0

    # Supertrend ensemble
    supertrend_period_1: int = 10
    supertrend_mult_1: float = 2.0
    supertrend_period_2: int = 14
    supertrend_mult_2: float = 3.0
    supertrend_period_3: int = 21
    supertrend_mult_3: float = 4.0

    # OBV / OTT / volatility breakout
    obv_ema_period: int = 20
    ott_period: int = 2
    ott_percent: float = 1.4
    volatility_atr_mult: float = 1.5

    # Risk
    initial_sl_atr_mult: float = 2.0
    trail_activation_atr_mult: float = 1.0
    trail_distance_atr_mult: float = 3.0
    tp_rr: float = 0.0
    cooldown_bars: int = 3
    exit_on_opposite: bool = True

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
        volume = np.asarray(self.data.Volume, dtype=float)

        self._ema_fast = self.I(ema, close, self.ema_fast, name=f"EMA({self.ema_fast})")
        self._ema_slow = self.I(ema, close, self.ema_slow, name=f"EMA({self.ema_slow})")
        self._sma_fast = self.I(sma, close, self.sma_fast, name=f"SMA({self.sma_fast})")
        self._sma_slow = self.I(sma, close, self.sma_slow, name=f"SMA({self.sma_slow})")
        self._adx = self.I(
            adx,
            high,
            low,
            close,
            self.adx_period,
            name=f"ADX({self.adx_period})",
        )
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

        self._obv = self.I(obv, close, volume, name="OBV", plot=False)
        self._obv_ema = self.I(
            ema,
            self._obv,
            self.obv_ema_period,
            name=f"OBV EMA({self.obv_ema_period})",
            plot=False,
        )

        self._ott_var = self.I(
            ott_var,
            close,
            self.ott_period,
            self.ott_percent,
            name=f"OTT VAR({self.ott_period}, {self.ott_percent:g})",
            overlay=True,
        )
        self._ott_line = self.I(
            ott_line,
            close,
            self.ott_period,
            self.ott_percent,
            name=f"OTT({self.ott_period}, {self.ott_percent:g})",
            overlay=True,
        )

        st1 = supertrend(high, low, close, self.supertrend_period_1, self.supertrend_mult_1, "rma")
        st2 = supertrend(high, low, close, self.supertrend_period_2, self.supertrend_mult_2, "rma")
        st3 = supertrend(high, low, close, self.supertrend_period_3, self.supertrend_mult_3, "rma")
        self._st1_dir = self.I(lambda a=st1[1]: a, name="Supertrend 1 direction", plot=False)
        self._st2_dir = self.I(lambda a=st2[1]: a, name="Supertrend 2 direction", plot=False)
        self._st3_dir = self.I(lambda a=st3[1]: a, name="Supertrend 3 direction", plot=False)
        self.I(lambda a=np.where(st1[1] > 0, st1[0], np.nan): a, name="Supertrend 1 Bullish", overlay=True)
        self.I(lambda a=np.where(st1[1] < 0, st1[0], np.nan): a, name="Supertrend 1 Bearish", overlay=True)

        self._build_htf_context()

        self._entry_atr: float | None = None
        self._last_exit_bar: int | None = None
        self._closed_trades_seen = 0

    def next(self) -> None:
        self._mark_recent_exit()

        price = float(self.data.Close[-1])
        atr_now = float(self._atr[-1])
        regime = float(self._htf_regime[-1])

        if not _finite(price, atr_now, regime):
            return

        long_score, short_score, signal_state = self._scores()

        if self.position:
            self._update_trailing(price)
            if self.exit_on_opposite:
                if self.position.is_long and self._entry_gate("SHORT", short_score, regime, price, atr_now):
                    self.position.close()
                    self._open_position("SHORT", price, atr_now, short_score, signal_state)
                    return
                if self.position.is_short and self._entry_gate("LONG", long_score, regime, price, atr_now):
                    self.position.close()
                    self._open_position("LONG", price, atr_now, long_score, signal_state)
                    return
            return

        if self._cooldown_active() or not self._volatility_is_tradable(price, atr_now):
            return

        long_ok = self._entry_gate("LONG", long_score, regime, price, atr_now)
        short_ok = self._entry_gate("SHORT", short_score, regime, price, atr_now)
        if long_ok and short_ok:
            if long_score == short_score:
                return
            long_ok = long_score > short_score
            short_ok = short_score > long_score

        if long_ok:
            self._open_position("LONG", price, atr_now, long_score, signal_state)
        elif short_ok:
            self._open_position("SHORT", price, atr_now, short_score, signal_state)

    def _build_htf_context(self) -> None:
        df_main = self.data.df
        rule = _safe_htf_rule(self.htf_rule)
        df_htf = closed_htf_ohlcv(df_main, rule)
        if df_htf.empty:
            empty = np.full(len(df_main), np.nan, dtype=float)
            self._htf_regime = self.I(lambda a=empty: a, name=f"HTF Regime {rule}", plot=False)
            self._htf_atr = self.I(lambda a=empty: a, name=f"HTF ATR {rule}", plot=False)
            return

        htf_close = np.asarray(df_htf["Close"], dtype=float)
        htf_high = np.asarray(df_htf["High"], dtype=float)
        htf_low = np.asarray(df_htf["Low"], dtype=float)

        htf_ema_fast = ema(htf_close, self.ema_fast)
        htf_ema_slow = ema(htf_close, self.ema_slow)
        htf_adx = adx(htf_high, htf_low, htf_close, self.adx_period)
        htf_atr = atr(htf_high, htf_low, htf_close, self.atr_period, "rma")

        valid = (
            np.isfinite(htf_ema_fast)
            & np.isfinite(htf_ema_slow)
            & np.isfinite(htf_adx)
        )
        bull = (
            valid
            & (htf_close > htf_ema_slow)
            & (htf_ema_fast > htf_ema_slow)
            & (htf_adx >= self.adx_threshold)
        )
        bear = (
            valid
            & (htf_close < htf_ema_slow)
            & (htf_ema_fast < htf_ema_slow)
            & (htf_adx >= self.adx_threshold)
        )
        regime = np.full(len(df_htf), _REGIME_NONE, dtype=float)
        regime[bull] = _REGIME_BULL
        regime[bear] = _REGIME_BEAR

        self._htf_regime = self.I(
            lambda a=self._align_htf(regime, df_htf.index, df_main.index): a,
            name=f"HTF Regime {rule} (1 bull, -1 bear)",
            plot=False,
        )
        self._htf_atr = self.I(
            lambda a=self._align_htf(htf_atr, df_htf.index, df_main.index): a,
            name=f"HTF ATR({self.atr_period}) {rule}",
            plot=False,
        )
        self.I(
            lambda a=self._align_htf(htf_ema_fast, df_htf.index, df_main.index): a,
            name=f"HTF EMA{self.ema_fast} {rule}",
            overlay=True,
        )
        self.I(
            lambda a=self._align_htf(htf_ema_slow, df_htf.index, df_main.index): a,
            name=f"HTF EMA{self.ema_slow} {rule}",
            overlay=True,
        )

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

    def _scores(self) -> tuple[int, int, dict[str, tuple[bool, bool]]]:
        mode = _safe_mode(self.mode)
        if mode == "ensemble":
            states = {name: self._signal_state(name) for name in _SIGNAL_MODES}
            return (
                sum(1 for long, _ in states.values() if long),
                sum(1 for _, short in states.values() if short),
                states,
            )

        long_signal, short_signal = self._signal_event(mode)
        states = {mode: (long_signal, short_signal)}
        return (1 if long_signal else 0, 1 if short_signal else 0, states)

    def _signal_state(self, name: str) -> tuple[bool, bool]:
        price = float(self.data.Close[-1])
        prev_price = float(self.data.Close[-2]) if len(self.data.Close) > 1 else math.nan
        regime = float(self._htf_regime[-1])

        if name == "adx_sma":
            adx_now = float(self._adx[-1])
            fast = float(self._sma_fast[-1])
            slow = float(self._sma_slow[-1])
            if not _finite(adx_now, fast, slow) or adx_now < self.adx_threshold:
                return False, False
            return fast > slow, fast < slow

        if name == "ema_reinforced":
            fast = float(self._ema_fast[-1])
            slow = float(self._ema_slow[-1])
            if not _finite(fast, slow, regime):
                return False, False
            return fast > slow and regime == _REGIME_BULL, fast < slow and regime == _REGIME_BEAR

        if name == "triple_supertrend":
            dirs = [float(self._st1_dir[-1]), float(self._st2_dir[-1]), float(self._st3_dir[-1])]
            if not _finite(*dirs):
                return False, False
            return all(v > 0 for v in dirs), all(v < 0 for v in dirs)

        if name == "trend_obv":
            ema_fast_now = float(self._ema_fast[-1])
            obv_now = float(self._obv[-1])
            obv_ema_now = float(self._obv_ema[-1])
            if not _finite(price, ema_fast_now, obv_now, obv_ema_now):
                return False, False
            return price > ema_fast_now and obv_now > obv_ema_now, price < ema_fast_now and obv_now < obv_ema_now

        if name == "volatility_breakout":
            htf_atr = float(self._htf_atr[-1])
            if not _finite(price, prev_price, htf_atr) or htf_atr <= 0:
                return False, False
            change = price - prev_price
            threshold = self.volatility_atr_mult * htf_atr
            return change > threshold, -change > threshold

        if name == "ott":
            var = float(self._ott_var[-1])
            line = float(self._ott_line[-1])
            if not _finite(var, line):
                return False, False
            return var > line, var < line

        return False, False

    def _signal_event(self, name: str) -> tuple[bool, bool]:
        if len(self.data.Close) < 2:
            return False, False

        if name == "adx_sma":
            adx_now = float(self._adx[-1])
            if not _finite(adx_now) or adx_now < self.adx_threshold:
                return False, False
            return (
                self._crossed_above(self._sma_fast, self._sma_slow),
                self._crossed_below(self._sma_fast, self._sma_slow),
            )

        if name == "ema_reinforced":
            regime = float(self._htf_regime[-1])
            return (
                regime == _REGIME_BULL and self._crossed_above(self._ema_fast, self._ema_slow),
                regime == _REGIME_BEAR and self._crossed_below(self._ema_fast, self._ema_slow),
            )

        if name == "triple_supertrend":
            dirs_now = [float(self._st1_dir[-1]), float(self._st2_dir[-1]), float(self._st3_dir[-1])]
            dirs_prev = [float(self._st1_dir[-2]), float(self._st2_dir[-2]), float(self._st3_dir[-2])]
            if not _finite(*dirs_now, *dirs_prev):
                return False, False
            long_now = all(v > 0 for v in dirs_now)
            short_now = all(v < 0 for v in dirs_now)
            long_prev = all(v > 0 for v in dirs_prev)
            short_prev = all(v < 0 for v in dirs_prev)
            return long_now and not long_prev, short_now and not short_prev

        if name == "trend_obv":
            obv_now = float(self._obv[-1])
            obv_prev = float(self._obv[-2])
            obv_ema_now = float(self._obv_ema[-1])
            if not _finite(obv_now, obv_prev, obv_ema_now):
                return False, False
            return (
                self._crossed_above(self.data.Close, self._ema_fast) and obv_now > obv_ema_now and obv_now > obv_prev,
                self._crossed_below(self.data.Close, self._ema_fast) and obv_now < obv_ema_now and obv_now < obv_prev,
            )

        if name == "volatility_breakout":
            return self._signal_state(name)

        if name == "ott":
            return (
                self._crossed_above(self._ott_var, self._ott_line),
                self._crossed_below(self._ott_var, self._ott_line),
            )

        return False, False

    @staticmethod
    def _crossed_above(left, right) -> bool:
        left_now, right_now = float(left[-1]), float(right[-1])
        left_prev, right_prev = float(left[-2]), float(right[-2])
        return _finite(left_now, right_now, left_prev, right_prev) and left_prev <= right_prev and left_now > right_now

    @staticmethod
    def _crossed_below(left, right) -> bool:
        left_now, right_now = float(left[-1]), float(right[-1])
        left_prev, right_prev = float(left[-2]), float(right[-2])
        return _finite(left_now, right_now, left_prev, right_prev) and left_prev >= right_prev and left_now < right_now

    def _entry_gate(self, side: str, score: int, regime: float, price: float, atr_now: float) -> bool:
        if side == "LONG" and not self.long_enabled:
            return False
        if side == "SHORT" and not self.short_enabled:
            return False
        if not self._volatility_is_tradable(price, atr_now):
            return False

        threshold = max(1, int(self.min_confirmations)) if _safe_mode(self.mode) == "ensemble" else 1
        if score < threshold:
            return False

        if not self.regime_filter_enabled:
            return True
        return (side == "LONG" and regime == _REGIME_BULL) or (
            side == "SHORT" and regime == _REGIME_BEAR
        )

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

    def _open_position(
        self,
        side: str,
        price: float,
        atr_now: float,
        score: int,
        signal_state: dict[str, tuple[bool, bool]],
    ) -> None:
        risk = self.initial_sl_atr_mult * atr_now
        if risk <= 0:
            return

        active = [
            name
            for name, (long_on, short_on) in signal_state.items()
            if (side == "LONG" and long_on) or (side == "SHORT" and short_on)
        ]
        tag = {
            "mode": _safe_mode(self.mode),
            "side": side,
            "score": int(score),
            "signals": ",".join(active),
            "atr": round(float(atr_now), 6),
        }

        if side == "LONG":
            sl = price - risk
            tp = price + self.tp_rr * risk if self.tp_rr > 0 else None
            if sl <= 0 or sl >= price:
                return
            self.buy(size=self._position_size(self.long_size_mult), sl=sl, tp=tp, tag=tag)
        else:
            sl = price + risk
            tp = price - self.tp_rr * risk if self.tp_rr > 0 else None
            if sl <= price or (tp is not None and tp <= 0):
                return
            self.sell(size=self._position_size(self.short_size_mult), sl=sl, tp=tp, tag=tag)

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
    "mode": {
        "type": "select",
        "default": "ensemble",
        "label": "Mode",
        "group": "Signals",
        "options": [
            {"value": "ensemble", "label": "Ensemble"},
            {"value": "adx_sma", "label": "ADX + SMA"},
            {"value": "ema_reinforced", "label": "EMA reinforced"},
            {"value": "triple_supertrend", "label": "Triple Supertrend"},
            {"value": "trend_obv", "label": "Trend + OBV"},
            {"value": "volatility_breakout", "label": "Volatility breakout"},
            {"value": "ott", "label": "OTT"},
        ],
    },
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
    "regime_filter_enabled": {"type": "bool", "default": True, "label": "Enable HTF regime filter", "group": "Regime"},
    "min_confirmations": {"type": "int", "default": 2, "min": 1, "max": 6, "label": "Min ensemble confirmations", "group": "Signals"},
    "ema_fast": {"type": "int", "default": 20, "min": 2, "max": 200, "label": "EMA fast", "group": "Signals"},
    "ema_slow": {"type": "int", "default": 50, "min": 3, "max": 500, "label": "EMA slow", "group": "Signals"},
    "sma_fast": {"type": "int", "default": 12, "min": 2, "max": 200, "label": "SMA fast", "group": "Signals"},
    "sma_slow": {"type": "int", "default": 48, "min": 3, "max": 500, "label": "SMA slow", "group": "Signals"},
    "adx_period": {"type": "int", "default": 14, "min": 5, "max": 100, "label": "ADX period", "group": "Signals"},
    "adx_threshold": {"type": "float", "default": 25.0, "min": 5.0, "max": 80.0, "label": "ADX threshold", "group": "Signals"},
    "atr_period": {"type": "int", "default": 14, "min": 5, "max": 100, "label": "ATR period", "group": "Risk"},
    "min_atr_pct": {"type": "float", "default": 0.15, "min": 0.0, "max": 10.0, "label": "Min ATR %", "group": "Risk"},
    "max_atr_pct": {"type": "float", "default": 6.0, "min": 0.1, "max": 30.0, "label": "Max ATR %", "group": "Risk"},
    "supertrend_period_1": {"type": "int", "default": 10, "min": 2, "max": 100, "label": "Supertrend 1 period", "group": "Supertrend"},
    "supertrend_mult_1": {"type": "float", "default": 2.0, "min": 0.5, "max": 10.0, "label": "Supertrend 1 multiplier", "group": "Supertrend"},
    "supertrend_period_2": {"type": "int", "default": 14, "min": 2, "max": 100, "label": "Supertrend 2 period", "group": "Supertrend"},
    "supertrend_mult_2": {"type": "float", "default": 3.0, "min": 0.5, "max": 10.0, "label": "Supertrend 2 multiplier", "group": "Supertrend"},
    "supertrend_period_3": {"type": "int", "default": 21, "min": 2, "max": 100, "label": "Supertrend 3 period", "group": "Supertrend"},
    "supertrend_mult_3": {"type": "float", "default": 4.0, "min": 0.5, "max": 10.0, "label": "Supertrend 3 multiplier", "group": "Supertrend"},
    "obv_ema_period": {"type": "int", "default": 20, "min": 2, "max": 200, "label": "OBV EMA period", "group": "Signals"},
    "ott_period": {"type": "int", "default": 2, "min": 1, "max": 100, "label": "OTT period", "group": "Signals"},
    "ott_percent": {"type": "float", "default": 1.4, "min": 0.1, "max": 10.0, "label": "OTT percent", "group": "Signals"},
    "volatility_atr_mult": {"type": "float", "default": 1.5, "min": 0.1, "max": 10.0, "label": "Volatility breakout x HTF ATR", "group": "Signals"},
    "initial_sl_atr_mult": {"type": "float", "default": 2.0, "min": 0.5, "max": 10.0, "label": "Initial SL x ATR", "group": "Risk"},
    "trail_activation_atr_mult": {"type": "float", "default": 1.0, "min": 0.0, "max": 10.0, "label": "Trail activation x ATR", "group": "Risk"},
    "trail_distance_atr_mult": {"type": "float", "default": 3.0, "min": 0.5, "max": 20.0, "label": "Trail distance x ATR", "group": "Risk"},
    "tp_rr": {"type": "float", "default": 0.0, "min": 0.0, "max": 10.0, "label": "Take-profit RR (0=off)", "group": "Risk"},
    "cooldown_bars": {"type": "int", "default": 3, "min": 0, "max": 200, "label": "Cooldown bars", "group": "Risk"},
    "exit_on_opposite": {"type": "bool", "default": True, "label": "Exit/reverse on opposite signal", "group": "Risk"},
    "long_enabled": {"type": "bool", "default": True, "label": "Enable LONG", "group": "Side"},
    "short_enabled": {"type": "bool", "default": True, "label": "Enable SHORT", "group": "Side"},
    "trade_amount": {"type": "float", "default": 100.0, "min": 1.0, "max": 1_000_000.0, "label": "Trade amount (USDT margin)", "group": "Sizing"},
    "leverage_enabled": {"type": "bool", "default": True, "label": "Enable leverage", "group": "Sizing"},
    "leverage": {"type": "int", "default": 3, "min": 1, "max": 125, "label": "Leverage (x)", "group": "Sizing"},
    "long_size_mult": {"type": "float", "default": 1.0, "min": 0.1, "max": 3.0, "label": "LONG size multiplier", "group": "Sizing"},
    "short_size_mult": {"type": "float", "default": 0.7, "min": 0.1, "max": 3.0, "label": "SHORT size multiplier", "group": "Sizing"},
}
