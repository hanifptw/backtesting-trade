"""BTC Trend-Pullback — risk-adjusted (Sharpe-optimized) 24/7 strategy.

Thesis: trade only with the HTF trend (EMA200 4h by default), and enter on
shallow pullbacks to a fast EMA (EMA20) when momentum resets (RSI re-crosses
the 50 level). Risk is ATR-sized — both the initial SL/TP and the trailing
stop scale with current volatility, so a single set of params transfers
better across regimes than fixed-percent rules.

Designed for BTC 1h–4h. Single-instrument focus; the live bot handles a
multi-coin universe upstream — this strategy is just the per-symbol brain.

Indicators registered in `init()`:
    main TF:  EMA(fast), EMA(slow), RSI, ATR
    HTF (4h): EMA200 (resampled manually for copy-on-write safety, then
              reindexed back to the main TF with ffill so each main-TF bar
              sees only the last fully-closed HTF bar)

Entry (when flat):
    LONG  := HTF bullish AND price has touched EMA_fast within
             `pullback_lookback` bars AND RSI crosses up through
             `rsi_threshold`.
    SHORT := mirror.

Exit:
    Hard SL/TP at entry (ATR-sized), then optional ATR trailing once profit
    >= `trail_trigger_atr × ATR`. Trailing only moves SL in the favorable
    direction.

Sizing & leverage: same convention as `stoch_state_machine` — runner sets
`margin = 1 / leverage`, and we pass `size = trade_amount / equity` so the
effective notional is `trade_amount × leverage`.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from backtesting import Strategy
from backtesting.lib import OHLCV_AGG

from app.core.indicators import atr, ema, rsi


# Minimum SL improvement (as fraction of current SL) to bother moving the
# trailing stop — avoids order spam in the live bot and noisy events in
# backtest trades.
_MIN_BUMP_PCT = 0.1


class BTCTrendPullback(Strategy):
    # ---------------------------- HTF trend filter
    htf_timeframe: str = "4h"
    htf_ema: int = 200

    # ---------------------------- Main-TF pullback & momentum
    ema_fast: int = 20
    ema_slow: int = 50
    rsi_period: int = 14
    rsi_threshold: float = 50.0
    pullback_lookback: int = 5  # bars to remember the last EMA-fast touch

    # ---------------------------- ATR risk
    atr_period: int = 14
    k_sl_atr: float = 2.0
    k_tp_atr: float = 3.0
    k_trail_atr: float = 2.0
    trail_trigger_atr: float = 1.5
    trailing_enabled: bool = True

    # ---------------------------- Sizing & side
    trade_amount: float = 100.0
    leverage: int = 5
    long_enabled: bool = True
    short_enabled: bool = True

    # ------------------------------------------------------------------ init

    def init(self):
        close = self.data.Close
        high = self.data.High
        low = self.data.Low

        self._ema_fast = self.I(ema, close, self.ema_fast, name=f"EMA({self.ema_fast})")
        self._ema_slow = self.I(ema, close, self.ema_slow, name=f"EMA({self.ema_slow})")
        self._rsi = self.I(rsi, close, self.rsi_period, name=f"RSI({self.rsi_period})")
        self._atr = self.I(
            atr, high, low, close, self.atr_period, "rma",
            name=f"ATR({self.atr_period})", plot=False,
        )

        # HTF EMA via manual resample. `resample_apply` produces read-only
        # buffers under pandas copy-on-write that FractionalBacktest then
        # tries to rescale in place — the manual path returns a contiguous,
        # writable numpy array.
        df_main = self.data.df
        df_htf = (
            df_main.resample(self.htf_timeframe, label="right", closed="right")
            .agg(OHLCV_AGG)
            .dropna()
        )
        htf_close = df_htf["Close"]
        htf_ema_arr = ema(htf_close, self.htf_ema)

        s = pd.Series(htf_ema_arr, index=df_htf.index)
        aligned = (
            s.reindex(df_main.index.union(df_htf.index), method="ffill")
            .reindex(df_main.index)
        )
        htf_ema_aligned = np.ascontiguousarray(aligned.to_numpy(dtype=float, copy=True))

        self._htf_ema = self.I(
            lambda a=htf_ema_aligned: a,
            name=f"EMA{self.htf_ema} {self.htf_timeframe}",
            overlay=True,
        )

        # Pullback tracking: bar index of the last touch of EMA fast.
        self._last_touch_bar: int = -10**9

    # ------------------------------------------------------------------ main loop

    def next(self):
        i = len(self.data) - 1
        if i < 2:
            return

        close_now = float(self.data.Close[-1])
        close_prev = float(self.data.Close[-2])
        high_now = float(self.data.High[-1])
        low_now = float(self.data.Low[-1])

        ema_fast_now = float(self._ema_fast[-1])
        ema_fast_prev = float(self._ema_fast[-2])
        htf_ema_now = float(self._htf_ema[-1])
        rsi_now = float(self._rsi[-1])
        rsi_prev = float(self._rsi[-2])
        atr_now = float(self._atr[-1])

        # Warm-up guard.
        if any(math.isnan(v) for v in (
            ema_fast_now, ema_fast_prev, htf_ema_now,
            rsi_now, rsi_prev, atr_now,
        )):
            return

        # 1) Update pullback-touch tracker. A "touch" = the bar's range
        #    straddles EMA fast, OR price crossed it since last bar.
        touched = (
            low_now <= ema_fast_now <= high_now
            or (close_prev < ema_fast_prev and close_now >= ema_fast_now)
            or (close_prev > ema_fast_prev and close_now <= ema_fast_now)
        )
        if touched:
            self._last_touch_bar = i

        # 2) Trailing-stop maintenance for any active position.
        if self.trailing_enabled and self.position:
            self._update_trailing(atr_now, close_now)

        # 3) Entries only when flat.
        if self.position:
            return

        htf_bull = close_now > htf_ema_now
        htf_bear = close_now < htf_ema_now
        recent_touch = (i - self._last_touch_bar) <= self.pullback_lookback

        if not recent_touch or atr_now <= 0.0:
            return

        rsi_cross_up = rsi_prev < self.rsi_threshold <= rsi_now
        rsi_cross_down = rsi_prev > self.rsi_threshold >= rsi_now

        if htf_bull and self.long_enabled and rsi_cross_up:
            self._open("LONG", close_now, atr_now)
        elif htf_bear and self.short_enabled and rsi_cross_down:
            self._open("SHORT", close_now, atr_now)

    # ------------------------------------------------------------------ helpers

    def _open(self, side: str, price: float, atr_now: float) -> None:
        equity = float(self.equity)
        size_fraction = self.trade_amount / equity if equity > 0 else 0.0
        size_fraction = max(1e-6, min(0.9999, size_fraction))

        sl_dist = self.k_sl_atr * atr_now
        tp_dist = self.k_tp_atr * atr_now

        if side == "LONG":
            sl = price - sl_dist
            tp = price + tp_dist
            # Defensive bounds: SL must be < price < TP for backtesting.py.
            if sl <= 0 or sl >= price or tp <= price:
                return
            self.buy(size=size_fraction, sl=sl, tp=tp)
        else:
            sl = price + sl_dist
            tp = price - tp_dist
            if tp <= 0 or sl <= price or tp >= price:
                return
            self.sell(size=size_fraction, sl=sl, tp=tp)

    def _update_trailing(self, atr_now: float, mark: float) -> None:
        """ATR-based trailing stop. Lock SL to `mark ∓ k_trail × ATR` once
        the trade is at least `trail_trigger_atr × ATR` in our favor. Only
        tighten — never loosen — and require a minimum bump to avoid churn.
        """
        if not self.trades or atr_now <= 0:
            return
        trade = self.trades[-1]
        entry = float(trade.entry_price)
        is_long = bool(trade.is_long)
        trigger = self.trail_trigger_atr * atr_now
        offset = self.k_trail_atr * atr_now

        if is_long:
            if (mark - entry) < trigger:
                return
            desired = mark - offset
            current = float(trade.sl) if trade.sl is not None else 0.0
            if desired <= current:
                return
            improvement = (desired - current) / current * 100.0 if current > 0 else 100.0
            if improvement < _MIN_BUMP_PCT:
                return
            trade.sl = desired
        else:
            if (entry - mark) < trigger:
                return
            desired = mark + offset
            current = float(trade.sl) if trade.sl is not None else float("inf")
            if desired >= current:
                return
            improvement = (
                (current - desired) / current * 100.0 if current != float("inf") else 100.0
            )
            if improvement < _MIN_BUMP_PCT:
                return
            trade.sl = desired


# ------------------------------------------------------------------------ schema


SCHEMA = {
    # HTF trend filter
    "htf_timeframe": {
        "type": "select",
        "default": "4h",
        "label": "HTF timeframe",
        "group": "HTF Trend",
        "options": [
            {"value": "1h", "label": "1h"},
            {"value": "2h", "label": "2h"},
            {"value": "4h", "label": "4h"},
            {"value": "1d", "label": "1d"},
        ],
    },
    "htf_ema": {"type": "int", "default": 200, "min": 20, "max": 400, "label": "HTF EMA period", "group": "HTF Trend"},
    # Pullback entry
    "ema_fast": {"type": "int", "default": 20, "min": 5, "max": 100, "label": "EMA fast (pullback target)", "group": "Pullback Entry"},
    "ema_slow": {"type": "int", "default": 50, "min": 10, "max": 200, "label": "EMA slow (context)", "group": "Pullback Entry"},
    "rsi_period": {"type": "int", "default": 14, "min": 2, "max": 50, "label": "RSI period", "group": "Pullback Entry"},
    "rsi_threshold": {"type": "float", "default": 50.0, "min": 30.0, "max": 70.0, "label": "RSI cross threshold", "group": "Pullback Entry"},
    "pullback_lookback": {"type": "int", "default": 5, "min": 1, "max": 30, "label": "Pullback lookback (bars)", "group": "Pullback Entry"},
    # Risk
    "atr_period": {"type": "int", "default": 14, "min": 2, "max": 100, "label": "ATR period", "group": "Risk (ATR)"},
    "k_sl_atr": {"type": "float", "default": 2.0, "min": 0.5, "max": 6.0, "label": "SL × ATR", "group": "Risk (ATR)"},
    "k_tp_atr": {"type": "float", "default": 3.0, "min": 0.5, "max": 10.0, "label": "TP × ATR", "group": "Risk (ATR)"},
    # Trailing
    "trailing_enabled": {"type": "bool", "default": True, "label": "Enable ATR trailing", "group": "Trailing"},
    "trail_trigger_atr": {"type": "float", "default": 1.5, "min": 0.1, "max": 5.0, "label": "Trigger × ATR", "group": "Trailing"},
    "k_trail_atr": {"type": "float", "default": 2.0, "min": 0.5, "max": 6.0, "label": "Trailing offset × ATR", "group": "Trailing"},
    # Sizing
    "trade_amount": {"type": "float", "default": 100.0, "min": 1.0, "max": 1_000_000.0, "label": "Trade amount (USDT margin)", "group": "Sizing"},
    "leverage": {"type": "int", "default": 5, "min": 1, "max": 125, "label": "Leverage (x)", "group": "Sizing"},
    # Side
    "long_enabled": {"type": "bool", "default": True, "label": "Enable LONG", "group": "Side"},
    "short_enabled": {"type": "bool", "default": True, "label": "Enable SHORT", "group": "Side"},
}
