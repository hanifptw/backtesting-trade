"""Regime-Adaptive Multi-Strategy.

Mendeteksi regime market (BULL / BEAR / RANGE / NOTRADE) menggunakan
EMA50/EMA200 + ADX, lalu pilih sub-strategy sesuai regime:

    BULL    : LONG-only. Trigger = Stoch K < oversold dan harga recently
              retrace ke EMA50. SL di bawah swing-low (1.5× ATR di luar).
              TP via RR ratio + trailing 2× ATR setelah profit 1× ATR.
    BEAR    : mirror of BULL untuk SHORT.
    RANGE   : trade kedua sisi pada level S/R yang dikunci saat masuk
              regime (snapshot 20 bar terakhir). Tight SL di luar range.
              TP di sisi berlawanan.
    NOTRADE : tutup posisi terbuka, tidak buka entry baru.

Leverage per regime dicapai lewat position sizing (margin global = 1/cap;
size_fraction = leverage_for_regime / cap). Cap default 10×.

Visualisasi regime memakai pipeline REGIME_BG yang sudah ada di runner.py
+ PriceChart.tsx — 4 background colored rectangles otomatis muncul.
"""

from __future__ import annotations

import math

import numpy as np
from backtesting import Strategy

from app.core.indicators import adx, atr, ema, stoch_d_line, stoch_k_line


_BULL = "BULL"
_BEAR = "BEAR"
_RANGE = "RANGE"
_NOTRADE = "NOTRADE"

# Index here = numeric code stored in regime_codes; order matters.
_REGIME_META: list[dict] = [
    {"id": _BULL,    "label": "Bull",    "color": "mediumseagreen"},
    {"id": _BEAR,    "label": "Bear",    "color": "tomato"},
    {"id": _RANGE,   "label": "Range",   "color": "silver"},
    {"id": _NOTRADE, "label": "NoTrade", "color": "dimgray"},
]


class RegimeAdaptiveStrategy(Strategy):
    # --- Indicator periods ------------------------------------------------
    ema_fast: int = 50
    ema_slow: int = 200
    adx_period: int = 14
    atr_period: int = 14
    rsi_period: int = 14
    stoch_k_period: int = 21
    stoch_smooth: int = 3
    stoch_d_period: int = 3

    # --- Regime thresholds ------------------------------------------------
    adx_trend_threshold: float = 25.0
    adx_range_threshold: float = 20.0

    # --- Entry filters ----------------------------------------------------
    ema_retrace_proximity_pct: float = 1.0
    ema_retrace_lookback: int = 5
    sr_proximity_pct: float = 1.0
    swing_lookback: int = 10
    range_lookback: int = 20
    stoch_oversold: float = 20.0
    stoch_overbought: float = 80.0

    # --- Risk -------------------------------------------------------------
    atr_sl_mult: float = 1.5
    rr_ratio: float = 3.0
    range_sl_buffer_pct: float = 1.0

    # --- Trailing ---------------------------------------------------------
    trailing_enabled: bool = True
    trailing_activation_atr_mult: float = 1.0
    trailing_distance_atr_mult: float = 2.0

    # --- Per-regime leverage ---------------------------------------------
    bull_leverage: float = 5.0
    bear_leverage: float = 3.0
    range_leverage: float = 3.0

    # Cap consumed by runner.py to set Backtest(margin = 1/leverage). Per-trade
    # size = regime_leverage / leverage cap → effective per-regime leverage.
    leverage: float = 10.0

    # ------------------------------------------------------------------ init

    def init(self) -> None:
        close = np.asarray(self.data.Close, dtype=float)
        high = np.asarray(self.data.High, dtype=float)
        low = np.asarray(self.data.Low, dtype=float)

        # Indicators (some overlay on price, some in oscillator pane)
        self._ema_fast = self.I(
            ema, close, self.ema_fast,
            name=f"EMA({self.ema_fast})",
            overlay=True,
            color="#3b82f6",
        )
        self._ema_slow = self.I(
            ema, close, self.ema_slow,
            name=f"EMA({self.ema_slow})",
            overlay=True,
            color="#f59e0b",
        )
        self._adx = self.I(
            adx, high, low, close, self.adx_period,
            name=f"ADX({self.adx_period})",
        )
        self._atr = self.I(
            atr, high, low, close, self.atr_period,
            name=f"ATR({self.atr_period})",
            plot=False,
        )
        self._k = self.I(
            stoch_k_line, close,
            self.rsi_period, self.stoch_k_period, self.stoch_smooth, self.stoch_d_period,
            name=f"%K({self.stoch_k_period})",
        )
        self._d = self.I(
            stoch_d_line, close,
            self.rsi_period, self.stoch_k_period, self.stoch_smooth, self.stoch_d_period,
            name=f"%D({self.stoch_d_period})",
        )

        # Precompute regime per bar (vectorized)
        ema_f = np.asarray(self._ema_fast)
        ema_s = np.asarray(self._ema_slow)
        adx_v = np.asarray(self._adx)
        n = len(close)

        valid = np.isfinite(ema_f) & np.isfinite(ema_s) & np.isfinite(adx_v)
        bull_mask = valid & (close > ema_s) & (ema_f > ema_s) & (adx_v > self.adx_trend_threshold)
        bear_mask = valid & (close < ema_s) & (ema_f < ema_s) & (adx_v > self.adx_trend_threshold)
        range_mask = valid & (adx_v < self.adx_range_threshold) & ~bull_mask & ~bear_mask
        notrade_mask = valid & ~(bull_mask | bear_mask | range_mask)

        regime_codes = np.full(n, np.nan, dtype=float)
        regime_codes[bull_mask] = 0.0
        regime_codes[bear_mask] = 1.0
        regime_codes[range_mask] = 2.0
        regime_codes[notrade_mask] = 3.0

        # Hidden combined indicator — used in next() via [-1] indexing
        self._regime = self.I(
            lambda x: x,
            regime_codes.copy(),
            name="Regime (0=Bull,1=Bear,2=Range,3=NoTrade)",
            overlay=False,
            plot=False,
        )

        # 4 REGIME_BG bands — renderer-recognized name pattern
        for code, meta in enumerate(_REGIME_META):
            band = np.where(
                np.isfinite(regime_codes) & (regime_codes == float(code)),
                float(code),
                np.nan,
            )
            self.I(
                lambda x: x,
                band.copy(),
                name=f"REGIME_BG:{meta['color']}:{meta['label']}",
                overlay=False,
                plot=True,
            )

        # Stateful per-bar tracking
        self._prev_regime: str | None = None
        self._range_high: float | None = None
        self._range_low: float | None = None
        self._entry_atr: float | None = None

    # ------------------------------------------------------------------ main

    def next(self) -> None:
        # Reset entry context when no position is open (SL/TP fill cleared it).
        if not self.position:
            self._entry_atr = None

        regime = self._current_regime()
        price = float(self.data.Close[-1])
        atr_raw = float(self._atr[-1])
        atr_now = atr_raw if not math.isnan(atr_raw) else None

        # 1. Range S/R locking — snapshot levels on entry into RANGE
        if regime == _RANGE and self._prev_regime != _RANGE:
            n = min(self.range_lookback, len(self.data.High))
            self._range_high = float(np.max(np.asarray(self.data.High[-n:], dtype=float)))
            self._range_low = float(np.min(np.asarray(self.data.Low[-n:], dtype=float)))
        elif regime != _RANGE:
            self._range_high = None
            self._range_low = None
        self._prev_regime = regime

        # 2. NOTRADE → flatten and skip
        if regime == _NOTRADE:
            if self.position:
                self.position.close()
                self._entry_atr = None
            return

        # 3. Trailing for any open position
        if self.position and self.trailing_enabled and self._entry_atr is not None:
            self._update_trailing(price)

        # 4. Entries (skip if already in a position)
        if self.position:
            return
        if atr_now is None or atr_now <= 0:
            return

        if regime == _BULL:
            self._try_bull_entry(price, atr_now)
        elif regime == _BEAR:
            self._try_bear_entry(price, atr_now)
        elif regime == _RANGE:
            self._try_range_entry(price, atr_now)

    # --------------------------------------------------------------- helpers

    def _current_regime(self) -> str:
        code = float(self._regime[-1])
        if math.isnan(code):
            return _NOTRADE
        c = int(code)
        if c == 0:
            return _BULL
        if c == 1:
            return _BEAR
        if c == 2:
            return _RANGE
        return _NOTRADE

    def _update_trailing(self, price: float) -> None:
        if not self.trades or self._entry_atr is None or self._entry_atr <= 0:
            return
        trade = self.trades[-1]
        entry = float(trade.entry_price)
        atr_entry = float(self._entry_atr)
        is_long = bool(trade.is_long)

        profit = (price - entry) if is_long else (entry - price)
        if profit < self.trailing_activation_atr_mult * atr_entry:
            return

        trail_dist = self.trailing_distance_atr_mult * atr_entry
        if is_long:
            new_sl = price - trail_dist
            current_sl = float(trade.sl) if trade.sl is not None else 0.0
            if new_sl > current_sl:
                trade.sl = new_sl
        else:
            new_sl = price + trail_dist
            current_sl = float(trade.sl) if trade.sl is not None else float("inf")
            if new_sl < current_sl:
                trade.sl = new_sl

    def _ema_retraced_recent(self) -> bool:
        """True if any of the last N bars closed within X% of EMA-fast."""
        n_max = min(self.ema_retrace_lookback, len(self.data.Close))
        if n_max < 1:
            return False
        thresh = self.ema_retrace_proximity_pct / 100.0
        ema_f = np.asarray(self._ema_fast)
        for i in range(1, n_max + 1):
            c = float(self.data.Close[-i])
            ef = float(ema_f[-i])
            if math.isnan(ef) or ef <= 0:
                continue
            if abs(c - ef) / ef <= thresh:
                return True
        return False

    def _size_fraction(self, regime_leverage: float) -> float:
        cap = float(self.leverage) if self.leverage else 1.0
        size = float(regime_leverage) / cap
        return max(1e-6, min(0.9999, size))

    # ----------------------------------------------------- per-regime entries

    def _try_bull_entry(self, price: float, atr_now: float) -> None:
        ema_f = float(self._ema_fast[-1])
        ema_s = float(self._ema_slow[-1])
        adx_v = float(self._adx[-1])
        k_now = float(self._k[-1])
        d_now = float(self._d[-1])
        if any(math.isnan(v) for v in (ema_f, ema_s, adx_v, k_now, d_now)):
            return

        # Regime filter (defensive — regime codes already gate this)
        if not (price > ema_s and ema_f > ema_s and adx_v > self.adx_trend_threshold):
            return
        # Trigger: Stoch oversold + recent retrace to EMA-fast
        if k_now >= self.stoch_oversold:
            return
        if not self._ema_retraced_recent():
            return

        n = min(self.swing_lookback, len(self.data.Low))
        swing_low = float(np.min(np.asarray(self.data.Low[-n:], dtype=float)))
        sl = swing_low - self.atr_sl_mult * atr_now
        if sl >= price:
            return  # invalid SL — swing low already above current price
        tp = price + self.rr_ratio * (price - sl)

        tag = {
            "k": round(k_now, 2),
            "d": round(d_now, 2),
            "regime": _BULL,
            "atr": round(atr_now, 6),
        }
        self.buy(size=self._size_fraction(self.bull_leverage), sl=sl, tp=tp, tag=tag)
        self._entry_atr = atr_now

    def _try_bear_entry(self, price: float, atr_now: float) -> None:
        ema_f = float(self._ema_fast[-1])
        ema_s = float(self._ema_slow[-1])
        adx_v = float(self._adx[-1])
        k_now = float(self._k[-1])
        d_now = float(self._d[-1])
        if any(math.isnan(v) for v in (ema_f, ema_s, adx_v, k_now, d_now)):
            return

        if not (price < ema_s and ema_f < ema_s and adx_v > self.adx_trend_threshold):
            return
        if k_now <= self.stoch_overbought:
            return
        if not self._ema_retraced_recent():
            return

        n = min(self.swing_lookback, len(self.data.High))
        swing_high = float(np.max(np.asarray(self.data.High[-n:], dtype=float)))
        sl = swing_high + self.atr_sl_mult * atr_now
        if sl <= price:
            return
        tp = price - self.rr_ratio * (sl - price)
        if tp <= 0:
            return

        tag = {
            "k": round(k_now, 2),
            "d": round(d_now, 2),
            "regime": _BEAR,
            "atr": round(atr_now, 6),
        }
        self.sell(size=self._size_fraction(self.bear_leverage), sl=sl, tp=tp, tag=tag)
        self._entry_atr = atr_now

    def _try_range_entry(self, price: float, atr_now: float) -> None:
        if self._range_high is None or self._range_low is None:
            return
        if self._range_high <= self._range_low:
            return
        k_now = float(self._k[-1])
        d_now = float(self._d[-1])
        if math.isnan(k_now) or math.isnan(d_now):
            return

        thresh = self.sr_proximity_pct / 100.0
        size = self._size_fraction(self.range_leverage)

        # Near support → BUY
        if (
            abs(price - self._range_low) / self._range_low <= thresh
            and k_now < self.stoch_oversold
        ):
            sl = self._range_low * (1.0 - self.range_sl_buffer_pct / 100.0)
            tp = self._range_high
            if sl < price < tp:
                tag = {
                    "k": round(k_now, 2),
                    "d": round(d_now, 2),
                    "regime": _RANGE,
                    "atr": round(atr_now, 6),
                }
                self.buy(size=size, sl=sl, tp=tp, tag=tag)
                self._entry_atr = atr_now
            return

        # Near resistance → SELL
        if (
            abs(price - self._range_high) / self._range_high <= thresh
            and k_now > self.stoch_overbought
        ):
            sl = self._range_high * (1.0 + self.range_sl_buffer_pct / 100.0)
            tp = self._range_low
            if tp < price < sl:
                tag = {
                    "k": round(k_now, 2),
                    "d": round(d_now, 2),
                    "regime": _RANGE,
                    "atr": round(atr_now, 6),
                }
                self.sell(size=size, sl=sl, tp=tp, tag=tag)
                self._entry_atr = atr_now


# ----------------------------------------------------------------- schema

SCHEMA: dict = {
    # Indicator periods
    "ema_fast": {"type": "int", "default": 50, "min": 5, "max": 200, "label": "EMA fast period", "group": "Indicator"},
    "ema_slow": {"type": "int", "default": 200, "min": 20, "max": 500, "label": "EMA slow period", "group": "Indicator"},
    "adx_period": {"type": "int", "default": 14, "min": 5, "max": 50, "label": "ADX period", "group": "Indicator"},
    "atr_period": {"type": "int", "default": 14, "min": 5, "max": 50, "label": "ATR period", "group": "Indicator"},
    "rsi_period": {"type": "int", "default": 14, "min": 5, "max": 50, "label": "RSI base period (Stoch RSI)", "group": "Indicator"},
    "stoch_k_period": {"type": "int", "default": 21, "min": 5, "max": 50, "label": "Stoch %K period", "group": "Indicator"},
    "stoch_smooth": {"type": "int", "default": 3, "min": 1, "max": 10, "label": "Stoch %K smoothing", "group": "Indicator"},
    "stoch_d_period": {"type": "int", "default": 3, "min": 1, "max": 10, "label": "Stoch %D smoothing", "group": "Indicator"},

    # Regime thresholds
    "adx_trend_threshold": {"type": "float", "default": 25.0, "min": 5.0, "max": 60.0, "label": "ADX > X → trend regime", "group": "Regime"},
    "adx_range_threshold": {"type": "float", "default": 20.0, "min": 5.0, "max": 40.0, "label": "ADX < X → range regime", "group": "Regime"},

    # Entry filters
    "ema_retrace_proximity_pct": {"type": "float", "default": 1.0, "min": 0.1, "max": 5.0, "label": "Retrace ke EMA-fast proximity (%)", "group": "Entry"},
    "ema_retrace_lookback": {"type": "int", "default": 5, "min": 1, "max": 30, "label": "Retrace lookback (bars)", "group": "Entry"},
    "sr_proximity_pct": {"type": "float", "default": 1.0, "min": 0.1, "max": 5.0, "label": "Range S/R proximity (%)", "group": "Entry"},
    "swing_lookback": {"type": "int", "default": 10, "min": 3, "max": 50, "label": "Swing low/high lookback (bars)", "group": "Entry"},
    "range_lookback": {"type": "int", "default": 20, "min": 5, "max": 100, "label": "Range S/R lookback on entry (bars)", "group": "Entry"},
    "stoch_oversold": {"type": "float", "default": 20.0, "min": 1.0, "max": 49.0, "label": "Stoch oversold", "group": "Entry"},
    "stoch_overbought": {"type": "float", "default": 80.0, "min": 51.0, "max": 99.0, "label": "Stoch overbought", "group": "Entry"},

    # Risk
    "atr_sl_mult": {"type": "float", "default": 1.5, "min": 0.5, "max": 5.0, "label": "SL = N × ATR di luar swing", "group": "Risk"},
    "rr_ratio": {"type": "float", "default": 3.0, "min": 1.0, "max": 10.0, "label": "TP risk:reward ratio (trend)", "group": "Risk"},
    "range_sl_buffer_pct": {"type": "float", "default": 1.0, "min": 0.1, "max": 3.0, "label": "Range SL buffer di luar level (%)", "group": "Risk"},

    # Trailing
    "trailing_enabled": {"type": "bool", "default": True, "label": "Enable trailing stop", "group": "Trailing"},
    "trailing_activation_atr_mult": {"type": "float", "default": 1.0, "min": 0.0, "max": 5.0, "label": "Activate trailing setelah profit N × ATR_entry", "group": "Trailing"},
    "trailing_distance_atr_mult": {"type": "float", "default": 2.0, "min": 0.5, "max": 10.0, "label": "Trail distance N × ATR_entry", "group": "Trailing"},

    # Per-regime leverage
    "bull_leverage": {"type": "float", "default": 5.0, "min": 1.0, "max": 10.0, "label": "Leverage di BULL", "group": "Leverage"},
    "bear_leverage": {"type": "float", "default": 3.0, "min": 1.0, "max": 10.0, "label": "Leverage di BEAR", "group": "Leverage"},
    "range_leverage": {"type": "float", "default": 3.0, "min": 1.0, "max": 10.0, "label": "Leverage di RANGE", "group": "Leverage"},
}
