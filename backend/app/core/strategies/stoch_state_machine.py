"""Stochastic Oscillator State-Machine Strategy.

Port of the user's live Binance Futures bot (see [STRATEGY.md](../../../STRATEGY.md))
to the backtesting.py framework. Logic mirrors STRATEGY.md §3–§8 exactly; only
operational concerns (Binance algoOrder quirks, mark-price retry, LOT_SIZE
quantization, in-progress-bar watermark, reconciliation TTL) are dropped
because the engine handles them natively.

State machine:
    IDLE ──K crosses UP through OS  (k_prev < OS ≤ k_now)──► IN_LONG  ┐
    IDLE ──K crosses DOWN through OB (k_prev > OB ≥ k_now)─► IN_SHORT ┤
                                                                       │
    IN_LONG / IN_SHORT exit ONLY via SL/TP (or trailing/auto-reverse) ─┘

Entry triggers purely on %K crossing the oversold/overbought thresholds.
No D-confirmation, no ARMED waiting state — fires on the first bar where
the cross is detected.

Sizing (STRATEGY.md §5):
    notional = trade_amount × leverage
    Caller (runner.py) sets `margin = 1 / leverage` on the Backtest.
    Strategy passes `size = trade_amount / equity` (fraction of margin).

Trailing (STRATEGY.md §6) and auto-reverse on SL (STRATEGY.md §8) are toggleable.
"""

from __future__ import annotations

import math

import numpy as np
from backtesting import Strategy

from app.core.indicators import stoch_d_line, stoch_k_line


# State machine states
IDLE = "IDLE"
IN_LONG = "IN_LONG"
IN_SHORT = "IN_SHORT"

# Minimum SL improvement (%) to actually move trailing stop — avoids API spam in live bot.
# Kept here for parity; in backtest it just avoids redundant order events in trades.
_MIN_BUMP_PCT = 0.1

# Tolerance for matching exit_price to recorded SL (as fraction of SL).
# SL fills should match exactly in backtest, but allow a tiny rounding window.
_SL_MATCH_TOL = 1e-4


class StochStateMachine(Strategy):
    # Indicator params — match TradingView/Binance "Stoch RSI" defaults
    rsi_period: int = 14       # LengthRSI
    stoch_k: int = 14          # LengthStoch
    stoch_smooth: int = 3      # SmoothK
    stoch_d: int = 3           # SmoothD
    oversold_level: float = 20.0
    overbought_level: float = 80.0

    # Side toggle
    long_enabled: bool = True
    short_enabled: bool = True

    # Sizing — leverage also consumed by runner.py to set Backtest(margin=1/leverage)
    trade_amount: float = 100.0
    leverage: int = 5

    # Risk
    sl_pct: float = 2.0
    tp_pct: float = 3.0

    # Trailing
    trailing_enabled: bool = False
    trailing_trigger_pct: float = 1.0
    trailing_offset_pct: float = 0.5

    # Auto-reverse
    auto_reverse_enabled: bool = False
    auto_reverse_max: int = 1

    def init(self):
        self._k = self.I(
            stoch_k_line,
            self.data.Close,
            self.rsi_period, self.stoch_k, self.stoch_smooth, self.stoch_d,
            name=f"%K({self.stoch_k})",
        )
        self._d = self.I(
            stoch_d_line,
            self.data.Close,
            self.rsi_period, self.stoch_k, self.stoch_smooth, self.stoch_d,
            name=f"%D({self.stoch_d})",
        )

        # State machine
        self._state: str = IDLE

        # Auto-reverse / SL-detection tracking
        self._closed_trades_seen: int = 0
        self._last_entry_side: str | None = None  # 'LONG' / 'SHORT'
        self._last_entry_sl: float | None = None
        self._reverse_chain: int = 0
        self._pending_reverse_side: str | None = None  # set after SL hit, executed on next bar

    # ------------------------------------------------------------------ main loop

    def next(self):
        # Need 2 valid bars to detect a cross.
        if len(self._k) < 2:
            return
        k_now, k_prev = float(self._k[-1]), float(self._k[-2])
        d_now, d_prev = float(self._d[-1]), float(self._d[-2])
        if any(math.isnan(v) for v in (k_now, k_prev, d_now, d_prev)):
            return

        # 1) Reconcile closed trades from previous bar (SL/TP fills happen between bars).
        self._reconcile_closed_trades()

        # 2) If a reverse is pending (SL hit on previous bar), open opposite side now.
        if self._pending_reverse_side is not None:
            side = self._pending_reverse_side
            self._pending_reverse_side = None
            self._open_position(side, bypass_side_toggle=True)
            return

        # 3) Trailing stop update for any active position.
        if self.trailing_enabled and self.position:
            self._update_trailing()

        # 4) Entry: react to %K crossing the OS/OB thresholds.
        #    IN_LONG / IN_SHORT exit ONLY via SL/TP (or trailing / auto-reverse).
        if self._state == IDLE and not self.position:
            cross_up_os = k_prev < self.oversold_level <= k_now
            cross_down_ob = k_prev > self.overbought_level >= k_now

            if cross_up_os and self.long_enabled:
                self._open_position("LONG")
                self._state = IN_LONG
            elif cross_down_ob and self.short_enabled:
                self._open_position("SHORT")
                self._state = IN_SHORT

    # ------------------------------------------------------------------ helpers

    def _open_position(self, side: str, *, bypass_side_toggle: bool = False) -> None:
        """Open LONG or SHORT with sizing & SL/TP per STRATEGY.md §5 & §6.

        Side toggle (long_enabled/short_enabled) is honored unless this is an
        auto-reverse flip — in which case the original entry was already
        authorized so we bypass (matches bot behavior: STRATEGY.md §8).
        """
        if not bypass_side_toggle:
            if side == "LONG" and not self.long_enabled:
                return
            if side == "SHORT" and not self.short_enabled:
                return

        price = float(self.data.Close[-1])
        equity = float(self.equity)
        # size as a fraction of available margin (backtesting.py convention).
        # With margin = 1/leverage, putting size = trade_amount/equity yields
        # an effective notional of trade_amount × leverage.
        size_fraction = self.trade_amount / equity if equity > 0 else 0.0
        size_fraction = max(1e-6, min(0.9999, size_fraction))

        k_at_entry = round(float(self._k[-1]), 2)
        d_at_entry = round(float(self._d[-1]), 2)
        tag = {"k": k_at_entry, "d": d_at_entry}

        if side == "LONG":
            sl = price * (1.0 - self.sl_pct / 100.0)
            tp = price * (1.0 + self.tp_pct / 100.0)
            self.buy(size=size_fraction, sl=sl, tp=tp, tag=tag)
        else:
            sl = price * (1.0 + self.sl_pct / 100.0)
            tp = price * (1.0 - self.tp_pct / 100.0)
            self.sell(size=size_fraction, sl=sl, tp=tp, tag=tag)

        self._last_entry_side = side
        self._last_entry_sl = sl

    def _reconcile_closed_trades(self) -> None:
        """Detect newly-closed trades. Reset state. Possibly queue an auto-reverse.

        We compare exit_price to the SL we recorded on entry — if they match
        within tolerance, it's an SL hit. (Trailing updates also touch
        self._last_entry_sl so this stays accurate.)
        """
        n_closed = len(self.closed_trades)
        if n_closed <= self._closed_trades_seen:
            return

        # Inspect newly-closed trades. Usually 1, but handle batches safely.
        for trade in self.closed_trades[self._closed_trades_seen:]:
            self._state = IDLE
            self._armed_extreme_k = None

            sl_hit = self._is_sl_hit(trade)
            if (
                sl_hit
                and self.auto_reverse_enabled
                and self._reverse_chain < self.auto_reverse_max
                and self._last_entry_side is not None
            ):
                opposite = "SHORT" if self._last_entry_side == "LONG" else "LONG"
                self._pending_reverse_side = opposite
                self._reverse_chain += 1
            elif not sl_hit:
                # TP or manual close → reset chain (STRATEGY.md §8).
                self._reverse_chain = 0

        self._closed_trades_seen = n_closed

    def _is_sl_hit(self, trade) -> bool:
        if self._last_entry_sl is None:
            return False
        try:
            exit_price = float(trade.exit_price)
        except Exception:
            return False
        return abs(exit_price - self._last_entry_sl) / self._last_entry_sl < _SL_MATCH_TOL

    def _update_trailing(self) -> None:
        """Tiered trailing stop per STRATEGY.md §6.

        Once profit ≥ trailing_trigger_pct, lock in milestones at step
        `trailing_offset_pct`. M1 = breakeven, M2 = entry ± step, M3 = entry ± 2·step, ...
        New SL is placed only when strictly better than current AND improvement ≥ 0.1 %.
        """
        if not self.trades:
            return
        trade = self.trades[-1]
        entry = float(trade.entry_price)
        mark = float(self.data.Close[-1])
        is_long = bool(trade.is_long)

        profit_pct = (mark - entry) / entry * 100.0 if is_long else (entry - mark) / entry * 100.0
        if profit_pct < self.trailing_trigger_pct:
            return

        milestone = int((profit_pct - self.trailing_trigger_pct) / self.trailing_offset_pct)
        offset_pct = milestone * self.trailing_offset_pct

        if is_long:
            desired_sl = entry * (1.0 + offset_pct / 100.0)
            current_sl = float(trade.sl) if trade.sl is not None else 0.0
            if desired_sl <= current_sl:
                return
            improvement_pct = (
                (desired_sl - current_sl) / current_sl * 100.0 if current_sl > 0 else 100.0
            )
        else:
            desired_sl = entry * (1.0 - offset_pct / 100.0)
            current_sl = float(trade.sl) if trade.sl is not None else float("inf")
            if desired_sl >= current_sl:
                return
            improvement_pct = (
                (current_sl - desired_sl) / current_sl * 100.0 if current_sl != float("inf") else 100.0
            )

        if improvement_pct < _MIN_BUMP_PCT:
            return

        trade.sl = desired_sl
        self._last_entry_sl = desired_sl


# ------------------------------------------------------------------------ schema


SCHEMA = {
    # Indicator — Stoch RSI (matches TradingView/Binance "Stoch RSI" indicator)
    "rsi_period": {"type": "int", "default": 14, "min": 1, "max": 100, "label": "RSI period (LengthRSI)", "group": "Indicator"},
    "stoch_k": {"type": "int", "default": 14, "min": 1, "max": 100, "label": "Stoch period (LengthStoch)", "group": "Indicator"},
    "stoch_smooth": {"type": "int", "default": 3, "min": 1, "max": 20, "label": "Smooth K", "group": "Indicator"},
    "stoch_d": {"type": "int", "default": 3, "min": 1, "max": 20, "label": "Smooth D", "group": "Indicator"},
    "oversold_level": {"type": "float", "default": 20.0, "min": 1.0, "max": 49.0, "label": "Oversold level", "group": "Indicator"},
    "overbought_level": {"type": "float", "default": 80.0, "min": 51.0, "max": 99.0, "label": "Overbought level", "group": "Indicator"},
    # Side
    "long_enabled": {"type": "bool", "default": True, "label": "Enable LONG", "group": "Side"},
    "short_enabled": {"type": "bool", "default": True, "label": "Enable SHORT", "group": "Side"},
    # Sizing
    "trade_amount": {"type": "float", "default": 100.0, "min": 1.0, "max": 1_000_000.0, "label": "Trade amount (USDT margin)", "group": "Sizing"},
    "leverage": {"type": "int", "default": 5, "min": 1, "max": 125, "label": "Leverage (x)", "group": "Sizing"},
    # Risk
    "sl_pct": {"type": "float", "default": 2.0, "min": 0.1, "max": 20.0, "label": "Stop Loss %", "group": "Risk"},
    "tp_pct": {"type": "float", "default": 3.0, "min": 0.1, "max": 50.0, "label": "Take Profit %", "group": "Risk"},
    # Trailing
    "trailing_enabled": {"type": "bool", "default": False, "label": "Enable trailing stop", "group": "Trailing"},
    "trailing_trigger_pct": {"type": "float", "default": 1.0, "min": 0.1, "max": 20.0, "label": "Trailing trigger %", "group": "Trailing"},
    "trailing_offset_pct": {"type": "float", "default": 0.5, "min": 0.1, "max": 10.0, "label": "Trailing step %", "group": "Trailing"},
    # Auto-reverse
    "auto_reverse_enabled": {"type": "bool", "default": False, "label": "Auto-reverse on SL", "group": "Auto-reverse"},
    "auto_reverse_max": {"type": "int", "default": 1, "min": 0, "max": 10, "label": "Max consecutive reverses", "group": "Auto-reverse"},
}
