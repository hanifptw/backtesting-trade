"""Supertrend flip strategy.

Entries follow Supertrend direction flips. The Supertrend line is the primary
stop. Optional fixed SL/TP can be enabled as an extra risk layer.
"""

from __future__ import annotations

import math

from backtesting import Strategy

from app.core.indicators import (
    supertrend_bearish_line,
    supertrend_bullish_line,
    supertrend_direction,
    supertrend_line,
)


class SupertrendStrategy(Strategy):
    # Indicator
    atr_period: int = 10
    multiplier: float = 3.0
    atr_method: str = "rma"

    # Side toggle
    long_enabled: bool = True
    short_enabled: bool = True

    # Sizing / margin
    trade_amount: float = 100.0
    leverage_enabled: bool = True
    leverage: int = 5

    # Optional fixed SL/TP
    fixed_sl_tp_enabled: bool = False
    sl_pct: float = 2.0
    tp_pct: float = 3.0

    def init(self):
        label = f"Supertrend({self.atr_period}, {self.multiplier:g}, {self.atr_method})"
        self._st = self.I(
            supertrend_line,
            self.data.High,
            self.data.Low,
            self.data.Close,
            self.atr_period,
            self.multiplier,
            self.atr_method,
            name=label,
            plot=False,
        )
        self.I(
            supertrend_bullish_line,
            self.data.High,
            self.data.Low,
            self.data.Close,
            self.atr_period,
            self.multiplier,
            self.atr_method,
            name=f"{label} Bullish",
            overlay=True,
        )
        self.I(
            supertrend_bearish_line,
            self.data.High,
            self.data.Low,
            self.data.Close,
            self.atr_period,
            self.multiplier,
            self.atr_method,
            name=f"{label} Bearish",
            overlay=True,
        )
        self._direction = self.I(
            supertrend_direction,
            self.data.High,
            self.data.Low,
            self.data.Close,
            self.atr_period,
            self.multiplier,
            self.atr_method,
            name="Supertrend direction",
            plot=False,
        )

    def next(self):
        if len(self._direction) < 2:
            return

        line = float(self._st[-1])
        direction = float(self._direction[-1])
        prev_direction = float(self._direction[-2])
        if any(math.isnan(v) for v in (line, direction, prev_direction)):
            return

        if self.position:
            self._update_supertrend_stop(line)

        flipped_bullish = prev_direction < 0 and direction > 0
        flipped_bearish = prev_direction > 0 and direction < 0

        if flipped_bullish and self.long_enabled:
            if self.position and self.position.is_short:
                self.position.close()
            if not self.position or self.position.is_short:
                self._open_position("LONG", line)
        elif flipped_bearish and self.short_enabled:
            if self.position and self.position.is_long:
                self.position.close()
            if not self.position or self.position.is_long:
                self._open_position("SHORT", line)

    def _position_size(self) -> float:
        equity = float(self.equity)
        size_fraction = self.trade_amount / equity if equity > 0 else 0.0
        return max(1e-6, min(0.9999, size_fraction))

    def _open_position(self, side: str, supertrend_stop: float) -> None:
        price = float(self.data.Close[-1])
        sl, tp = self._initial_sl_tp(side, price, supertrend_stop)
        if sl is None:
            return

        size = self._position_size()
        if side == "LONG":
            self.buy(size=size, sl=sl, tp=tp)
        else:
            self.sell(size=size, sl=sl, tp=tp)

    def _initial_sl_tp(
        self,
        side: str,
        price: float,
        supertrend_stop: float,
    ) -> tuple[float | None, float | None]:
        if side == "LONG":
            stop_candidates = [supertrend_stop] if 0 < supertrend_stop < price else []
            if self.fixed_sl_tp_enabled:
                fixed_sl = price * (1.0 - self.sl_pct / 100.0)
                if 0 < fixed_sl < price:
                    stop_candidates.append(fixed_sl)
                tp = price * (1.0 + self.tp_pct / 100.0)
            else:
                tp = None
            return (max(stop_candidates) if stop_candidates else None, tp)

        stop_candidates = [supertrend_stop] if supertrend_stop > price else []
        if self.fixed_sl_tp_enabled:
            fixed_sl = price * (1.0 + self.sl_pct / 100.0)
            if fixed_sl > price:
                stop_candidates.append(fixed_sl)
            tp = price * (1.0 - self.tp_pct / 100.0)
        else:
            tp = None
        return (min(stop_candidates) if stop_candidates else None, tp)

    def _update_supertrend_stop(self, supertrend_stop: float) -> None:
        if not self.trades or not math.isfinite(supertrend_stop):
            return

        trade = self.trades[-1]
        price = float(self.data.Close[-1])

        if trade.is_long:
            if not (0 < supertrend_stop < price):
                return
            current_sl = float(trade.sl) if trade.sl is not None else 0.0
            if supertrend_stop > current_sl:
                trade.sl = supertrend_stop
        else:
            if supertrend_stop <= price:
                return
            current_sl = float(trade.sl) if trade.sl is not None else float("inf")
            if supertrend_stop < current_sl:
                trade.sl = supertrend_stop


SCHEMA = {
    # Indicator
    "atr_period": {"type": "int", "default": 10, "min": 1, "max": 200, "label": "ATR period", "group": "Indicator"},
    "multiplier": {"type": "float", "default": 3.0, "min": 0.1, "max": 20.0, "label": "Multiplier", "group": "Indicator"},
    "atr_method": {
        "type": "select",
        "default": "rma",
        "label": "ATR method",
        "group": "Indicator",
        "options": [
            {"value": "rma", "label": "Wilder/RMA"},
            {"value": "sma", "label": "SMA"},
            {"value": "ema", "label": "EMA"},
        ],
    },
    # Side
    "long_enabled": {"type": "bool", "default": True, "label": "Enable LONG", "group": "Side"},
    "short_enabled": {"type": "bool", "default": True, "label": "Enable SHORT", "group": "Side"},
    # Sizing
    "trade_amount": {"type": "float", "default": 100.0, "min": 1.0, "max": 1_000_000.0, "label": "Trade amount (USDT margin)", "group": "Sizing"},
    "leverage_enabled": {"type": "bool", "default": True, "label": "Enable leverage", "group": "Sizing"},
    "leverage": {"type": "int", "default": 5, "min": 1, "max": 125, "label": "Leverage (x)", "group": "Sizing"},
    # Optional fixed SL/TP
    "fixed_sl_tp_enabled": {"type": "bool", "default": False, "label": "Enable fixed SL/TP", "group": "Risk"},
    "sl_pct": {"type": "float", "default": 2.0, "min": 0.1, "max": 20.0, "label": "Fixed Stop Loss %", "group": "Risk"},
    "tp_pct": {"type": "float", "default": 3.0, "min": 0.1, "max": 50.0, "label": "Fixed Take Profit %", "group": "Risk"},
}
