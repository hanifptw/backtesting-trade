"""RSI Mean Reversion baseline strategy.

Long-only. Buy when RSI dips into oversold zone (no position yet),
close when RSI rises into overbought zone. SL/TP attached on entry.
"""

from __future__ import annotations

from backtesting import Strategy

from app.core.indicators import rsi


class RSIStrategy(Strategy):
    period: int = 14
    oversold: float = 30.0
    overbought: float = 70.0
    sl_pct: float = 2.0
    tp_pct: float = 3.0

    def init(self):
        self.rsi = self.I(rsi, self.data.Close, self.period, name=f"RSI({self.period})")

    def next(self):
        price = float(self.data.Close[-1])
        current = self.rsi[-1]
        if current < self.oversold and not self.position:
            self.buy(
                sl=price * (1 - self.sl_pct / 100.0),
                tp=price * (1 + self.tp_pct / 100.0),
            )
        elif current > self.overbought and self.position:
            self.position.close()


SCHEMA = {
    "period": {"type": "int", "default": 14, "min": 2, "max": 100, "label": "RSI period"},
    "oversold": {"type": "float", "default": 30.0, "min": 1.0, "max": 49.0, "label": "Oversold level"},
    "overbought": {"type": "float", "default": 70.0, "min": 51.0, "max": 99.0, "label": "Overbought level"},
    "sl_pct": {"type": "float", "default": 2.0, "min": 0.1, "max": 20.0, "label": "Stop Loss %"},
    "tp_pct": {"type": "float", "default": 3.0, "min": 0.1, "max": 50.0, "label": "Take Profit %"},
}
