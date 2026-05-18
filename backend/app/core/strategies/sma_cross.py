"""SMA Crossover baseline strategy.

Long-only. Buy when fast SMA crosses above slow SMA, close on opposite cross.
SL/TP attached on entry.
"""

from __future__ import annotations

from backtesting import Strategy
from backtesting.lib import crossover

from app.core.indicators import sma


class SMACrossStrategy(Strategy):
    n1: int = 10
    n2: int = 30
    sl_pct: float = 2.0
    tp_pct: float = 3.0

    def init(self):
        self.sma_fast = self.I(sma, self.data.Close, self.n1, name=f"SMA({self.n1})")
        self.sma_slow = self.I(sma, self.data.Close, self.n2, name=f"SMA({self.n2})")

    def next(self):
        price = float(self.data.Close[-1])
        if crossover(self.sma_fast, self.sma_slow) and not self.position:
            self.buy(
                sl=price * (1 - self.sl_pct / 100.0),
                tp=price * (1 + self.tp_pct / 100.0),
            )
        elif crossover(self.sma_slow, self.sma_fast) and self.position:
            self.position.close()


SCHEMA = {
    "n1": {"type": "int", "default": 10, "min": 2, "max": 200, "label": "Fast SMA period"},
    "n2": {"type": "int", "default": 30, "min": 2, "max": 200, "label": "Slow SMA period"},
    "sl_pct": {"type": "float", "default": 2.0, "min": 0.1, "max": 20.0, "label": "Stop Loss %"},
    "tp_pct": {"type": "float", "default": 3.0, "min": 0.1, "max": 50.0, "label": "Take Profit %"},
}
