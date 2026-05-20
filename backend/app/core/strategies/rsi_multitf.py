"""RSI Multi-Timeframe strategy.

Entry LONG saat RSI daily & RSI weekly keduanya > level (default 70)
dengan RSI weekly lebih tinggi dari RSI daily, dan price berada di atas
MA bertingkat (MA10 > MA20 > MA50 > MA100).

Exit via:
- SL otomatis sl_pct% dari harga entry (dipasang saat buy)
- Kondisi: price < (1 - ma_exit_gap_pct/100) * MA10

Catatan warm-up: weekly RSI period 30 butuh ≥ 30 minggu (~7 bulan) data
agar sinyal pertama muncul. Jika data terlalu pendek, tidak akan ada trade.
"""

from __future__ import annotations

from backtesting import Strategy
from backtesting.lib import resample_apply

from app.core.indicators import rsi, sma


class RSIMultiTF(Strategy):
    d_rsi: int = 30
    w_rsi: int = 30
    level: float = 70.0
    sl_pct: float = 8.0
    ma_exit_gap_pct: float = 2.0

    def init(self):
        close = self.data.Close

        self.daily_rsi = resample_apply(
            "D", rsi, close, self.d_rsi, name=f"RSI_D({self.d_rsi})"
        )
        self.weekly_rsi = resample_apply(
            "W", rsi, close, self.w_rsi, name=f"RSI_W({self.w_rsi})"
        )

        self.ma10  = self.I(sma, close, 10,  name="MA10")
        self.ma20  = self.I(sma, close, 20,  name="MA20")
        self.ma50  = self.I(sma, close, 50,  name="MA50")
        self.ma100 = self.I(sma, close, 100, name="MA100")

    def next(self):
        price = float(self.data.Close[-1])

        if not self.position:
            if (
                self.daily_rsi[-1] > self.level
                and self.weekly_rsi[-1] > self.level
                and self.weekly_rsi[-1] > self.daily_rsi[-1]
                and self.ma10[-1] > self.ma20[-1]
                and self.ma20[-1] > self.ma50[-1]
                and self.ma50[-1] > self.ma100[-1]
                and price > self.ma10[-1]
            ):
                self.buy(sl=price * (1.0 - self.sl_pct / 100.0))

        elif price < (1.0 - self.ma_exit_gap_pct / 100.0) * self.ma10[-1]:
            self.position.close()


SCHEMA = {
    "d_rsi":           {"type": "int",   "default": 30,   "min": 5,    "max": 100, "label": "RSI Daily Period"},
    "w_rsi":           {"type": "int",   "default": 30,   "min": 5,    "max": 100, "label": "RSI Weekly Period"},
    "level":           {"type": "float", "default": 70.0, "min": 50.0, "max": 95.0, "label": "RSI Level"},
    "sl_pct":          {"type": "float", "default": 8.0,  "min": 1.0,  "max": 20.0, "label": "Stop Loss %"},
    "ma_exit_gap_pct": {"type": "float", "default": 2.0,  "min": 0.5,  "max": 10.0, "label": "MA10 Exit Gap %"},
}
