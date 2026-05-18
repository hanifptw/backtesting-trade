"""Strategy registry.

Tambah preset baru = bikin file di folder ini, import & daftar di REGISTRY.
Tidak perlu mengubah endpoint API atau frontend — schema otomatis terexpose.
"""

from __future__ import annotations

from typing import TypedDict

from backtesting import Strategy

from . import rsi as _rsi
from . import sma_cross as _sma
from . import stoch_state_machine as _stoch
from . import supertrend as _supertrend


class StrategyEntry(TypedDict):
    cls: type[Strategy]
    schema: dict
    label: str
    description: str


REGISTRY: dict[str, StrategyEntry] = {
    "stoch_state_machine": {
        "cls": _stoch.StochStateMachine,
        "schema": _stoch.SCHEMA,
        "label": "Stochastic State-Machine",
        "description": (
            "Stoch RSI threshold strategy. LONG when %K crosses UP through the oversold "
            "level; SHORT when %K crosses DOWN through the overbought level. Exits only "
            "via SL/TP (or trailing / auto-reverse) — no oscillator-based close. "
            "Port of the user's live Binance Futures bot (see STRATEGY.md)."
        ),
    },
    "sma_cross": {
        "cls": _sma.SMACrossStrategy,
        "schema": _sma.SCHEMA,
        "label": "SMA Crossover",
        "description": "Long when fast SMA crosses above slow SMA; close on opposite cross.",
    },
    "rsi": {
        "cls": _rsi.RSIStrategy,
        "schema": _rsi.SCHEMA,
        "label": "RSI Mean Reversion",
        "description": "Long when RSI enters oversold zone; close when RSI enters overbought zone.",
    },
    "supertrend": {
        "cls": _supertrend.SupertrendStrategy,
        "schema": _supertrend.SCHEMA,
        "label": "Supertrend",
        "description": (
            "Trend-following Supertrend strategy. Enters on bullish/bearish "
            "direction flips, uses the Supertrend line as the primary stop, "
            "with optional fixed SL/TP and leverage toggles."
        ),
    },
}


def get(name: str) -> StrategyEntry:
    if name not in REGISTRY:
        raise KeyError(f"Unknown strategy: {name}. Available: {list(REGISTRY)}")
    return REGISTRY[name]
