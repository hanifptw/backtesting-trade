"""Strategy registry.

Tambah preset baru = bikin file di folder ini, import & daftar di REGISTRY.
Tidak perlu mengubah endpoint API atau frontend — schema otomatis terexpose.
"""

from __future__ import annotations

from typing import TypedDict

from backtesting import Strategy

from . import adaptive_trend_24h as _adaptive_trend_24h
from . import btc_trend_pullback as _btp
from . import freqtrade_futures_24h as _freqtrade_futures_24h
from . import hmm_regime as _hmm
from . import regime_adaptive as _regime_adaptive
from . import rsi as _rsi
from . import rsi_multitf as _rsi_multitf
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
    "rsi_multitf": {
        "cls": _rsi_multitf.RSIMultiTF,
        "schema": _rsi_multitf.SCHEMA,
        "label": "RSI Multi-Timeframe",
        "description": (
            "Long-only. Entry saat RSI daily & weekly keduanya > 70 "
            "dengan MA stack tersusun rapi (price > MA10 > MA20 > MA50 > MA100). "
            "Exit via SL 8% atau price jatuh 2% di bawah MA10."
        ),
    },
    "hmm_regime": {
        "cls": _hmm.HMMRegimeStrategy,
        "schema": _hmm.SCHEMA,
        "label": "HMM Regime Detection",
        "description": (
            "Klasifikasi kondisi pasar ke 3 regime menggunakan Hidden Markov Model: "
            "Bear / Sideways / Bull. "
            "HMM dilatih pada data yang dimasukkan (gunakan BTC/USDT 2019–2023). "
            "Fase ini hanya visualisasi regime di chart — sub-strategy per regime "
            "akan dikembangkan selanjutnya."
        ),
    },
    "regime_adaptive": {
        "cls": _regime_adaptive.RegimeAdaptiveStrategy,
        "schema": _regime_adaptive.SCHEMA,
        "label": "Regime-Adaptive (EMA/ADX)",
        "description": (
            "Multi-mode strategy. EMA50/EMA200 + ADX klasifikasi market ke "
            "Bull / Bear / Range / NoTrade. Tiap regime punya entry-trigger sendiri "
            "(Stoch oversold/overbought + retrace ke EMA-fast untuk trend, "
            "S/R level yang dikunci untuk range). Risk ATR-based (SL 1.5× ATR, "
            "trailing 2× ATR setelah profit 1× ATR). Leverage configurable per-regime "
            "lewat position sizing."
        ),
    },
    "adaptive_trend_24h": {
        "cls": _adaptive_trend_24h.AdaptiveTrend24H,
        "schema": _adaptive_trend_24h.SCHEMA,
        "label": "Adaptive Trend 24H",
        "description": (
            "24/7 crypto futures trend-following strategy. Closed HTF candles "
            "(default 4h) classify bull/bear regimes with EMA50/EMA200, ADX, "
            "and Supertrend. Entries use current-TF Donchian breakouts, ATR% "
            "volatility filter, ATR stop, ATR trailing, and conservative short sizing."
        ),
    },
    "freqtrade_futures_24h": {
        "cls": _freqtrade_futures_24h.FreqtradeFutures24H,
        "schema": _freqtrade_futures_24h.SCHEMA,
        "label": "Freqtrade Futures 24H",
        "description": (
            "Native composite inspired by freqtrade-strategies futures examples. "
            "Default ensemble mode combines ADX/SMA, EMA+HTF regime, triple "
            "Supertrend, Trend+OBV, volatility breakout, and OTT signals with "
            "ATR stop/trailing risk controls for 24/7 crypto futures backtests."
        ),
    },
    "btc_trend_pullback": {
        "cls": _btp.BTCTrendPullback,
        "schema": _btp.SCHEMA,
        "label": "BTC Trend-Pullback (Sharpe)",
        "description": (
            "Trend-following pullback untuk bot 24/7. Filter HTF EMA200 (default 4h) "
            "menentukan arah; entry hanya pada pullback ke EMA fast (default EMA20) "
            "saat RSI cross balik level 50. SL/TP/trailing semua ATR-sized, jadi "
            "satu set param robust di banyak regime. Tuned untuk BTC 1h–4h, target "
            "Sharpe tinggi (cek `scripts/optimize_btc_trend_pullback.py` untuk "
            "grid-search per market)."
        ),
    },
}


def get(name: str) -> StrategyEntry:
    if name not in REGISTRY:
        raise KeyError(f"Unknown strategy: {name}. Available: {list(REGISTRY)}")
    return REGISTRY[name]
