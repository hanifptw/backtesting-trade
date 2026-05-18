"""
CCXT wrapper for fetching OHLCV from crypto exchanges.

Returns a pandas DataFrame with Title-case columns (Open/High/Low/Close/Volume)
and UTC datetime index — ready for backtesting.py.

Default exchange is `binanceusdm` (USDT-M Futures), matching the live bot
described in STRATEGY.md. Override via the `exchange` parameter.
"""

from __future__ import annotations

import os
from pathlib import Path

import ccxt
import pandas as pd
from dotenv import load_dotenv

# Load backend/.env if present. Safe to call multiple times.
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

DEFAULT_EXCHANGE = "binanceusdm"
DEFAULT_TIMEFRAME = "15m"
MAX_CANDLES_PER_REQUEST = 1000  # Binance futures caps at 1000 per request
HARD_CAP_CANDLES = 5000


def _testnet_enabled() -> bool:
    return os.environ.get("EXCHANGE_TESTNET", "true").strip().lower() in {"1", "true", "yes"}

VALID_TIMEFRAMES = {
    "1m", "3m", "5m", "15m", "30m",
    "1h", "2h", "4h", "6h", "8h", "12h",
    "1d", "3d", "1w", "1M",
}


def _build_exchange(exchange_id: str) -> ccxt.Exchange:
    if not hasattr(ccxt, exchange_id):
        raise ValueError(f"Unknown exchange: {exchange_id}")
    cls = getattr(ccxt, exchange_id)
    config = {"enableRateLimit": True}
    api_key = os.environ.get("BINANCE_API_KEY")
    api_secret = os.environ.get("BINANCE_API_SECRET")
    if api_key and api_secret and exchange_id.startswith("binance"):
        config["apiKey"] = api_key
        config["secret"] = api_secret
    exchange = cls(config)
    if _testnet_enabled() and exchange_id.startswith("binance"):
        exchange.set_sandbox_mode(True)
    return exchange


def fetch_ohlcv(
    symbol: str,
    timeframe: str = DEFAULT_TIMEFRAME,
    since_ms: int | None = None,
    until_ms: int | None = None,
    exchange_id: str = DEFAULT_EXCHANGE,
    max_candles: int = HARD_CAP_CANDLES,
) -> pd.DataFrame:
    """Fetch OHLCV, paginating internally until until_ms (or max_candles).

    Returned DataFrame: index=DatetimeIndex(UTC), cols=Open,High,Low,Close,Volume.
    """
    if timeframe not in VALID_TIMEFRAMES:
        raise ValueError(f"Invalid timeframe '{timeframe}'. Allowed: {sorted(VALID_TIMEFRAMES)}")
    if max_candles > HARD_CAP_CANDLES:
        max_candles = HARD_CAP_CANDLES

    exchange = _build_exchange(exchange_id)

    all_candles: list[list[float]] = []
    cursor = since_ms

    while len(all_candles) < max_candles:
        remaining = max_candles - len(all_candles)
        limit = min(MAX_CANDLES_PER_REQUEST, remaining)
        chunk = exchange.fetch_ohlcv(symbol, timeframe, since=cursor, limit=limit)
        if not chunk:
            break
        all_candles.extend(chunk)
        last_ts = chunk[-1][0]
        if until_ms is not None and last_ts >= until_ms:
            break
        if len(chunk) < limit:
            break  # exchange exhausted
        cursor = last_ts + 1

    if until_ms is not None:
        all_candles = [c for c in all_candles if c[0] <= until_ms]

    if not all_candles:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

    df = pd.DataFrame(all_candles, columns=["timestamp", "Open", "High", "Low", "Close", "Volume"])
    df = df.drop_duplicates(subset="timestamp")
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp").sort_index()
    return df


def df_to_records(df: pd.DataFrame) -> list[dict]:
    """Serialize DataFrame to a JSON-friendly list of records."""
    out = []
    for ts, row in df.iterrows():
        out.append({
            "time": int(ts.timestamp()),  # unix seconds — what lightweight-charts expects
            "open": float(row["Open"]),
            "high": float(row["High"]),
            "low": float(row["Low"]),
            "close": float(row["Close"]),
            "volume": float(row["Volume"]),
        })
    return out
