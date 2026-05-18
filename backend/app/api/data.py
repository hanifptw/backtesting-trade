"""GET /api/ohlcv — fetch candles from CCXT, return JSON ready for chart."""

from __future__ import annotations

from typing import Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from app.core.data_source import (
    DEFAULT_EXCHANGE,
    DEFAULT_TIMEFRAME,
    HARD_CAP_CANDLES,
    df_to_records,
    fetch_ohlcv,
)

router = APIRouter(prefix="/api", tags=["data"])


def _iso_to_ms(s: Optional[str]) -> Optional[int]:
    if s is None:
        return None
    try:
        ts = pd.Timestamp(s, tz="UTC") if pd.Timestamp(s).tz is None else pd.Timestamp(s)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid datetime '{s}': {exc}")
    return int(ts.timestamp() * 1000)


@router.get("/ohlcv")
def get_ohlcv(
    symbol: str = Query(..., description="e.g. BTC/USDT"),
    timeframe: str = Query(DEFAULT_TIMEFRAME, description="e.g. 1m, 15m, 1h, 1d"),
    since: Optional[str] = Query(None, description="ISO datetime (UTC) — start of range"),
    until: Optional[str] = Query(None, description="ISO datetime (UTC) — end of range"),
    exchange: str = Query(DEFAULT_EXCHANGE, description="CCXT exchange id, default binanceusdm"),
    limit: int = Query(HARD_CAP_CANDLES, ge=1, le=HARD_CAP_CANDLES),
):
    """Fetch OHLCV candles. Default exchange is Binance USDT-M Futures."""
    try:
        df = fetch_ohlcv(
            symbol=symbol,
            timeframe=timeframe,
            since_ms=_iso_to_ms(since),
            until_ms=_iso_to_ms(until),
            exchange_id=exchange,
            max_candles=limit,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Exchange error: {exc}")

    return {
        "exchange": exchange,
        "symbol": symbol,
        "timeframe": timeframe,
        "count": len(df),
        "candles": df_to_records(df),
    }
