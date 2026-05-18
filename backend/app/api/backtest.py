"""POST /api/backtest/run — execute a backtest on supplied candles."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core.runner import run_backtest

router = APIRouter(prefix="/api", tags=["backtest"])


class Candle(BaseModel):
    time: int
    open: float
    high: float
    low: float
    close: float
    volume: float


class BacktestRequest(BaseModel):
    strategy: str
    params: dict[str, Any] = Field(default_factory=dict)
    candles: list[Candle]
    cash: float = 10_000.0
    commission: float = 0.0004
    margin: float | None = None  # auto-derived from params.leverage if omitted
    trade_on_close: bool = False
    hedging: bool = False
    exclusive_orders: bool = False


@router.post("/backtest/run")
def post_backtest_run(req: BacktestRequest):
    try:
        result = run_backtest(
            candles=[c.model_dump() for c in req.candles],
            strategy_name=req.strategy,
            params=req.params,
            cash=req.cash,
            commission=req.commission,
            margin=req.margin,
            trade_on_close=req.trade_on_close,
            hedging=req.hedging,
            exclusive_orders=req.exclusive_orders,
        )
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Backtest failed: {exc}")
    return result
