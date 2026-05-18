"""FastAPI entry point for the local backtesting backend."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import backtest, data, strategies

app = FastAPI(
    title="Backtesting Trade",
    description="Local crypto backtesting (Binance Futures + backtesting.py)",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(data.router)
app.include_router(strategies.router)
app.include_router(backtest.router)


@app.get("/health")
def health():
    return {"status": "ok"}
