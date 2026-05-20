"""Unit tests for Adaptive Trend 24H.

Run from backend/ with: `uv run python -m tests.test_adaptive_trend_24h`
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from backtesting.lib import FractionalBacktest

from app.core.runner import run_backtest
from app.core.strategies.adaptive_trend_24h import (
    AdaptiveTrend24H,
    donchian_high_prior,
    donchian_low_prior,
)


def make_adaptive_trend_path(n: int = 3600) -> pd.DataFrame:
    """Synthetic 1h path with long bull, bear, then bull regimes."""
    third = n // 3
    close = np.r_[
        np.linspace(100.0, 260.0, third),
        np.linspace(260.0, 80.0, third),
        np.linspace(80.0, 190.0, n - 2 * third),
    ]
    wave = 3.0 * np.sin(np.arange(n) / 3.0)
    close = close + wave
    open_ = np.r_[close[0], close[:-1]]
    spread = 0.15 + np.abs(np.sin(np.arange(n) / 5.0)) * 0.05
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    volume = np.full(n, 1000.0)
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=idx,
    )


def make_low_vol_chop(n: int = 1600) -> pd.DataFrame:
    t = np.arange(n)
    close = 100.0 + 0.02 * np.sin(t / 4.0)
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + 0.02
    low = np.minimum(open_, close) - 0.02
    volume = np.full(n, 1000.0)
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=idx,
    )


def to_candles(df: pd.DataFrame) -> list[dict]:
    return [
        {
            "time": int(ts.timestamp()),
            "open": float(row["Open"]),
            "high": float(row["High"]),
            "low": float(row["Low"]),
            "close": float(row["Close"]),
            "volume": float(row["Volume"]),
        }
        for ts, row in df.iterrows()
    ]


def run_strategy(df: pd.DataFrame, **params):
    bt = FractionalBacktest(
        df,
        AdaptiveTrend24H,
        cash=10_000,
        commission=0.0,
        margin=1 / 3,
        trade_on_close=True,
    )
    return bt.run(**params)


def test_synthetic_trend_path_trades_long_and_short() -> None:
    df = make_adaptive_trend_path()
    stats = run_strategy(df)
    trades = stats["_trades"]
    n_long = int((trades["Size"] > 0).sum())
    n_short = int((trades["Size"] < 0).sum())
    print(f"[adaptive-trend] trades={len(trades)} long={n_long} short={n_short}")
    assert n_long > 0, "expected long trades in bullish HTF regime"
    assert n_short > 0, "expected short trades in bearish HTF regime"


def test_low_vol_chop_stays_flat() -> None:
    df = make_low_vol_chop()
    stats = run_strategy(df)
    n_trades = int(stats["# Trades"])
    print(f"[low-vol-chop] trades={n_trades}")
    assert n_trades == 0, "ATR% filter should block low-volatility chop"


def test_donchian_uses_prior_window_without_lookahead() -> None:
    high = np.array([10.0, 11.0, 12.0, 100.0])
    low = np.array([10.0, 9.0, 8.0, 1.0])

    high_prior = donchian_high_prior(high, 3)
    low_prior = donchian_low_prior(low, 3)

    assert np.isnan(high_prior[:3]).all()
    assert np.isnan(low_prior[:3]).all()
    assert high_prior[-1] == 12.0, "current high spike must be excluded"
    assert low_prior[-1] == 8.0, "current low spike must be excluded"


def test_runner_serializes_adaptive_trend_result() -> None:
    df = make_adaptive_trend_path()
    result = run_backtest(
        to_candles(df),
        "adaptive_trend_24h",
        params={},
        cash=10_000,
        commission=0.0,
    )

    assert result["strategy"] == "adaptive_trend_24h"
    assert result["stats"]
    assert result["trades"], "expected serialized trades"
    assert result["equity_curve"], "expected serialized equity curve"
    assert result["indicators"], "expected serialized indicators"
    assert any(
        item["name"].startswith("Donchian High prev")
        for item in result["indicators"]
    )


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    test_synthetic_trend_path_trades_long_and_short()
    test_low_vol_chop_stays_flat()
    test_donchian_uses_prior_window_without_lookahead()
    test_runner_serializes_adaptive_trend_result()
    print("\nAll Adaptive Trend 24H tests passed")
