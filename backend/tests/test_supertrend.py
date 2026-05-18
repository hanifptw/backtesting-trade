"""Unit-test Supertrend indicator and strategy.

Run from backend/ with: `uv run python -m tests.test_supertrend`
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from backtesting.lib import FractionalBacktest

from app.core.indicators import (
    supertrend,
    supertrend_bearish_line,
    supertrend_bullish_line,
    supertrend_direction,
    supertrend_line,
)
from app.core.runner import run_backtest
from app.core.strategies.supertrend import SupertrendStrategy


def make_trend_flip_path(n: int = 480) -> pd.DataFrame:
    """Synthetic path with clear up/down trends to force Supertrend flips."""
    quarter = n // 4
    close = np.r_[
        np.linspace(160.0, 100.0, quarter),
        np.linspace(100.0, 180.0, quarter),
        np.linspace(180.0, 90.0, quarter),
        np.linspace(90.0, 155.0, n - 3 * quarter),
    ]
    wave = np.sin(np.arange(n) / 4.0) * 1.2
    close = close + wave
    open_ = np.r_[close[0], close[:-1]]
    spread = 2.0 + np.abs(np.sin(np.arange(n) / 7.0))
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
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


def test_indicator_outputs() -> None:
    df = make_trend_flip_path()
    close = df["Close"].values

    for method in ("rma", "sma", "ema"):
        values = supertrend(
            df["High"].values,
            df["Low"].values,
            close,
            atr_period=10,
            multiplier=3.0,
            atr_method=method,
        )
        line = values[0]
        direction = values[1]
        assert values.shape == (2, len(df))
        assert np.isnan(line[:9]).any(), "expected warmup NaNs"
        assert np.isfinite(line[20:]).any(), f"expected valid line for {method}"
        assert set(np.unique(direction[np.isfinite(direction)])) <= {-1.0, 1.0}

    line = supertrend_line(df["High"].values, df["Low"].values, close, 10, 3.0, "rma")
    direction = supertrend_direction(df["High"].values, df["Low"].values, close, 10, 3.0, "rma")

    bullish = np.where(direction > 0)[0]
    bearish = np.where(direction < 0)[0]
    assert len(bullish) > 0, "expected bullish Supertrend segment"
    assert len(bearish) > 0, "expected bearish Supertrend segment"
    assert line[bullish[-1]] < close[bullish[-1]], "bullish line should sit below price"
    assert line[bearish[-1]] > close[bearish[-1]], "bearish line should sit above price"

    bullish_line = supertrend_bullish_line(
        df["High"].values, df["Low"].values, close, 10, 3.0, "rma"
    )
    bearish_line = supertrend_bearish_line(
        df["High"].values, df["Low"].values, close, 10, 3.0, "rma"
    )
    assert np.isfinite(bullish_line[bullish]).any()
    assert np.isnan(bullish_line[bearish]).all()
    assert np.isfinite(bearish_line[bearish]).any()
    assert np.isnan(bearish_line[bullish]).all()


def test_strategy_trades_fire() -> None:
    df = make_trend_flip_path()
    bt = FractionalBacktest(df, SupertrendStrategy, cash=10_000, commission=0.0, margin=1 / 5)
    stats = bt.run()
    n_trades = int(stats["# Trades"])
    print(f"[supertrend] trades={n_trades} return={stats['Return [%]']:.2f}%")
    assert n_trades > 0, "expected trades on trend-flip synthetic data"


def test_side_toggles() -> None:
    df = make_trend_flip_path()

    long_disabled = FractionalBacktest(
        df, SupertrendStrategy, cash=10_000, commission=0.0, margin=1 / 5
    ).run(long_enabled=False, short_enabled=True)
    trades = long_disabled["_trades"]
    assert int((trades["Size"] > 0).sum()) == 0
    assert int((trades["Size"] < 0).sum()) > 0

    short_disabled = FractionalBacktest(
        df, SupertrendStrategy, cash=10_000, commission=0.0, margin=1 / 5
    ).run(long_enabled=True, short_enabled=False)
    trades = short_disabled["_trades"]
    assert int((trades["Size"] < 0).sum()) == 0
    assert int((trades["Size"] > 0).sum()) > 0


def test_runner_leverage_toggle() -> None:
    df = make_trend_flip_path()
    candles = to_candles(df)
    base_params = {
        "atr_period": 10,
        "multiplier": 3.0,
        "atr_method": "rma",
        "trade_amount": 100.0,
        "leverage": 5,
    }

    leveraged = run_backtest(
        candles,
        "supertrend",
        params={**base_params, "leverage_enabled": True},
        cash=10_000,
        commission=0.0,
    )
    unleveraged = run_backtest(
        candles,
        "supertrend",
        params={**base_params, "leverage_enabled": False},
        cash=10_000,
        commission=0.0,
    )

    assert leveraged["trades"], "expected leveraged run trades"
    assert unleveraged["trades"], "expected unleveraged run trades"
    assert abs(leveraged["trades"][0]["size"]) > abs(unleveraged["trades"][0]["size"])


def test_fixed_sl_tp_enabled() -> None:
    df = make_trend_flip_path()
    stats = FractionalBacktest(
        df, SupertrendStrategy, cash=10_000, commission=0.0, margin=1 / 5
    ).run(fixed_sl_tp_enabled=True, sl_pct=2.0, tp_pct=4.0)
    print(f"[fixed_sl_tp] trades={stats['# Trades']}")
    assert int(stats["# Trades"]) > 0


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    test_indicator_outputs()
    test_strategy_trades_fire()
    test_side_toggles()
    test_runner_leverage_toggle()
    test_fixed_sl_tp_enabled()
    print("\nAll Supertrend tests passed")
