"""Unit tests for Freqtrade Futures 24H.

Run from backend/ with: `uv run python -m tests.test_freqtrade_futures_24h`
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from backtesting.lib import FractionalBacktest

from app.core.runner import run_backtest
from app.core.strategies.freqtrade_futures_24h import (
    FreqtradeFutures24H,
    closed_htf_ohlcv,
    crossed_above_prior,
    crossed_below_prior,
)


FAST_PARAMS = {
    "ema_fast": 8,
    "ema_slow": 20,
    "sma_fast": 6,
    "sma_slow": 18,
    "adx_period": 8,
    "adx_threshold": 8.0,
    "atr_period": 8,
    "min_atr_pct": 0.0,
    "max_atr_pct": 20.0,
    "supertrend_period_1": 5,
    "supertrend_mult_1": 1.3,
    "supertrend_period_2": 8,
    "supertrend_mult_2": 1.8,
    "supertrend_period_3": 13,
    "supertrend_mult_3": 2.3,
    "obv_ema_period": 8,
    "volatility_atr_mult": 0.45,
    "initial_sl_atr_mult": 2.0,
    "trail_activation_atr_mult": 1.0,
    "trail_distance_atr_mult": 2.5,
    "cooldown_bars": 1,
}


def make_futures_trend_path(n: int = 1800) -> pd.DataFrame:
    """Synthetic 1h path with bullish, bearish, and bullish regimes."""
    third = n // 3
    close = np.r_[
        np.linspace(100.0, 230.0, third),
        np.linspace(230.0, 70.0, third),
        np.linspace(70.0, 185.0, n - 2 * third),
    ]
    wave = 2.0 * np.sin(np.arange(n) / 5.0) + 0.7 * np.sin(np.arange(n) / 17.0)
    close = close + wave
    open_ = np.r_[close[0], close[:-1]]
    spread = 0.35 + np.abs(np.sin(np.arange(n) / 9.0)) * 0.20
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    volume = 1000.0 + 150.0 * np.sin(np.arange(n) / 11.0)
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=idx,
    )


def make_low_vol_chop(n: int = 800) -> pd.DataFrame:
    t = np.arange(n)
    close = 100.0 + 0.015 * np.sin(t / 3.0)
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + 0.015
    low = np.minimum(open_, close) - 0.015
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
        FreqtradeFutures24H,
        cash=10_000,
        commission=0.0,
        margin=1 / 3,
        trade_on_close=True,
    )
    return bt.run(**params)


def test_synthetic_trend_path_trades_long_and_short() -> None:
    df = make_futures_trend_path()
    stats = run_strategy(df, **FAST_PARAMS, mode="ensemble", min_confirmations=2)
    trades = stats["_trades"]
    n_long = int((trades["Size"] > 0).sum())
    n_short = int((trades["Size"] < 0).sum())
    print(f"[freqtrade-futures] trades={len(trades)} long={n_long} short={n_short}")
    assert n_long > 0, "expected long trades in bullish regimes"
    assert n_short > 0, "expected short trades in bearish regime"


def test_each_mode_smoke_runs_and_serializes() -> None:
    df = make_futures_trend_path(1200)
    candles = to_candles(df)
    for mode in (
        "ensemble",
        "adx_sma",
        "ema_reinforced",
        "triple_supertrend",
        "trend_obv",
        "volatility_breakout",
        "ott",
    ):
        result = run_backtest(
            candles,
            "freqtrade_futures_24h",
            params={
                **FAST_PARAMS,
                "mode": mode,
                "regime_filter_enabled": False,
                "min_confirmations": 1,
            },
            cash=10_000,
            commission=0.0,
        )
        assert result["strategy"] == "freqtrade_futures_24h"
        assert result["stats"], f"expected stats for {mode}"
        assert result["equity_curve"], f"expected equity curve for {mode}"
        assert result["indicators"], f"expected indicators for {mode}"


def test_low_vol_chop_stays_flat() -> None:
    df = make_low_vol_chop()
    params = {**FAST_PARAMS, "min_atr_pct": 0.2}
    stats = run_strategy(df, **params, mode="ensemble")
    n_trades = int(stats["# Trades"])
    print(f"[freqtrade-low-vol-chop] trades={n_trades}")
    assert n_trades == 0, "ATR% filter should block low-volatility chop"


def test_cross_helpers_use_prior_bar_without_lookahead() -> None:
    left = np.array([1.0, 1.5, 3.0, 1.0])
    right = np.array([2.0, 2.0, 2.0, 2.0])

    crossed_up = crossed_above_prior(left, right)
    crossed_down = crossed_below_prior(left, right)

    assert crossed_up.tolist() == [0.0, 0.0, 1.0, 0.0]
    assert crossed_down.tolist() == [0.0, 0.0, 0.0, 1.0]


def test_htf_resample_excludes_current_open_candle() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="1h", tz="UTC")
    df = pd.DataFrame(
        {
            "Open": [10.0, 11.0, 12.0, 13.0, 999.0],
            "High": [10.0, 11.0, 12.0, 13.0, 999.0],
            "Low": [9.0, 10.0, 11.0, 12.0, 998.0],
            "Close": [10.0, 11.0, 12.0, 13.0, 999.0],
            "Volume": [1.0, 1.0, 1.0, 1.0, 1.0],
        },
        index=idx,
    )

    htf = closed_htf_ohlcv(df, "4h")
    first_closed = htf.iloc[0]

    assert htf.index[0] == pd.Timestamp("2024-01-01 04:00:00", tz="UTC")
    assert first_closed["High"] == 13.0, "04:00 open candle must be excluded"
    assert first_closed["Close"] == 13.0


def test_runner_serializes_freqtrade_futures_result() -> None:
    df = make_futures_trend_path()
    result = run_backtest(
        to_candles(df),
        "freqtrade_futures_24h",
        params={**FAST_PARAMS, "mode": "ensemble", "min_confirmations": 2},
        cash=10_000,
        commission=0.0,
    )

    assert result["strategy"] == "freqtrade_futures_24h"
    assert result["stats"]
    assert result["trades"], "expected serialized trades"
    assert result["equity_curve"], "expected serialized equity curve"
    assert result["indicators"], "expected serialized indicators"
    assert any(item["name"].startswith("HTF EMA") for item in result["indicators"])


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    test_synthetic_trend_path_trades_long_and_short()
    test_each_mode_smoke_runs_and_serializes()
    test_low_vol_chop_stays_flat()
    test_cross_helpers_use_prior_bar_without_lookahead()
    test_htf_resample_excludes_current_open_candle()
    test_runner_serializes_freqtrade_futures_result()
    print("\nAll Freqtrade Futures 24H tests passed")
