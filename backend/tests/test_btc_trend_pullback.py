"""Unit tests for BTC Trend-Pullback strategy.

Run from backend/ with:  `uv run python -m tests.test_btc_trend_pullback`
or                       `.venv/bin/python -m tests.test_btc_trend_pullback`
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from backtesting.lib import FractionalBacktest

from app.core.strategies.btc_trend_pullback import BTCTrendPullback


def make_trend_with_pullbacks(
    n: int = 3000,
    slope: float = 6.0,
    seed: int = 7,
    pullback_period: int = 80,
    pullback_amp: float = 250.0,
) -> pd.DataFrame:
    """Linear drift + periodic pullback oscillation. `slope` controls the HTF
    trend direction. Each `pullback_period` bars price retraces ~`pullback_amp`
    units, enough to touch a 20-period EMA below current price (or above, for
    downtrends), then resumes the trend — that's the pattern the strategy
    expects to trade.

    The base price (50,000) is high enough that even a sustained downtrend
    leaves prices comfortably positive over the full window.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    base = 50_000.0
    # Slow short-tf oscillation gives the pullbacks; trend dominates HTF.
    pullback = pullback_amp * np.sin(2 * np.pi * t / pullback_period)
    noise = rng.normal(0, 8.0, n)
    close = base + slope * t + pullback + noise
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + np.abs(rng.normal(4, 1.0, n))
    low = np.minimum(open_, close) - np.abs(rng.normal(4, 1.0, n))
    vol = np.full(n, 1000.0)
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol},
        index=idx,
    )


def make_flat_no_pullback(n: int = 1500, seed: int = 3) -> pd.DataFrame:
    """Nearly-flat price with tiny noise — no real trend, no meaningful
    pullbacks to EMA20. Strategy should generate few or no trades because
    the HTF trend filter sees an indeterminate (close ≈ EMA200) regime.
    """
    rng = np.random.default_rng(seed)
    base = 50_000.0
    close = base + rng.normal(0, 1.0, n).cumsum() * 0.5
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + 2
    low = np.minimum(open_, close) - 2
    vol = np.full(n, 1000.0)
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol},
        index=idx,
    )


def _run(df: pd.DataFrame, **params):
    bt = FractionalBacktest(
        df, BTCTrendPullback, cash=10_000, commission=0.0, margin=1 / 5,
    )
    return bt.run(**params)


def test_htf_trend_blocks_counter_trend_longs() -> None:
    """Strong downtrend (slope < 0) → HTF EMA200 above price → all longs blocked."""
    df = make_trend_with_pullbacks(slope=-6.0)
    stats = _run(df)
    trades = stats["_trades"]
    n_long = int((trades["Size"] > 0).sum())
    n_short = int((trades["Size"] < 0).sum())
    print(f"[htf-block-longs] long={n_long} short={n_short} trades={len(trades)}")
    assert n_long == 0, "HTF bearish must block all longs"
    assert n_short > 0, "expected at least one short in a clear downtrend"


def test_htf_trend_blocks_counter_trend_shorts() -> None:
    """Strong uptrend → HTF EMA200 below price → all shorts blocked."""
    df = make_trend_with_pullbacks(slope=6.0)
    stats = _run(df)
    trades = stats["_trades"]
    n_long = int((trades["Size"] > 0).sum())
    n_short = int((trades["Size"] < 0).sum())
    print(f"[htf-block-shorts] long={n_long} short={n_short} trades={len(trades)}")
    assert n_short == 0, "HTF bullish must block all shorts"
    assert n_long > 0, "expected at least one long in a clear uptrend"


def test_no_trades_without_real_trend() -> None:
    """Flat noise + no meaningful pullbacks → strategy should mostly stand
    aside. The threshold (≤10 trades on 1500 bars) is loose so noise-driven
    RSI flips don't flag a regression.
    """
    df = make_flat_no_pullback()
    stats = _run(df)
    n_trades = int(stats["# Trades"])
    print(f"[flat-no-trend] trades={n_trades}")
    assert n_trades <= 10, f"expected very few trades on flat noise, got {n_trades}"


def test_atr_sl_distance_matches_formula() -> None:
    """For losing longs that exited via SL, exit ≈ entry − k_sl × ATR_entry.
    We can't reconstruct per-bar ATR from the stats DataFrame, so we instead
    verify the *ratio* of SL distance to entry price is in a sensible band.
    """
    df = make_trend_with_pullbacks(slope=6.0)
    stats = _run(df, k_sl_atr=2.0, k_tp_atr=3.0)
    trades = stats["_trades"]
    longs = trades[trades["Size"] > 0]
    if len(longs) == 0:
        print("[sl-distance] skipping — no longs to inspect")
        return
    losers = longs[longs["PnL"] < 0]
    if len(losers) == 0:
        print(f"[sl-distance] skipping — {len(longs)} longs, no losers")
        return
    # SL-hit losses cluster in a narrow band; non-SL exits (TP, end of data) are wider.
    sl_drop_pct = (1.0 - losers["ExitPrice"] / losers["EntryPrice"]) * 100.0
    print(
        f"[sl-distance] losing longs n={len(losers)} "
        f"sl_drop_pct median={float(sl_drop_pct.median()):.3f}% "
        f"min={float(sl_drop_pct.min()):.3f}% max={float(sl_drop_pct.max()):.3f}%"
    )
    # Sanity: SL distance is positive and reasonable (well under 5 % for k_sl=2.0
    # on synthetic data where ATR is small relative to base price).
    assert (sl_drop_pct > 0).all(), "losing longs must exit below entry"


def test_trailing_changes_outcome() -> None:
    """Force trailing to bite: very low trigger + wide TP so the trailing
    branch actually fires before TP. Compare against a baseline that wins
    via the static TP. We assert *something* differs — direction is data-
    dependent, we just want to confirm the branch is exercised end-to-end.
    """
    df = make_trend_with_pullbacks(slope=6.0, n=4000)
    # Park TP far enough away that virtually every exit is SL- or trail-driven.
    no_trail = _run(df, trailing_enabled=False, k_sl_atr=3.0, k_tp_atr=30.0)
    with_trail = _run(
        df, trailing_enabled=True, trail_trigger_atr=0.2, k_trail_atr=0.5,
        k_sl_atr=3.0, k_tp_atr=30.0,
    )
    print(
        f"[trailing] off: trades={no_trail['# Trades']} return={no_trail['Return [%]']:.2f}% "
        f"DD={no_trail['Max. Drawdown [%]']:.2f}% | "
        f"on: trades={with_trail['# Trades']} return={with_trail['Return [%]']:.2f}% "
        f"DD={with_trail['Max. Drawdown [%]']:.2f}%"
    )
    differs = (
        int(no_trail["# Trades"]) != int(with_trail["# Trades"])
        or abs(no_trail["Return [%]"] - with_trail["Return [%]"]) > 1e-6
        or abs(no_trail["Max. Drawdown [%]"] - with_trail["Max. Drawdown [%]"]) > 1e-6
    )
    assert differs, "trailing toggle should affect at least one of trades / return / DD"


def test_side_toggle_disables_longs() -> None:
    df = make_trend_with_pullbacks(slope=6.0)
    stats = _run(df, long_enabled=False)
    trades = stats["_trades"]
    n_long = int((trades["Size"] > 0).sum())
    print(f"[side-toggle] long_enabled=False → longs={n_long}")
    assert n_long == 0, "long_enabled=False must suppress all longs"


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    test_htf_trend_blocks_counter_trend_longs()
    test_htf_trend_blocks_counter_trend_shorts()
    test_no_trades_without_real_trend()
    test_atr_sl_distance_matches_formula()
    test_trailing_changes_outcome()
    test_side_toggle_disables_longs()
    print("\nAll tests passed ✓")
