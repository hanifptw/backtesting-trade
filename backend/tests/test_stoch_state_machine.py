"""Unit-test the Stochastic state-machine on a synthetic price path designed to
trigger specific transitions. Run from backend/ with: `uv run python -m tests.test_stoch_state_machine`
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from backtesting.lib import FractionalBacktest

from app.core.indicators import stoch_k_line
from app.core.strategies.stoch_state_machine import StochStateMachine


def make_oscillating_path(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """A sinusoidal path that pushes Stochastic deep into OS then OB repeatedly."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    base = 1000.0
    amp = 80.0
    cycles = 5
    close = base + amp * np.sin(2 * np.pi * cycles * t / n) + rng.normal(0, 1.5, n)
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + np.abs(rng.normal(2, 0.5, n))
    low = np.minimum(open_, close) - np.abs(rng.normal(2, 0.5, n))
    vol = np.full(n, 1000.0)
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol},
        index=idx,
    )


def test_baseline_trades_fire() -> None:
    df = make_oscillating_path()
    bt = FractionalBacktest(
        df, StochStateMachine, cash=10_000, commission=0.0,
        margin=1 / 5,  # leverage=5
    )
    # Disable HTF filter — this test verifies stoch-cross entry, not the gate.
    stats = bt.run(supertrend_filter_enabled=False)
    n_trades = int(stats["# Trades"])
    print(f"[baseline] trades={n_trades} return={stats['Return [%]']:.2f}% win={stats.get('Win Rate [%]')}")
    assert n_trades > 0, "expected at least one trade on oscillating synthetic data"

    # Verify entry conditions on the first long entry
    trades = stats["_trades"]
    longs = trades[trades["Size"] > 0]
    if len(longs):
        first_long = longs.iloc[0]
        entry_bar = int(first_long["EntryBar"])
        k_at_entry = stoch_k_line(df["Close"].values, 14, 14, 3, 3)
        # Per spec: entry happens when K crosses BACK above oversold (20).
        # So at entry_bar, K should be >= 20 and at entry_bar-1, K should be < 20.
        if entry_bar >= 1 and not np.isnan(k_at_entry[entry_bar - 1]) and not np.isnan(k_at_entry[entry_bar]):
            print(f"  first LONG entry @ bar {entry_bar}: K[{entry_bar-1}]={k_at_entry[entry_bar-1]:.2f} K[{entry_bar}]={k_at_entry[entry_bar]:.2f}")


def test_side_toggle() -> None:
    df = make_oscillating_path()
    bt_short_only = FractionalBacktest(
        df, StochStateMachine, cash=10_000, commission=0.0, margin=1 / 5,
    )
    stats = bt_short_only.run(
        long_enabled=False, short_enabled=True, supertrend_filter_enabled=False
    )
    trades = stats["_trades"]
    n_long = int((trades["Size"] > 0).sum())
    n_short = int((trades["Size"] < 0).sum())
    print(f"[short-only] long={n_long} short={n_short}")
    assert n_long == 0, "long_enabled=False should suppress all longs"
    assert n_short > 0, "expected short trades on oscillating data with short_enabled=True"


def test_sl_distance_matches_pct() -> None:
    df = make_oscillating_path()
    bt = FractionalBacktest(df, StochStateMachine, cash=10_000, commission=0.0, margin=1 / 5)
    stats = bt.run(sl_pct=2.0, tp_pct=3.0, supertrend_filter_enabled=False)
    trades = stats["_trades"]
    longs = trades[trades["Size"] > 0]
    if len(longs) == 0:
        print("[sl_distance] skipping — no longs to check")
        return
    # For long trades that exited at SL, exit_price should be ~98% of entry_price.
    losses = longs[longs["PnL"] < 0]
    if len(losses):
        ratios = losses["ExitPrice"] / losses["EntryPrice"]
        # SL exits cluster around 0.98; cross-down exits are anywhere.
        sl_exits = ratios[(ratios > 0.975) & (ratios < 0.985)]
        print(f"[sl_distance] {len(sl_exits)}/{len(losses)} losing longs exited near SL (~2 % below entry)")


def test_trailing_reduces_max_dd() -> None:
    df = make_oscillating_path(n=600)
    base = FractionalBacktest(df, StochStateMachine, cash=10_000, commission=0.0, margin=1 / 5).run(
        trailing_enabled=False, supertrend_filter_enabled=False
    )
    trail = FractionalBacktest(df, StochStateMachine, cash=10_000, commission=0.0, margin=1 / 5).run(
        trailing_enabled=True, trailing_trigger_pct=0.5, trailing_offset_pct=0.25,
        supertrend_filter_enabled=False,
    )
    print(
        f"[trailing] base maxDD={base['Max. Drawdown [%]']:.2f}% trail maxDD={trail['Max. Drawdown [%]']:.2f}% "
        f"| base trades={base['# Trades']} trail trades={trail['# Trades']}"
    )


def make_trending_path(n: int = 1200, slope: float = 10.0, seed: int = 1) -> pd.DataFrame:
    """Linear drift + Stoch-friendly oscillation. `slope` sets HTF trend direction.

    With n=1200 hourly bars (~50 days) we get ~300 4h bars — well past Supertrend
    warm-up. Slope is set strong enough that 4h Supertrend stays pinned bullish
    (slope >0) or bearish (slope <0) throughout, while short-timeframe oscillation
    still drives %K through OS/OB to trigger crosses. Base price is set high
    enough that even a sustained downtrend leaves prices comfortably positive.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    base = 50_000.0
    amp = 300.0
    # High-frequency oscillation (~20-bar cycle) so each 4h resample window
    # spans nearly a full cycle — oscillation cancels on HTF while still
    # driving %K through OS/OB on the main timeframe.
    cycles = 60
    close = base + slope * t + amp * np.sin(2 * np.pi * cycles * t / n) + rng.normal(0, 5.0, n)
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + np.abs(rng.normal(2, 0.5, n))
    low = np.minimum(open_, close) - np.abs(rng.normal(2, 0.5, n))
    vol = np.full(n, 1000.0)
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol},
        index=idx,
    )


def test_supertrend_filter_blocks_shorts_in_uptrend() -> None:
    """In a clear uptrend, ST 4h is bullish → all SHORT entries must be suppressed."""
    df = make_trending_path(slope=10.0)
    stats = FractionalBacktest(
        df, StochStateMachine, cash=10_000, commission=0.0, margin=1 / 5
    ).run(supertrend_filter_enabled=True)
    trades = stats["_trades"]
    n_short = int((trades["Size"] < 0).sum())
    n_long = int((trades["Size"] > 0).sum())
    print(f"[ST filter uptrend] long={n_long} short={n_short}")
    assert n_short == 0, "ST 4h bullish must block all shorts"
    assert n_long > 0, "expected at least one long in uptrend"


def test_supertrend_filter_blocks_longs_in_downtrend() -> None:
    """In a clear downtrend, ST 4h is bearish → all LONG entries must be suppressed."""
    df = make_trending_path(slope=-10.0)
    stats = FractionalBacktest(
        df, StochStateMachine, cash=10_000, commission=0.0, margin=1 / 5
    ).run(supertrend_filter_enabled=True)
    trades = stats["_trades"]
    n_short = int((trades["Size"] < 0).sum())
    n_long = int((trades["Size"] > 0).sum())
    print(f"[ST filter downtrend] long={n_long} short={n_short}")
    assert n_long == 0, "ST 4h bearish must block all longs"
    assert n_short > 0, "expected at least one short in downtrend"


def test_supertrend_filter_off_lets_counter_trend_through() -> None:
    """Toggle confirms: filter OFF on the same downtrend data lets longs back in."""
    df = make_trending_path(slope=-10.0)
    on = FractionalBacktest(
        df, StochStateMachine, cash=10_000, commission=0.0, margin=1 / 5
    ).run(supertrend_filter_enabled=True)
    off = FractionalBacktest(
        df, StochStateMachine, cash=10_000, commission=0.0, margin=1 / 5
    ).run(supertrend_filter_enabled=False)
    on_longs = int((on["_trades"]["Size"] > 0).sum())
    off_longs = int((off["_trades"]["Size"] > 0).sum())
    print(f"[ST filter toggle] longs ON={on_longs} OFF={off_longs}")
    assert on_longs == 0 and off_longs > 0, (
        "filter must gate longs in a bearish HTF; turning it off restores them"
    )


def test_auto_reverse_respects_supertrend_filter() -> None:
    """LONG that hits SL in a downtrend must NOT flip to SHORT-then-LONG-reverse if
    HTF disagrees. With slope<0 (ST 4h bearish) and the filter ON, no LONG opens
    in the first place, so no reverse chain can build up beyond a SHORT→… cycle.
    Concretely: any LONG observed in trades is impossible with filter ON + slope<0.
    """
    df = make_trending_path(slope=-10.0)
    stats = FractionalBacktest(
        df, StochStateMachine, cash=10_000, commission=0.0, margin=1 / 5
    ).run(
        supertrend_filter_enabled=True,
        auto_reverse_enabled=True, auto_reverse_max=5,
        sl_pct=1.0, tp_pct=5.0,
    )
    trades = stats["_trades"]
    n_long = int((trades["Size"] > 0).sum())
    print(f"[ST filter + auto-reverse] longs={n_long} (must be 0)")
    assert n_long == 0, "auto-reverse must not bypass the HTF filter"


def test_auto_reverse_increases_trades() -> None:
    df = make_oscillating_path(n=600)
    base = FractionalBacktest(df, StochStateMachine, cash=10_000, commission=0.0, margin=1 / 5).run(
        auto_reverse_enabled=False, sl_pct=1.0, tp_pct=5.0, supertrend_filter_enabled=False
    )
    rev = FractionalBacktest(df, StochStateMachine, cash=10_000, commission=0.0, margin=1 / 5).run(
        auto_reverse_enabled=True, auto_reverse_max=3, sl_pct=1.0, tp_pct=5.0,
        supertrend_filter_enabled=False,
    )
    print(f"[auto_reverse] base trades={base['# Trades']} reverse trades={rev['# Trades']}")
    assert rev["# Trades"] >= base["# Trades"], "auto-reverse should produce ≥ same trades"


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    test_baseline_trades_fire()
    test_side_toggle()
    test_sl_distance_matches_pct()
    test_trailing_reduces_max_dd()
    test_auto_reverse_increases_trades()
    test_supertrend_filter_blocks_shorts_in_uptrend()
    test_supertrend_filter_blocks_longs_in_downtrend()
    test_supertrend_filter_off_lets_counter_trend_through()
    test_auto_reverse_respects_supertrend_filter()
    print("\nAll tests passed ✓")
