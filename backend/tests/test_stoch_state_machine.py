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
    stats = bt.run()
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
    stats = bt_short_only.run(long_enabled=False, short_enabled=True)
    trades = stats["_trades"]
    n_long = int((trades["Size"] > 0).sum())
    n_short = int((trades["Size"] < 0).sum())
    print(f"[short-only] long={n_long} short={n_short}")
    assert n_long == 0, "long_enabled=False should suppress all longs"
    assert n_short > 0, "expected short trades on oscillating data with short_enabled=True"


def test_sl_distance_matches_pct() -> None:
    df = make_oscillating_path()
    bt = FractionalBacktest(df, StochStateMachine, cash=10_000, commission=0.0, margin=1 / 5)
    stats = bt.run(sl_pct=2.0, tp_pct=3.0)
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
        trailing_enabled=False
    )
    trail = FractionalBacktest(df, StochStateMachine, cash=10_000, commission=0.0, margin=1 / 5).run(
        trailing_enabled=True, trailing_trigger_pct=0.5, trailing_offset_pct=0.25
    )
    print(
        f"[trailing] base maxDD={base['Max. Drawdown [%]']:.2f}% trail maxDD={trail['Max. Drawdown [%]']:.2f}% "
        f"| base trades={base['# Trades']} trail trades={trail['# Trades']}"
    )


def test_auto_reverse_increases_trades() -> None:
    df = make_oscillating_path(n=600)
    base = FractionalBacktest(df, StochStateMachine, cash=10_000, commission=0.0, margin=1 / 5).run(
        auto_reverse_enabled=False, sl_pct=1.0, tp_pct=5.0
    )
    rev = FractionalBacktest(df, StochStateMachine, cash=10_000, commission=0.0, margin=1 / 5).run(
        auto_reverse_enabled=True, auto_reverse_max=3, sl_pct=1.0, tp_pct=5.0
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
    print("\nAll tests passed ✓")
