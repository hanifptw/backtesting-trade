"""Grid-search BTCTrendPullback params, rank by Sharpe (Max-DD constrained).

Run from `backend/`:

    .venv/bin/python -m scripts.optimize_btc_trend_pullback \\
        --symbol BTC/USDT --timeframe 1h --bars 5000 --max-dd 30 --top 15

Notes
-----
- We fetch live OHLCV through the existing `data_source.fetch_ohlcv`, which
  is capped at 5000 candles per call. That's ~7 months of 1h or ~28 months
  of 4h — plenty for an out-of-sample-friendly sweep.
- We call `runner.run_backtest` so any cleanup or stats serialization that
  the API uses also applies here (single source of truth for "what the
  number means").
- The grid below is deliberately small (~1.5k combinations). Edit `PARAM_GRID`
  to widen it; multiprocessing not used to keep the script easy to read.
- Output: top-N to stdout + full CSV next to this script.
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

import pandas as pd

from app.core.data_source import df_to_records, fetch_ohlcv
from app.core.runner import run_backtest


# Edit this to widen / narrow the sweep. Total combos = product of list lengths.
PARAM_GRID: dict[str, list] = {
    "ema_fast":          [10, 20, 30],
    "ema_slow":          [50, 100],
    "rsi_threshold":     [45.0, 50.0, 55.0],
    "k_sl_atr":          [1.5, 2.0, 2.5, 3.0],
    "k_tp_atr":          [2.0, 3.0, 4.0],
    "pullback_lookback": [3, 5, 8],
    "trailing_enabled":  [True, False],
}

REPORT_STATS = (
    "Sharpe Ratio",
    "Return [%]",
    "Max. Drawdown [%]",
    "# Trades",
    "Win Rate [%]",
    "Profit Factor",
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--symbol", default="BTC/USDT", help="Trading pair (default: BTC/USDT)")
    p.add_argument("--timeframe", default="1h", choices=["1h", "2h", "4h"], help="Main TF (default: 1h)")
    p.add_argument("--bars", type=int, default=5000, help="Max candles to fetch (default 5000 = source cap)")
    p.add_argument("--exchange", default="binanceusdm")
    p.add_argument("--max-dd", type=float, default=30.0, help="Discard configs with Max DD worse than this (%%, positive number)")
    p.add_argument("--min-trades", type=int, default=20, help="Discard configs with fewer than this many trades")
    p.add_argument("--top", type=int, default=15, help="How many rows to print")
    p.add_argument("--csv", type=str, default=None, help="Override output CSV path")
    p.add_argument("--leverage", type=int, default=5)
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    print(f"Fetching {args.symbol} {args.timeframe} (up to {args.bars} bars)…", flush=True)
    df = fetch_ohlcv(
        args.symbol,
        timeframe=args.timeframe,
        max_candles=args.bars,
        exchange_id=args.exchange,
    )
    if df.empty:
        print(f"ERROR: no candles returned for {args.symbol} {args.timeframe}", file=sys.stderr)
        return 1
    candles = df_to_records(df)
    print(
        f"  → got {len(candles)} candles, "
        f"{df.index[0].date()} → {df.index[-1].date()}", flush=True,
    )

    keys = list(PARAM_GRID.keys())
    combos = list(itertools.product(*PARAM_GRID.values()))
    print(f"Sweeping {len(combos)} combinations…", flush=True)

    rows: list[dict] = []
    errors = 0
    skipped_dd = 0
    skipped_trades = 0
    t_start = time.time()

    for n, combo in enumerate(combos, 1):
        params = dict(zip(keys, combo))
        params["leverage"] = args.leverage
        try:
            result = run_backtest(
                candles,
                "btc_trend_pullback",
                params=params,
                cash=10_000.0,
                commission=0.0004,
            )
        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"  [{n}/{len(combos)}] error: {e}", file=sys.stderr, flush=True)
            continue

        stats = result["stats"]
        n_trades = int(stats.get("# Trades") or 0)
        if n_trades < args.min_trades:
            skipped_trades += 1
            continue
        max_dd = stats.get("Max. Drawdown [%]")
        # max_dd is reported as a negative number; compare its magnitude.
        if max_dd is not None and abs(float(max_dd)) > args.max_dd:
            skipped_dd += 1
            continue

        row = {**params}
        for k in REPORT_STATS:
            row[k] = stats.get(k)
        rows.append(row)

        if n % 50 == 0 or n == len(combos):
            elapsed = time.time() - t_start
            rate = n / elapsed if elapsed > 0 else 0.0
            eta = (len(combos) - n) / rate if rate > 0 else 0.0
            print(
                f"  [{n}/{len(combos)}] keep={len(rows)} skip_dd={skipped_dd} "
                f"skip_trades={skipped_trades} err={errors} "
                f"rate={rate:.1f}/s eta={eta/60:.1f}m",
                flush=True,
            )

    if not rows:
        print("\nNo configurations passed the filters. Loosen --max-dd / --min-trades, "
              "or widen the grid.", file=sys.stderr)
        return 2

    df_res = pd.DataFrame(rows)
    df_res = df_res.sort_values("Sharpe Ratio", ascending=False, na_position="last")

    print(f"\nTop {min(args.top, len(df_res))} by Sharpe (max DD ≤ {args.max_dd}%, "
          f"trades ≥ {args.min_trades}):\n")
    with pd.option_context("display.max_columns", None, "display.width", 200, "display.float_format", "{:.4f}".format):
        print(df_res.head(args.top).to_string(index=False))

    out_path = Path(args.csv) if args.csv else Path(__file__).with_suffix(".csv")
    df_res.to_csv(out_path, index=False)
    print(f"\nFull results ({len(df_res)} rows) → {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
