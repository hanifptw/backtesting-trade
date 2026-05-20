"""Orchestrate a single backtest run and serialize results for the frontend.

Output shape:
    {
        "stats": {"Return [%]": ..., "Sharpe Ratio": ..., ...},
        "equity_curve": [{"time": <unix_s>, "equity": ..., "drawdown_pct": ...}, ...],
        "trades":       [{"entry_time", "exit_time", "entry_price", "exit_price",
                          "size", "pl", "pl_pct", "is_long"}, ...],
        "indicators":   [{"name": "SMA(10)", "overlay": true, "values": [...]}, ...]
    }
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from backtesting import Backtest
from backtesting.lib import FractionalBacktest

from app.core.strategies import REGISTRY, get


REQUIRED_COLS = {"Open", "High", "Low", "Close", "Volume"}


def _safe_value(v: Any) -> Any:
    """Convert pandas/numpy scalars to JSON-safe types. NaN/Inf → None."""
    if v is None:
        return None
    if isinstance(v, (pd.Timestamp, pd.Timedelta)):
        return str(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if hasattr(v, "item"):
        try:
            return _safe_value(v.item())
        except Exception:
            return str(v)
    return v


def _candles_to_df(candles: list[dict]) -> pd.DataFrame:
    if not candles:
        raise ValueError("`candles` is empty")
    df = pd.DataFrame(candles)
    expected = {"time", "open", "high", "low", "close", "volume"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Candles missing columns: {missing}")
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.rename(columns={
        "open": "Open", "high": "High", "low": "Low",
        "close": "Close", "volume": "Volume",
    }).set_index("time").sort_index()
    return df[["Open", "High", "Low", "Close", "Volume"]].astype(float)


def _build_runtime_class(base_cls, params: dict):
    """Create a subclass with `params` bound as class attributes.

    Avoids mutating the original class (which would be racy across requests).
    """
    return type(f"{base_cls.__name__}_runtime", (base_cls,), dict(params))


def _serialize_equity(equity_df: pd.DataFrame) -> list[dict]:
    out = []
    for ts, row in equity_df.iterrows():
        out.append({
            "time": int(pd.Timestamp(ts).timestamp()),
            "equity": _safe_value(row["Equity"]),
            "drawdown_pct": _safe_value(row["DrawdownPct"]),
        })
    return out


def _serialize_trades(trades_df: pd.DataFrame) -> list[dict]:
    has_tag = "Tag" in trades_df.columns
    out = []
    for _, row in trades_df.iterrows():
        size = float(row["Size"])
        tag = row["Tag"] if has_tag else None
        k_at_entry = tag.get("k") if isinstance(tag, dict) else None
        d_at_entry = tag.get("d") if isinstance(tag, dict) else None
        out.append({
            "entry_time": int(pd.Timestamp(row["EntryTime"]).timestamp()),
            "exit_time": int(pd.Timestamp(row["ExitTime"]).timestamp()),
            "entry_price": _safe_value(row["EntryPrice"]),
            "exit_price": _safe_value(row["ExitPrice"]),
            "size": size,
            "pl": _safe_value(row["PnL"]),
            "pl_pct": _safe_value(row["ReturnPct"] * 100.0),
            "is_long": size > 0,
            "k_at_entry": k_at_entry,
            "d_at_entry": d_at_entry,
        })
    return out


def _is_oscillator(values: np.ndarray) -> bool:
    """Heuristic: indicator in [0, 100] range → plot in separate pane."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return False
    return float(finite.min()) >= -1.0 and float(finite.max()) <= 101.0


def _serialize_indicators(strategy, index: pd.DatetimeIndex) -> list[dict]:
    times = [int(ts.timestamp()) for ts in index]
    out: list[dict] = []
    for ind in getattr(strategy, "_indicators", []):
        opts = getattr(ind, "_opts", {})
        if opts.get("plot") is False:
            continue

        arr = np.asarray(ind)
        name_attr = getattr(ind, "name", None) or "indicator"

        if arr.ndim == 1:
            lines = [(str(name_attr), arr)]
        else:
            if isinstance(name_attr, (list, tuple)) and len(name_attr) == arr.shape[0]:
                names = [str(n) for n in name_attr]
            else:
                names = [f"{name_attr}[{i}]" for i in range(arr.shape[0])]
            lines = list(zip(names, arr))

        for line_name, values in lines:
            values = np.asarray(values, dtype=float)
            n = min(len(values), len(times))
            series = []
            for i in range(n):
                v = values[i]
                if np.isnan(v) or np.isinf(v):
                    # Whitespace point — keeps the time slot so the oscillator
                    # pane stays index-aligned with the price pane.
                    series.append({"time": times[i]})
                else:
                    series.append({"time": times[i], "value": float(v)})

            if line_name.startswith("REGIME_BG:"):
                parts = line_name.split(":", 2)
                color = parts[1] if len(parts) > 1 else "gray"
                label = parts[2] if len(parts) > 2 else line_name
                out.append({
                    "name": label,
                    "type": "regime_bg",
                    "color": color,
                    "overlay": True,
                    "values": series,
                })
                continue

            out.append({
                "name": line_name,
                "overlay": not _is_oscillator(values),
                "values": series,
            })
    return out


def run_backtest(
    candles: list[dict],
    strategy_name: str,
    params: dict | None = None,
    cash: float = 10_000.0,
    commission: float = 0.0004,
    margin: float | None = None,
    trade_on_close: bool = True,
    hedging: bool = False,
    exclusive_orders: bool = False,
) -> dict:
    df = _candles_to_df(candles)
    entry = get(strategy_name)
    params = params or {}

    # If the strategy declares a `leverage` param, derive margin from it
    # (margin = 1 / leverage). Explicit `margin` in the request overrides.
    # Strategies can opt out at runtime with `leverage_enabled=False`.
    if margin is None:
        leverage_enabled = params.get(
            "leverage_enabled",
            getattr(entry["cls"], "leverage_enabled", True),
        )
        if leverage_enabled is False:
            margin = 1.0
        else:
            leverage = params.get("leverage")
            if leverage is None:
                # Fall back to the class default if any
                leverage = getattr(entry["cls"], "leverage", 1)
            try:
                leverage = float(leverage)
                margin = 1.0 / leverage if leverage > 0 else 1.0
            except (TypeError, ValueError):
                margin = 1.0

    runtime_cls = _build_runtime_class(entry["cls"], params)

    # FractionalBacktest allows trading fractional units — essential for crypto
    # where 1 BTC may exceed initial cash. Same API as Backtest.
    bt = FractionalBacktest(
        df,
        runtime_cls,
        cash=cash,
        commission=commission,
        margin=margin,
        trade_on_close=trade_on_close,
        hedging=hedging,
        exclusive_orders=exclusive_orders,
    )
    stats = bt.run()

    metrics = {k: _safe_value(v) for k, v in stats.items() if not str(k).startswith("_")}
    equity_curve = _serialize_equity(stats["_equity_curve"])
    trades = _serialize_trades(stats["_trades"])
    indicators = _serialize_indicators(stats["_strategy"], df.index)

    return {
        "strategy": strategy_name,
        "params": params or {},
        "n_candles": len(df),
        "stats": metrics,
        "equity_curve": equity_curve,
        "trades": trades,
        "indicators": indicators,
    }


def list_strategies() -> list[dict]:
    return [
        {
            "id": sid,
            "label": entry["label"],
            "description": entry["description"],
            "schema": entry["schema"],
        }
        for sid, entry in REGISTRY.items()
    ]
