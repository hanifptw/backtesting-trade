"""Vectorized indicator functions, designed to be used with `Strategy.I()`.

Each function accepts numpy arrays / pandas Series and returns numpy arrays
of the same length (with NaN padding at the start where insufficient history).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(values, period: int) -> np.ndarray:
    """Simple moving average. Returns a writable numpy array."""
    return np.array(pd.Series(values).rolling(int(period)).mean(), dtype=float)


def rsi(values, period: int = 14) -> np.ndarray:
    """Relative Strength Index — Wilder's smoothing (RMA), TradingView-compatible.

    RMA is equivalent to EMA with alpha = 1/period. This is what TradingView and
    Binance use for RSI. Plain SMA gives noticeably different values.
    """
    close = pd.Series(values).astype(float)
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / int(period), adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / int(period), adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi_series = 100.0 - (100.0 / (1.0 + rs))
    return np.array(rsi_series, dtype=float)


def stoch_rsi_kd(
    close,
    rsi_period: int = 14,
    stoch_period: int = 14,
    smooth_k: int = 3,
    smooth_d: int = 3,
) -> np.ndarray:
    """Stochastic RSI — matches TradingView / Binance "Stoch RSI" indicator.

    Formula:
        1. RSI(rsi_period) of close
        2. raw_K = (RSI - lowest_RSI(stoch_period)) / (highest_RSI(stoch_period) - lowest_RSI(stoch_period)) * 100
        3. %K   = SMA(raw_K, smooth_k)
        4. %D   = SMA(%K,    smooth_d)

    Returns shape (2, N): row 0 = %K, row 1 = %D.
    """
    rsi_vals = pd.Series(rsi(close, rsi_period))
    lowest = rsi_vals.rolling(int(stoch_period)).min()
    highest = rsi_vals.rolling(int(stoch_period)).max()
    rng = (highest - lowest).replace(0.0, np.nan)
    raw_k = 100.0 * (rsi_vals - lowest) / rng
    k = raw_k.rolling(int(smooth_k)).mean()
    d = k.rolling(int(smooth_d)).mean()
    return np.array(np.vstack([k.values, d.values]), dtype=float)


def stoch_k_line(close, rsi_period: int = 14, stoch_period: int = 14, smooth_k: int = 3, smooth_d: int = 3) -> np.ndarray:
    """Stoch RSI %K line. Convenience wrapper for `Strategy.I()`."""
    return np.array(stoch_rsi_kd(close, rsi_period, stoch_period, smooth_k, smooth_d)[0], dtype=float)


def stoch_d_line(close, rsi_period: int = 14, stoch_period: int = 14, smooth_k: int = 3, smooth_d: int = 3) -> np.ndarray:
    """Stoch RSI %D line. Convenience wrapper for `Strategy.I()`."""
    return np.array(stoch_rsi_kd(close, rsi_period, stoch_period, smooth_k, smooth_d)[1], dtype=float)


def atr(high, low, close, period: int = 10, method: str = "rma") -> np.ndarray:
    """Average True Range with selectable smoothing.

    method:
        rma: Wilder/RMA smoothing (TradingView-style ATR)
        sma: Simple moving average
        ema: Exponential moving average
    """
    period = int(period)
    method = str(method).lower()
    high_s = pd.Series(high).astype(float)
    low_s = pd.Series(low).astype(float)
    close_s = pd.Series(close).astype(float)

    prev_close = close_s.shift(1)
    tr = pd.concat(
        [
            high_s - low_s,
            (high_s - prev_close).abs(),
            (low_s - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    if method == "rma":
        values = tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    elif method == "sma":
        values = tr.rolling(period).mean()
    elif method == "ema":
        values = tr.ewm(span=period, adjust=False, min_periods=period).mean()
    else:
        raise ValueError("atr_method must be one of: rma, sma, ema")

    return np.array(values, dtype=float)


def supertrend(
    high,
    low,
    close,
    atr_period: int = 10,
    multiplier: float = 3.0,
    atr_method: str = "rma",
) -> np.ndarray:
    """Supertrend indicator.

    Returns shape (2, N):
        row 0 = Supertrend line
        row 1 = direction (1 bullish, -1 bearish)
    """
    high_arr = np.asarray(high, dtype=float)
    low_arr = np.asarray(low, dtype=float)
    close_arr = np.asarray(close, dtype=float)
    atr_arr = atr(high_arr, low_arr, close_arr, atr_period, atr_method)

    hl2 = (high_arr + low_arr) / 2.0
    basic_upper = hl2 + float(multiplier) * atr_arr
    basic_lower = hl2 - float(multiplier) * atr_arr

    n = len(close_arr)
    final_upper = np.full(n, np.nan, dtype=float)
    final_lower = np.full(n, np.nan, dtype=float)
    line = np.full(n, np.nan, dtype=float)
    direction = np.full(n, np.nan, dtype=float)

    first_valid = None
    for i in range(n):
        if np.isfinite(atr_arr[i]):
            first_valid = i
            break
    if first_valid is None:
        return np.array(np.vstack([line, direction]), dtype=float)

    final_upper[first_valid] = basic_upper[first_valid]
    final_lower[first_valid] = basic_lower[first_valid]
    direction[first_valid] = 1.0 if close_arr[first_valid] >= hl2[first_valid] else -1.0
    line[first_valid] = (
        final_lower[first_valid] if direction[first_valid] > 0 else final_upper[first_valid]
    )

    for i in range(first_valid + 1, n):
        if not np.isfinite(atr_arr[i]):
            continue

        prev_upper = final_upper[i - 1]
        prev_lower = final_lower[i - 1]
        prev_close = close_arr[i - 1]
        prev_direction = direction[i - 1]

        final_upper[i] = (
            basic_upper[i]
            if basic_upper[i] < prev_upper or prev_close > prev_upper
            else prev_upper
        )
        final_lower[i] = (
            basic_lower[i]
            if basic_lower[i] > prev_lower or prev_close < prev_lower
            else prev_lower
        )

        if prev_direction < 0 and close_arr[i] > final_upper[i]:
            direction[i] = 1.0
        elif prev_direction > 0 and close_arr[i] < final_lower[i]:
            direction[i] = -1.0
        else:
            direction[i] = prev_direction

        line[i] = final_lower[i] if direction[i] > 0 else final_upper[i]

    return np.array(np.vstack([line, direction]), dtype=float)


def supertrend_line(
    high,
    low,
    close,
    atr_period: int = 10,
    multiplier: float = 3.0,
    atr_method: str = "rma",
) -> np.ndarray:
    """Supertrend price line. Convenience wrapper for `Strategy.I()`."""
    return np.array(supertrend(high, low, close, atr_period, multiplier, atr_method)[0], dtype=float)


def supertrend_direction(
    high,
    low,
    close,
    atr_period: int = 10,
    multiplier: float = 3.0,
    atr_method: str = "rma",
) -> np.ndarray:
    """Supertrend direction: 1 bullish, -1 bearish."""
    return np.array(supertrend(high, low, close, atr_period, multiplier, atr_method)[1], dtype=float)


def supertrend_bullish_line(
    high,
    low,
    close,
    atr_period: int = 10,
    multiplier: float = 3.0,
    atr_method: str = "rma",
) -> np.ndarray:
    """Supertrend line only while direction is bullish; NaN otherwise."""
    values = supertrend(high, low, close, atr_period, multiplier, atr_method)
    line = values[0]
    direction = values[1]
    return np.where(direction > 0, line, np.nan).astype(float)


def supertrend_bearish_line(
    high,
    low,
    close,
    atr_period: int = 10,
    multiplier: float = 3.0,
    atr_method: str = "rma",
) -> np.ndarray:
    """Supertrend line only while direction is bearish; NaN otherwise."""
    values = supertrend(high, low, close, atr_period, multiplier, atr_method)
    line = values[0]
    direction = values[1]
    return np.where(direction < 0, line, np.nan).astype(float)
