"""Hidden Markov Model — Market Regime Detection Strategy.

Mengklasifikasikan kondisi pasar ke dalam 3 regime menggunakan Gaussian HMM:
    0 = Bear     (return negatif, volatilitas tinggi)
    1 = Sideways (sideways, volatilitas sedang)
    2 = Bull     (return positif, tren naik)

HMM dilatih di init() menggunakan seluruh data yang dimasukkan ke backtest.
Gunakan data BTC/USDT 2019–2023 sebagai periode training yang direkomendasikan.

Sub-strategy per regime belum diimplementasi — fase ini hanya deteksi + visualisasi.

Cara kerja:
    Features: log_returns, rolling_volatility, rolling_trend (ketiganya per-bar)
    Model   : GaussianHMM (full covariance, 3 komponen)
    Labeling: state di-sort berdasarkan mean log-return ascending → bear(0)…bull(2)
    Output  : 3 indikator berwarna di chart (satu per regime)
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from backtesting import Strategy

# --- Regime metadata -------------------------------------------------------

_N_REGIMES = 3

_REGIME_META: list[dict] = [
    {"label": "Bear",     "color": "tomato"},
    {"label": "Sideways", "color": "silver"},
    {"label": "Bull",     "color": "mediumseagreen"},
]


# --- Feature engineering ---------------------------------------------------

def _build_features(
    close: np.ndarray,
    volume: np.ndarray,
    lookback: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, valid_indices) where X has shape (M, 3).

    Features (all computed per-bar):
        col 0 — log return
        col 1 — rolling std of log returns (volatility proxy)
        col 2 — rolling mean of log returns (trend proxy)
    """
    close_s = pd.Series(np.asarray(close, dtype=float))
    volume_s = pd.Series(np.asarray(volume, dtype=float))

    log_ret = np.log(close_s / close_s.shift(1))
    roll_vol = log_ret.rolling(int(lookback)).std()
    roll_mean = log_ret.rolling(int(lookback)).mean()

    # Optional: log-volume change as extra signal
    log_vol_chg = np.log(volume_s / volume_s.shift(1)).replace([np.inf, -np.inf], np.nan)

    X_df = pd.concat(
        [log_ret, roll_vol, roll_mean, log_vol_chg],
        axis=1,
    )
    X_df.columns = ["returns", "volatility", "trend", "vol_change"]

    valid_mask = X_df.notna().all(axis=1)
    X_valid = X_df.loc[valid_mask].values
    valid_indices = np.where(valid_mask.values)[0]
    return X_valid, valid_indices


# --- Strategy class --------------------------------------------------------

class HMMRegimeStrategy(Strategy):
    """HMM regime detection — visualization only, no trade logic yet."""

    # HMM training
    n_iter: int = 300
    lookback_vol: int = 20
    random_state: int = 42

    def init(self) -> None:  # noqa: D401
        try:
            from hmmlearn.hmm import GaussianHMM
            from sklearn.preprocessing import StandardScaler
        except ImportError as exc:
            raise ImportError(
                "hmmlearn dan scikit-learn diperlukan. "
                "Jalankan: uv add hmmlearn scikit-learn"
            ) from exc

        close = np.asarray(self.data.Close, dtype=float)
        volume = np.asarray(self.data.Volume, dtype=float)
        n = len(close)

        # 1. Build & scale features
        X_raw, valid_idx = _build_features(close, volume, self.lookback_vol)
        if len(X_raw) < _N_REGIMES * 10:
            # Too few bars — fill everything with Neutral
            full_regime = np.full(n, 2.0)
        else:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_raw)

            # 2. Train HMM
            model = GaussianHMM(
                n_components=_N_REGIMES,
                covariance_type="full",
                n_iter=self.n_iter,
                random_state=self.random_state,
            )
            model.fit(X_scaled)

            # 3. Predict hidden states
            raw_states = model.predict(X_scaled)

            # 4. Map raw HMM states → ordered regime labels
            #    Sort by mean log-return: lowest mean → Crash (0), highest → Euphoria (4)
            mean_returns = np.array([
                X_raw[raw_states == s, 0].mean()
                if (raw_states == s).any() else 0.0
                for s in range(_N_REGIMES)
            ])
            order = np.argsort(mean_returns)           # ascending
            state_to_regime = {int(order[i]): i for i in range(_N_REGIMES)}
            regimes = np.array(
                [state_to_regime[int(s)] for s in raw_states],
                dtype=float,
            )

            # 5. Embed into full-length array (NaN for warm-up bars)
            full_regime = np.full(n, np.nan)
            full_regime[valid_idx] = regimes

        # 6. Register one indicator per regime — each visible only in its regime bars
        for regime_id, meta in enumerate(_REGIME_META):
            band = np.where(
                np.isfinite(full_regime) & (full_regime == regime_id),
                float(regime_id),
                np.nan,
            )
            setattr(
                self,
                f"_band_{regime_id}",
                self.I(
                    lambda x: x,   # noqa: E731  — identity; arg already computed
                    band.copy(),
                    name=f"REGIME_BG:{meta['color']}:{meta['label']}",
                    overlay=False,
                    plot=True,
                ),
            )

        # 7. Single combined line (hidden from chart — used internally)
        self._regime = self.I(
            lambda x: x,
            full_regime.copy(),
            name="HMM Regime (0=Bear, 1=Sideways, 2=Bull)",
            overlay=False,
            plot=False,
        )

    def next(self) -> None:
        """No trade logic yet — regime mapping only."""
        regime = float(self._regime[-1])
        if math.isnan(regime):
            return

        # === Placeholder: sub-strategy per regime will be added here ===
        # regime 0 → Bear     strategy (e.g., short-only / stay flat)
        # regime 1 → Sideways strategy (e.g., range-bound mean reversion)
        # regime 2 → Bull     strategy (e.g., trend-following long)


# --- Parameter schema (auto-rendered by StrategyPicker) -------------------

SCHEMA: dict = {
    "n_iter": {
        "type": "int",
        "default": 300,
        "min": 50,
        "max": 2000,
        "label": "HMM training iterations (EM steps)",
        "group": "HMM",
    },
    "lookback_vol": {
        "type": "int",
        "default": 20,
        "min": 5,
        "max": 200,
        "label": "Rolling window — volatility & trend features",
        "group": "HMM",
    },
    "random_state": {
        "type": "int",
        "default": 42,
        "min": 0,
        "max": 9999,
        "label": "Random seed (reproducibility)",
        "group": "HMM",
    },
}
