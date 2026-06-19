"""
src/features.py
---------------
Constructs a feature vector from a TransitCandidate for the ML classifier.
Also handles the labelled training dataset (curated CSV).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Feature definitions
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "sde",
    "snr",
    "depth_ppm",
    "log_depth_ppm",
    "duration_hr",
    "duration_ratio",         # duration / period
    "rp_rs",
    "log_rp_rs",
    "n_transits",
    "odd_even_diff",
    "secondary_depth",
    "secondary_depth_ratio",  # secondary / primary depth
    "shape_score",
    "scatter_in",             # std of in-transit flux
    "scatter_out",            # std of out-of-transit flux
    "scatter_ratio",          # scatter_in / scatter_out
    # BLS-derived features
    "bls_sde",                # BLS Signal Detection Efficiency
    "bls_snr",                # BLS signal-to-noise
    "bls_power",              # BLS peak power (SR statistic)
    "period_agreement",       # TLS/BLS period agreement (0 = match)
]

CLASS_LABELS = {
    0: "PLANET",
    1: "EB",       # Eclipsing Binary
    2: "BLEND",
    3: "OTHER",
}

LABEL_MAP = {v: k for k, v in CLASS_LABELS.items()}


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features_from_candidate(
    candidate,  # TransitCandidate
    time: Optional[np.ndarray] = None,
    flux: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Build a feature dict from a TransitCandidate object.

    Parameters
    ----------
    candidate : TransitCandidate
    time, flux : optional arrays for computing scatter metrics

    Returns
    -------
    dict of feature_name → float
    """
    depth = max(candidate.depth_ppm, 1.0)  # avoid log(0)
    rp_rs = max(candidate.rp_rs, 1e-6)
    period = max(candidate.period, 1e-6)

    features: Dict[str, float] = {
        "sde":                  candidate.sde,
        "snr":                  candidate.snr,
        "depth_ppm":            depth,
        "log_depth_ppm":        np.log10(depth),
        "duration_hr":          candidate.duration,
        "duration_ratio":       (candidate.duration / 24.0) / period,
        "rp_rs":                rp_rs,
        "log_rp_rs":            np.log10(rp_rs),
        "n_transits":           float(candidate.n_transits),
        "odd_even_diff":        candidate.odd_even_diff,
        "secondary_depth":      candidate.secondary_depth * 1e6,  # convert to ppm
        "secondary_depth_ratio": candidate.secondary_depth / max(candidate.depth, 1e-12),
        "shape_score":          candidate.transit_shape_score,
        "scatter_in":           0.0,
        "scatter_out":          0.0,
        "scatter_ratio":        1.0,
        # BLS features
        "bls_sde":              getattr(candidate, "bls_sde", 0.0),
        "bls_snr":              getattr(candidate, "bls_snr", 0.0),
        "bls_power":            getattr(candidate, "bls_power", 0.0),
        "period_agreement":     getattr(candidate, "period_agreement", 1.0),
    }

    # Compute scatter metrics if time/flux available
    if time is not None and flux is not None and candidate.period > 0:
        scatter_in, scatter_out = _compute_scatter(
            time, flux, candidate.period, candidate.t0, candidate.duration / 24.0
        )
        features["scatter_in"] = scatter_in
        features["scatter_out"] = scatter_out
        features["scatter_ratio"] = scatter_in / max(scatter_out, 1e-12)

    return features


def _compute_scatter(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    t0: float,
    duration: float,
) -> Tuple[float, float]:
    """Return std of flux inside and outside transit windows."""
    phase = ((time - t0) % period) / period
    half_dur = (duration / period) / 2.0
    # Handle wrap-around: in-transit near phase 0 or phase 1
    in_mask = (phase < half_dur) | (phase > 1.0 - half_dur)
    out_mask = ~in_mask

    std_in = float(np.nanstd(flux[in_mask])) if in_mask.sum() > 1 else 0.0
    std_out = float(np.nanstd(flux[out_mask])) if out_mask.sum() > 1 else 0.0
    return std_in, std_out


def features_to_array(
    features: Dict[str, float],
    columns: List[str] = FEATURE_COLS,
) -> np.ndarray:
    """Convert feature dict to a fixed-order numpy array."""
    return np.array([features.get(col, 0.0) for col in columns], dtype=np.float32)


# ---------------------------------------------------------------------------
# Training dataset handling
# ---------------------------------------------------------------------------

def load_labelled_dataset(
    csv_path: Path,
    label_col: str = "label",
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Load a curated labelled dataset CSV.

    Expected CSV columns (at minimum):
        - One or more feature columns (see FEATURE_COLS)
        - A label column (str: 'PLANET', 'EB', 'BLEND', 'OTHER')

    Returns
    -------
    X : pd.DataFrame of features
    y : np.ndarray of integer class labels
    """
    df = pd.read_csv(csv_path)

    # Map string labels to integers
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in {csv_path}.")

    # Normalise label strings
    df[label_col] = df[label_col].str.upper().str.strip()

    # Handle common alternative label names
    alias = {
        "PC":     "PLANET",
        "PLANET": "PLANET",
        "CONFIRMED": "PLANET",
        "KP":     "PLANET",
        "TRANSIT": "PLANET",
        "EB":     "EB",
        "ECLIPSING BINARY": "EB",
        "FP":     "EB",
        "FALSE POSITIVE": "EB",
        "BEB":    "BLEND",
        "BLEND":  "BLEND",
        "BACKGROUND EB": "BLEND",
        "OTHER":  "OTHER",
        "NTP":    "OTHER",
    }
    df[label_col] = df[label_col].map(alias).fillna("OTHER")
    y = df[label_col].map(LABEL_MAP).values

    # Select available feature columns
    avail = [c for c in FEATURE_COLS if c in df.columns]
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        logger.warning(f"Missing feature columns (will fill with 0): {missing}")
        for col in missing:
            df[col] = 0.0

    X = df[FEATURE_COLS]
    return X, y


def build_feature_dataframe(
    candidates: List,  # List[TransitCandidate]
    processed_dir: Path = Path("data/processed"),
) -> pd.DataFrame:
    """
    Build a feature DataFrame from a list of TransitCandidates.
    Loads corresponding processed CSVs for scatter metrics.
    """
    rows = []
    for cand in candidates:
        # Try to load the LC for scatter metrics
        csv_candidates = list(processed_dir.glob(f"TIC{cand.tic_id}*.csv"))
        if csv_candidates:
            try:
                lc_df = pd.read_csv(csv_candidates[0])
                time = lc_df["time"].values
                flux = lc_df["flux"].values
            except Exception:
                time, flux = None, None
        else:
            time, flux = None, None

        feats = extract_features_from_candidate(cand, time, flux)
        feats["tic_id"] = cand.tic_id
        feats["period_d"] = cand.period
        feats["passed_threshold"] = cand.passed_threshold
        rows.append(feats)

    return pd.DataFrame(rows)
