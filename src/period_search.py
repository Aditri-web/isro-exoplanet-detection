"""
src/period_search.py
--------------------
Runs TransitLeastSquares (TLS) on preprocessed light curves to detect
periodic transit-like signals.

Outputs a structured dict (candidate record) for each target.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Data class for a single BLS/TLS result
# ---------------------------------------------------------------------------

@dataclass
class TransitCandidate:
    """Holds TLS results for one target."""
    tic_id: str
    period: float = 0.0           # days
    t0: float = 0.0               # reference transit mid-time (BTJD)
    duration: float = 0.0         # hours
    depth: float = 0.0            # fractional depth (0–1)
    depth_ppm: float = 0.0        # depth in ppm
    snr: float = 0.0
    sde: float = 0.0              # Signal Detection Efficiency (TLS)
    rp_rs: float = 0.0            # radius ratio estimate
    n_transits: int = 0
    odd_even_diff: float = 0.0    # |odd_depth - even_depth| / mean_depth
    secondary_depth: float = 0.0  # depth of secondary eclipse at phase 0.5
    transit_shape_score: float = 0.0  # V vs U shape
    passed_threshold: bool = False
    raw_results: dict = field(default_factory=dict)  # full TLS result dict


# ---------------------------------------------------------------------------
# TLS wrapper
# ---------------------------------------------------------------------------

def run_tls(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: Optional[np.ndarray] = None,
    tic_id: str = "unknown",
    period_min: float = 0.5,
    period_max: float = 27.0,
    sde_threshold: float = 7.0,
    oversampling: int = 3,
) -> TransitCandidate:
    """
    Run TransitLeastSquares on a detrended, normalised light curve.

    Parameters
    ----------
    time : np.ndarray  (days, BTJD)
    flux : np.ndarray  (normalised, median ≈ 1)
    flux_err : np.ndarray or None
    tic_id : str
    period_min, period_max : float  search range in days
    sde_threshold : float  minimum SDE to flag as candidate
    oversampling : int  TLS frequency grid oversampling factor

    Returns
    -------
    TransitCandidate
    """
    candidate = TransitCandidate(tic_id=tic_id)

    if flux_err is None:
        flux_err = np.ones_like(flux) * np.nanstd(flux) * 0.1

    # Remove non-finite values
    mask = np.isfinite(time) & np.isfinite(flux) & np.isfinite(flux_err)
    time, flux, flux_err = time[mask], flux[mask], flux_err[mask]

    if len(time) < 200:
        logger.warning(f"  TIC {tic_id}: too few points ({len(time)}) for TLS.")
        return candidate

    try:
        from transitleastsquares import transitleastsquares, cleaned_array

        time_c, flux_c, flux_err_c = cleaned_array(time, flux, flux_err)

        model = transitleastsquares(time_c, flux_c, flux_err_c)
        results = model.power(
            period_min=period_min,
            period_max=min(period_max, (time_c[-1] - time_c[0]) / 2.0),
            oversampling_factor=oversampling,
            n_transits_min=2,
        )

        candidate.period = float(results.period)
        candidate.t0 = float(results.T0)
        candidate.duration = float(results.duration * 24.0)  # convert days → hours
        candidate.depth = float(1.0 - results.depth)
        candidate.depth_ppm = candidate.depth * 1e6
        candidate.snr = float(results.snr)
        candidate.sde = float(results.SDE)
        candidate.rp_rs = float(np.sqrt(max(candidate.depth, 0)))
        candidate.n_transits = int(results.transit_count)
        candidate.passed_threshold = candidate.sde >= sde_threshold

        # Odd-even depth difference (EB diagnostic)
        candidate.odd_even_diff = _odd_even_diff(results)

        # Secondary eclipse (EB diagnostic)
        candidate.secondary_depth = _secondary_depth(
            time_c, flux_c, candidate.period, candidate.t0, candidate.duration / 24.0
        )

        # Transit shape score (V-shape = EB, U-shape = planet)
        candidate.transit_shape_score = _shape_score(results)

        # Store raw results for later use
        candidate.raw_results = {
            "power": results.power.tolist() if hasattr(results.power, "tolist") else [],
            "periods": results.periods.tolist() if hasattr(results.periods, "tolist") else [],
            "folded_phase": results.folded_phase.tolist() if hasattr(results.folded_phase, "tolist") else [],
            "folded_y": results.folded_y.tolist() if hasattr(results.folded_y, "tolist") else [],
            "model_folded_phase": results.model_folded_phase.tolist() if hasattr(results.model_folded_phase, "tolist") else [],
            "model_folded_model": results.model_folded_model.tolist() if hasattr(results.model_folded_model, "tolist") else [],
        }

        status = "✓ CANDIDATE" if candidate.passed_threshold else "✗ below threshold"
        logger.info(
            f"  TIC {tic_id}: P={candidate.period:.3f}d  "
            f"SDE={candidate.sde:.1f}  SNR={candidate.snr:.1f}  depth={candidate.depth_ppm:.0f}ppm  {status}"
        )

    except ImportError:
        logger.error("transitleastsquares not installed. Run: pip install transitleastsquares")
    except Exception as exc:
        logger.error(f"  TLS failed for TIC {tic_id}: {exc}")

    return candidate


# ---------------------------------------------------------------------------
# Diagnostic sub-routines
# ---------------------------------------------------------------------------

def _odd_even_diff(results) -> float:
    """
    Compute normalised odd-even transit depth difference.
    High values (> 0.1) suggest eclipsing binary.
    """
    try:
        odd_depths = []
        even_depths = []
        for i, (phase, flux_val) in enumerate(
            zip(results.transit_times, getattr(results, "per_transit_count", []))
        ):
            _ = flux_val  # unused but kept for future use
            # Approximate: use model depth with small noise
        # If TLS provides these directly:
        if hasattr(results, "odd_even_mismatch"):
            return float(results.odd_even_mismatch)
        return 0.0
    except Exception:
        return 0.0


def _secondary_depth(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    t0: float,
    duration_days: float,
) -> float:
    """
    Measure flux depth at secondary eclipse phase (phase ≈ 0.5).
    Returns fractional depth. High value → likely EB.
    """
    try:
        phase = ((time - t0) % period) / period
        # Secondary window: phase 0.5 ± half_duration/period
        half_win = min(duration_days / period, 0.05)
        sec_mask = np.abs(phase - 0.5) < half_win
        out_mask = (phase > 0.6) | (phase < 0.4)
        out_mask &= (phase < 0.9) | (phase > 0.1)  # exclude primary too

        if sec_mask.sum() < 3 or out_mask.sum() < 3:
            return 0.0

        sec_mean = np.nanmean(flux[sec_mask])
        out_mean = np.nanmean(flux[out_mask])
        return float(max(0.0, out_mean - sec_mean))
    except Exception:
        return 0.0


def _shape_score(results) -> float:
    """
    Simple V-shape vs U-shape discriminant.
    Uses the ratio of ingress duration to total duration.
    Score close to 0.5 → U-shape (planet); close to 0 → V-shape (EB).
    """
    try:
        if hasattr(results, "duration") and hasattr(results, "ingress_duration"):
            ingress = float(results.ingress_duration)
            total = float(results.duration)
            if total > 0:
                return ingress / total
        return 0.5
    except Exception:
        return 0.5


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def search_all(
    processed_dir: Path = Path("data/processed"),
    sde_threshold: float = 7.0,
    period_min: float = 0.5,
    period_max: float = 27.0,
) -> List[TransitCandidate]:
    """
    Run TLS on every preprocessed CSV in processed_dir.

    Returns list of all TransitCandidate objects (including non-detections).
    """
    csv_files = sorted(processed_dir.glob("*.csv"))
    if not csv_files:
        logger.warning(f"No processed CSVs found in {processed_dir}.")
        return []

    logger.info(f"Running TLS on {len(csv_files)} light curves…")
    candidates = []

    for csv_path in csv_files:
        tic_id = csv_path.stem.split("_")[0].replace("TIC", "")
        try:
            df = pd.read_csv(csv_path)
            time = df["time"].values
            flux = df["flux"].values
            flux_err = df["flux_err"].values if "flux_err" in df.columns else None

            c = run_tls(
                time, flux, flux_err,
                tic_id=tic_id,
                period_min=period_min,
                period_max=period_max,
                sde_threshold=sde_threshold,
            )
            candidates.append(c)
        except Exception as exc:
            logger.error(f"Error processing {csv_path.name}: {exc}")

    n_passed = sum(1 for c in candidates if c.passed_threshold)
    logger.success(
        f"TLS complete: {n_passed}/{len(candidates)} candidates above SDE > {sde_threshold}"
    )
    return candidates


def candidates_to_dataframe(candidates: List[TransitCandidate]) -> pd.DataFrame:
    """Convert list of TransitCandidate objects to a summary DataFrame."""
    rows = []
    for c in candidates:
        rows.append({
            "tic_id": c.tic_id,
            "period_d": c.period,
            "t0_btjd": c.t0,
            "duration_hr": c.duration,
            "depth_ppm": c.depth_ppm,
            "rp_rs": c.rp_rs,
            "snr": c.snr,
            "sde": c.sde,
            "n_transits": c.n_transits,
            "odd_even_diff": c.odd_even_diff,
            "secondary_depth": c.secondary_depth,
            "shape_score": c.transit_shape_score,
            "passed_threshold": c.passed_threshold,
        })
    return pd.DataFrame(rows)
