"""
src/period_search.py
--------------------
Runs TransitLeastSquares (TLS) and Box Least Squares (BLS) on
preprocessed light curves to detect periodic transit-like signals.

BLS (Kovács, Zucker & Mazeh 2002) is used via astropy.timeseries.
TLS (Hippke & Heller 2019) provides an optimised transit-shaped search.
Both algorithms are run in parallel for cross-validation.

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
    """Holds TLS + BLS results for one target."""
    tic_id: str
    period: float = 0.0           # days (best from TLS or BLS)
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

    # BLS-specific fields
    bls_period: float = 0.0       # BLS best-fit period (days)
    bls_t0: float = 0.0           # BLS transit mid-time
    bls_depth: float = 0.0        # BLS transit depth (fractional)
    bls_duration: float = 0.0     # BLS duration (hours)
    bls_power: float = 0.0        # BLS peak power (SR statistic)
    bls_sde: float = 0.0          # BLS Signal Detection Efficiency
    bls_snr: float = 0.0          # BLS signal-to-noise
    period_agreement: float = 0.0 # |P_TLS - P_BLS| / P_TLS (0 = perfect match)
    bls_raw: dict = field(default_factory=dict)  # full BLS periodogram


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
        candidate.odd_even_diff = _odd_even_diff(
            time_c, flux_c, candidate.period, candidate.t0,
            candidate.duration / 24.0,
        )

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
# BLS wrapper (astropy)
# ---------------------------------------------------------------------------

def run_bls(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: Optional[np.ndarray] = None,
    tic_id: str = "unknown",
    period_min: float = 0.5,
    period_max: float = 27.0,
    duration_min_hr: float = 0.5,
    duration_max_hr: float = 12.0,
    n_durations: int = 20,
    frequency_factor: float = 1.0,
) -> dict:
    """
    Run Box Least Squares (BLS) on a detrended, normalised light curve.

    Uses astropy.timeseries.BoxLeastSquares (Kovács, Zucker & Mazeh 2002).

    Parameters
    ----------
    time : np.ndarray  (days, BTJD)
    flux : np.ndarray  (normalised, median ≈ 1)
    flux_err : np.ndarray or None
    tic_id : str
    period_min, period_max : float  search range in days
    duration_min_hr, duration_max_hr : float  trial transit durations
    n_durations : int  number of trial durations
    frequency_factor : float  oversampling for the BLS frequency grid

    Returns
    -------
    dict with BLS results: period, t0, depth, duration, power, sde, snr,
         periods array, power_spectrum array.
    """
    result = {
        "period": 0.0, "t0": 0.0, "depth": 0.0, "duration_hr": 0.0,
        "power": 0.0, "sde": 0.0, "snr": 0.0,
        "periods": [], "power_spectrum": [],
    }

    # Remove non-finite values
    mask = np.isfinite(time) & np.isfinite(flux)
    if flux_err is not None:
        mask &= np.isfinite(flux_err)
    time_c, flux_c = time[mask], flux[mask]
    flux_err_c = flux_err[mask] if flux_err is not None else None

    if len(time_c) < 200:
        logger.warning(f"  TIC {tic_id}: too few points ({len(time_c)}) for BLS.")
        return result

    try:
        from astropy.timeseries import BoxLeastSquares
        import astropy.units as u

        # Build the BLS model
        if flux_err_c is not None:
            bls = BoxLeastSquares(time_c * u.day, flux_c, dy=flux_err_c)
        else:
            bls = BoxLeastSquares(time_c * u.day, flux_c)

        # Generate trial durations
        durations = np.linspace(
            duration_min_hr / 24.0, duration_max_hr / 24.0, n_durations
        ) * u.day

        # Compute the BLS periodogram
        # Limit period_max to half the time baseline
        effective_max = min(period_max, (time_c[-1] - time_c[0]) / 2.0)
        periodogram = bls.autopower(
            durations,
            minimum_period=period_min * u.day,
            maximum_period=effective_max * u.day,
            frequency_factor=frequency_factor,
        )

        # Extract best-fit parameters
        best_idx = np.argmax(periodogram.power)
        best_period = float(periodogram.period[best_idx].value)
        best_power = float(periodogram.power[best_idx])

        # Get transit parameters at best period
        stats = bls.compute_stats(
            best_period * u.day,
            float(periodogram.duration[best_idx].value) * u.day,
            float(periodogram.transit_time[best_idx].value) * u.day,
        )

        best_t0 = float(periodogram.transit_time[best_idx].value)
        best_depth = float(stats["depth"][0]) if "depth" in stats else float(periodogram.depth[best_idx])
        best_duration_hr = float(periodogram.duration[best_idx].value) * 24.0

        # Compute SDE: (peak - mean) / std of the power spectrum
        power_arr = periodogram.power.copy()
        power_mean = float(np.nanmean(power_arr))
        power_std = float(np.nanstd(power_arr))
        bls_sde = (best_power - power_mean) / max(power_std, 1e-12)

        # Compute SNR from the transit depth and out-of-transit scatter
        phase = ((time_c - best_t0) % best_period) / best_period
        half_dur_phase = (best_duration_hr / 24.0) / best_period / 2.0
        in_transit = (phase < half_dur_phase) | (phase > 1.0 - half_dur_phase)
        out_transit = ~in_transit
        if in_transit.sum() > 0 and out_transit.sum() > 0:
            bls_snr = abs(np.nanmean(flux_c[in_transit]) - np.nanmean(flux_c[out_transit])) / (
                np.nanstd(flux_c[out_transit]) + 1e-12
            )
        else:
            bls_snr = 0.0

        result = {
            "period": best_period,
            "t0": best_t0,
            "depth": best_depth,
            "duration_hr": best_duration_hr,
            "power": best_power,
            "sde": float(bls_sde),
            "snr": float(bls_snr),
            "periods": periodogram.period.value.tolist(),
            "power_spectrum": periodogram.power.tolist(),
        }

        logger.info(
            f"  TIC {tic_id} BLS: P={best_period:.3f}d  "
            f"SDE={bls_sde:.1f}  depth={best_depth*1e6:.0f}ppm  "
            f"dur={best_duration_hr:.2f}h"
        )

    except ImportError:
        logger.error("astropy.timeseries not available. Run: pip install astropy>=5.3")
    except Exception as exc:
        logger.error(f"  BLS failed for TIC {tic_id}: {exc}")

    return result


def merge_bls_into_candidate(
    candidate: TransitCandidate,
    bls_result: dict,
) -> TransitCandidate:
    """
    Merge BLS results into an existing TransitCandidate.
    Computes period agreement between TLS and BLS.
    """
    candidate.bls_period = bls_result.get("period", 0.0)
    candidate.bls_t0 = bls_result.get("t0", 0.0)
    candidate.bls_depth = bls_result.get("depth", 0.0)
    candidate.bls_duration = bls_result.get("duration_hr", 0.0)
    candidate.bls_power = bls_result.get("power", 0.0)
    candidate.bls_sde = bls_result.get("sde", 0.0)
    candidate.bls_snr = bls_result.get("snr", 0.0)
    candidate.bls_raw = {
        "periods": bls_result.get("periods", []),
        "power_spectrum": bls_result.get("power_spectrum", []),
    }

    # Period agreement: check if TLS and BLS agree
    if candidate.period > 0 and candidate.bls_period > 0:
        # Check for exact match or harmonic (P, 2P, P/2)
        ratio = candidate.bls_period / candidate.period
        # Closest harmonic
        harmonics = [0.5, 1.0, 2.0, 3.0, 1.0/3.0]
        diffs = [abs(ratio - h) / h for h in harmonics]
        candidate.period_agreement = min(diffs)
    else:
        candidate.period_agreement = 1.0  # no agreement possible

    return candidate


# ---------------------------------------------------------------------------
# Diagnostic sub-routines
# ---------------------------------------------------------------------------

def _odd_even_diff(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    t0: float,
    duration_days: float,
) -> float:
    """
    Compute normalised odd-even transit depth difference.

    Separates transit events into odd (1st, 3rd, 5th, ...) and even
    (2nd, 4th, 6th, ...) transits, measures the median in-transit
    depth of each group, and returns the normalised difference:

        odd_even_diff = |depth_odd - depth_even| / mean(depth_odd, depth_even)

    High values (> 0.1) suggest an eclipsing binary (since the primary
    and secondary eclipses would have different depths at P and P/2).

    Parameters
    ----------
    time : np.ndarray   time array (BTJD)
    flux : np.ndarray   normalised flux
    period : float      orbital period in days
    t0 : float          reference transit mid-time (BTJD)
    duration_days : float  transit duration in days

    Returns
    -------
    float : normalised odd-even depth difference (0 = identical)
    """
    if period <= 0 or duration_days <= 0:
        return 0.0

    try:
        # Find transit number for each point
        transit_number = np.round((time - t0) / period).astype(int)
        half_dur = duration_days / 2.0

        # Identify in-transit points
        phase_offset = np.abs((time - t0) - transit_number * period)
        in_transit = phase_offset < half_dur

        if in_transit.sum() < 4:
            return 0.0

        # Separate odd and even transit events
        odd_mask = in_transit & (transit_number % 2 != 0)
        even_mask = in_transit & (transit_number % 2 == 0)

        if odd_mask.sum() < 2 or even_mask.sum() < 2:
            return 0.0

        # Compute median depth relative to out-of-transit baseline
        out_transit = ~in_transit
        if out_transit.sum() < 10:
            baseline = 1.0
        else:
            baseline = float(np.nanmedian(flux[out_transit]))

        odd_depth = baseline - float(np.nanmedian(flux[odd_mask]))
        even_depth = baseline - float(np.nanmedian(flux[even_mask]))

        mean_depth = (odd_depth + even_depth) / 2.0
        if mean_depth <= 0:
            return 0.0

        return float(abs(odd_depth - even_depth) / mean_depth)

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
    run_bls_search: bool = True,
) -> List[TransitCandidate]:
    """
    Run TLS and (optionally) BLS on every preprocessed CSV.

    Both algorithms are run independently; the BLS results are merged
    into the TransitCandidate for downstream use as extra features and
    for cross-validation of the TLS period.

    Returns list of all TransitCandidate objects (including non-detections).
    """
    csv_files = sorted(processed_dir.glob("*.csv"))
    if not csv_files:
        logger.warning(f"No processed CSVs found in {processed_dir}.")
        return []

    algo_label = "TLS + BLS" if run_bls_search else "TLS"
    logger.info(f"Running {algo_label} on {len(csv_files)} light curves…")
    candidates = []

    for csv_path in csv_files:
        tic_id = csv_path.stem.split("_")[0].replace("TIC", "")
        try:
            df = pd.read_csv(csv_path)
            time = df["time"].values
            flux = df["flux"].values
            flux_err = df["flux_err"].values if "flux_err" in df.columns else None

            # --- TLS ---
            c = run_tls(
                time, flux, flux_err,
                tic_id=tic_id,
                period_min=period_min,
                period_max=period_max,
                sde_threshold=sde_threshold,
            )

            # --- BLS ---
            if run_bls_search:
                bls_result = run_bls(
                    time, flux, flux_err,
                    tic_id=tic_id,
                    period_min=period_min,
                    period_max=period_max,
                )
                c = merge_bls_into_candidate(c, bls_result)

                # If TLS missed it but BLS found a strong signal, flag it
                if not c.passed_threshold and c.bls_sde >= sde_threshold:
                    logger.info(
                        f"  TIC {tic_id}: TLS below threshold but BLS SDE={c.bls_sde:.1f} "
                        f"— flagging as candidate via BLS."
                    )
                    c.passed_threshold = True

            candidates.append(c)
        except Exception as exc:
            logger.error(f"Error processing {csv_path.name}: {exc}")

    n_passed = sum(1 for c in candidates if c.passed_threshold)
    logger.success(
        f"{algo_label} complete: {n_passed}/{len(candidates)} candidates above SDE > {sde_threshold}"
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
            # BLS fields
            "bls_period_d": c.bls_period,
            "bls_t0": c.bls_t0,
            "bls_depth_ppm": c.bls_depth * 1e6 if c.bls_depth < 1 else c.bls_depth,
            "bls_duration_hr": c.bls_duration,
            "bls_power": c.bls_power,
            "bls_sde": c.bls_sde,
            "bls_snr": c.bls_snr,
            "period_agreement": c.period_agreement,
        })
    return pd.DataFrame(rows)
