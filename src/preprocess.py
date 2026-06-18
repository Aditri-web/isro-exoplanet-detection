"""
src/preprocess.py
-----------------
Light curve preprocessing:
  1. Load PDCSAP flux from FITS / LightCurve object
  2. Quality mask & NaN removal
  3. Sigma-clip outlier rejection
  4. Wotan biweight detrending (removes stellar variability)
  5. Normalise to median = 1.0
  6. Save to CSV
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional, Tuple

import lightkurve as lk
import numpy as np
import pandas as pd
from astropy.stats import sigma_clip
from loguru import logger

warnings.filterwarnings("ignore")

PROCESSED_DIR = Path("data/processed")


# ---------------------------------------------------------------------------
# Core preprocessing
# ---------------------------------------------------------------------------

def preprocess_lightcurve(
    lc: lk.LightCurve,
    detrend: bool = True,
    window_length: float = 0.75,
    sigma_upper: float = 3.0,
    sigma_lower: float = 10.0,
) -> Optional[pd.DataFrame]:
    """
    Full preprocessing pipeline for a single TESS light curve.

    Parameters
    ----------
    lc : LightCurve
        Raw lightkurve object (PDCSAP preferred).
    detrend : bool
        Whether to apply Wotan biweight detrending.
    window_length : float
        Detrending window in days (default 0.75 d).
    sigma_upper : float
        Upper sigma threshold for outlier clipping.
    sigma_lower : float
        Lower sigma threshold (keep deep dips).

    Returns
    -------
    pd.DataFrame with columns [time, flux, flux_err] or None on failure.
    """
    try:
        # --- Step 1: Select PDCSAP flux column ---
        if hasattr(lc, "pdcsap_flux"):
            flux = lc.pdcsap_flux.value
            flux_err = lc.pdcsap_flux_err.value
        elif "pdcsap_flux" in lc.colnames:
            flux = lc["pdcsap_flux"].value
            flux_err = lc["pdcsap_flux_err"].value
        else:
            flux = lc.flux.value
            flux_err = lc.flux_err.value if hasattr(lc, "flux_err") else np.ones_like(flux) * 1e-3

        time = lc.time.value  # BJD or BTJD

        # --- Step 2: Quality mask & NaN removal ---
        if hasattr(lc, "quality"):
            good = lc.quality.value == 0
        else:
            good = np.ones(len(time), dtype=bool)

        finite = np.isfinite(flux) & np.isfinite(flux_err) & (flux_err > 0)
        mask = good & finite
        time, flux, flux_err = time[mask], flux[mask], flux_err[mask]

        if len(time) < 100:
            logger.warning("  Too few valid points after quality masking.")
            return None

        # --- Step 3: Sigma-clip outliers (upper only; preserve transit dips) ---
        clipped = sigma_clip(
            flux,
            sigma_upper=sigma_upper,
            sigma_lower=sigma_lower,
            maxiters=5,
            masked=True,
        )
        keep = ~clipped.mask
        time, flux, flux_err = time[keep], flux[keep], flux_err[keep]

        # --- Step 4: Wotan biweight detrending ---
        if detrend:
            try:
                from wotan import flatten
                flat_flux, trend = flatten(
                    time,
                    flux,
                    method="biweight",
                    window_length=window_length,
                    return_trend=True,
                    edge_cutoff=0.1,
                    break_tolerance=0.5,
                )
                flux = flat_flux
                # Propagate uncertainty: divide by trend
                flux_err = flux_err / trend
            except ImportError:
                logger.warning("  Wotan not found; skipping detrending.")
            except Exception as exc:
                logger.warning(f"  Detrending failed: {exc}; using raw flux.")

        # --- Step 5: Normalise to median = 1.0 ---
        med = np.nanmedian(flux)
        if med <= 0:
            logger.warning("  Non-positive median flux; skipping.")
            return None
        flux = flux / med
        flux_err = flux_err / med

        # Final finite check
        finite2 = np.isfinite(flux) & np.isfinite(flux_err)
        time, flux, flux_err = time[finite2], flux[finite2], flux_err[finite2]

        df = pd.DataFrame({"time": time, "flux": flux, "flux_err": flux_err})
        return df

    except Exception as exc:
        logger.error(f"  preprocess_lightcurve failed: {exc}")
        return None


def preprocess_fits(
    fits_path: Path,
    output_dir: Path = PROCESSED_DIR,
    **kwargs,
) -> Optional[Path]:
    """
    Load a FITS file, preprocess, and save as CSV.

    Returns path to saved CSV, or None on failure.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = fits_path.stem
    csv_path = output_dir / f"{stem}.csv"

    if csv_path.exists():
        return csv_path

    try:
        lc = lk.read(str(fits_path))
    except Exception as exc:
        logger.warning(f"  Could not read {fits_path.name}: {exc}")
        return None

    df = preprocess_lightcurve(lc, **kwargs)
    if df is None or len(df) < 50:
        return None

    df.to_csv(csv_path, index=False)
    logger.debug(f"  Preprocessed → {csv_path.name}  ({len(df)} pts)")
    return csv_path


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def fold_lightcurve(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    t0: float,
    n_bins: int = 200,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Phase-fold and bin a light curve.

    Returns
    -------
    phase : np.ndarray  (−0.5 to 0.5)
    binned_flux : np.ndarray
    """
    phase = ((time - t0) % period) / period
    phase[phase > 0.5] -= 1.0

    sort_idx = np.argsort(phase)
    phase = phase[sort_idx]
    flux_sorted = flux[sort_idx]

    bins = np.linspace(-0.5, 0.5, n_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    binned = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (phase >= bins[i]) & (phase < bins[i + 1])
        binned[i] = np.nanmedian(flux_sorted[mask]) if mask.sum() > 0 else 1.0

    return bin_centers, binned


def compute_snr(
    flux: np.ndarray,
    transit_mask: np.ndarray,
) -> float:
    """
    Compute transit SNR = |mean_in - mean_out| / std_out.
    """
    in_transit = flux[transit_mask]
    out_transit = flux[~transit_mask]
    if len(in_transit) == 0 or len(out_transit) == 0:
        return 0.0
    return abs(np.nanmean(in_transit) - np.nanmean(out_transit)) / (np.nanstd(out_transit) + 1e-10)


def load_preprocessed(csv_path: Path) -> Optional[pd.DataFrame]:
    """Load a preprocessed CSV."""
    try:
        return pd.read_csv(csv_path)
    except Exception:
        return None
