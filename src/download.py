"""
src/download.py
---------------
Downloads TESS PDCSAP light curves from MAST via lightkurve.
Supports bulk sector downloads with local caching.
"""

import os
import time
import warnings
from pathlib import Path
from typing import List, Optional

import lightkurve as lk
import numpy as np
import pandas as pd
from astroquery.mast import Catalogs, Observations
from loguru import logger
from tqdm import tqdm

warnings.filterwarnings("ignore")

RAW_DIR = Path("data/raw")
META_CSV = Path("data/target_list.csv")


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def ensure_dirs():
    """Create data directory tree if not present."""
    for d in ["data/raw", "data/processed", "data/labels", "results/figures"]:
        Path(d).mkdir(parents=True, exist_ok=True)


def query_sector_targets(sector: int, max_targets: int = 500) -> pd.DataFrame:
    """
    Query MAST for TIC IDs observed in a given TESS sector.

    Parameters
    ----------
    sector : int
        TESS sector number (1-based).
    max_targets : int
        Maximum number of targets to retrieve.

    Returns
    -------
    pd.DataFrame with columns [TIC, ra, dec, Tmag]
    """
    logger.info(f"Querying MAST for Sector {sector} SPOC targets (max={max_targets})…")

    try:
        obs = Observations.query_criteria(
            project="TESS",
            sequence_number=sector,
            obs_collection="TESS",
            provenance_name="SPOC"
        )
        results = obs.to_pandas()

        # Clean target names (keep only numeric TIC IDs)
        results = results[results["target_name"].astype(str).str.isnumeric()]

        if len(results) > max_targets:
            results = results.sample(max_targets, random_state=42)

        # Map columns to TIC, ra, dec, Tmag
        results = results[["target_name", "s_ra", "s_dec"]].rename(
            columns={"target_name": "TIC", "s_ra": "ra", "s_dec": "dec"}
        )
        results["Tmag"] = np.nan  # Place-holder since Tmag is not in Observations table

        logger.success(f"Found {len(results)} targets.")
        return results

    except Exception as exc:
        logger.warning(f"MAST observations query failed: {exc}. Falling back to lightkurve search.")
        return _fallback_sector_search(sector, max_targets)


def _fallback_sector_search(sector: int, max_targets: int) -> pd.DataFrame:
    """Use lightkurve sector search as fallback."""
    # Search for known bright targets observed in this sector
    search = lk.search_lightcurve("TrES-2", sector=sector, author="SPOC")
    if len(search) == 0:
        raise RuntimeError(f"No targets found for sector {sector}.")
    return pd.DataFrame({"TIC": ["TrES-2"], "ra": [None], "dec": [None], "Tmag": [None]})


def download_lightcurve(
    tic_id: str,
    sector: int,
    output_dir: Path = RAW_DIR,
    cadence: str = "short",
) -> Optional[Path]:
    """
    Download a single TESS light curve from MAST and save as FITS.

    Parameters
    ----------
    tic_id : str
        TIC identifier (e.g. "261136679").
    sector : int
        TESS sector.
    output_dir : Path
        Directory to save the FITS file.
    cadence : str
        "short" (2-min) or "long" (30-min).

    Returns
    -------
    Path to saved FITS file, or None on failure.
    """
    fits_path = output_dir / f"TIC{tic_id}_s{sector:04d}.fits"

    if fits_path.exists():
        logger.debug(f"  Cache hit: {fits_path.name}")
        return fits_path

    try:
        search = lk.search_lightcurve(
            f"TIC {tic_id}",
            sector=sector,
            author="SPOC",
            exptime=120 if cadence == "short" else 1800,
        )
        if len(search) == 0:
            logger.debug(f"  No SPOC LC for TIC {tic_id}, sector {sector}.")
            return None

        lc_collection = search.download()
        if lc_collection is None:
            return None

        lc = lc_collection[0] if hasattr(lc_collection, "__iter__") else lc_collection
        lc.to_fits(fits_path, overwrite=True)
        logger.debug(f"  Saved: {fits_path.name}")
        return fits_path

    except Exception as exc:
        logger.warning(f"  Download failed for TIC {tic_id}: {exc}")
        return None


def bulk_download(
    tic_ids: List[str],
    sector: int,
    output_dir: Path = RAW_DIR,
    delay: float = 0.5,
) -> List[Path]:
    """
    Download multiple TESS light curves with rate limiting.

    Parameters
    ----------
    tic_ids : list of str
    sector : int
    output_dir : Path
    delay : float
        Seconds to wait between requests (be polite to MAST).

    Returns
    -------
    List of successfully downloaded file paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded = []

    logger.info(f"Downloading {len(tic_ids)} light curves for Sector {sector}…")
    for tic in tqdm(tic_ids, desc="Downloading", unit="LC"):
        path = download_lightcurve(str(tic), sector, output_dir)
        if path:
            downloaded.append(path)
        time.sleep(delay)

    logger.success(f"Downloaded {len(downloaded)}/{len(tic_ids)} light curves.")
    return downloaded


def download_known_planets(
    sector: int = 1,
    output_dir: Path = RAW_DIR,
    max_targets: int = 100,
) -> List[Path]:
    """
    Download light curves for TESS confirmed planets (for validation).
    Uses the NASA Exoplanet Archive known TESS planet host list.
    """
    # Curated list of well-known TESS planet hosts
    known_hosts = [
        "261136679",  # WASP-126
        "55652896",   # LTT 9779
        "307210830",  # TOI-132
        "254113311",  # HD 213885
        "149603524",  # HD 2685
        "350618622",  # WASP-100
        "272272592",  # WASP-19
        "24358417",   # WASP-18
        "100100827",  # KELT-9
        "268644785",  # TOI-216
    ][:max_targets]

    logger.info(f"Downloading {len(known_hosts)} known planet hosts…")
    return bulk_download(known_hosts, sector, output_dir)


# ---------------------------------------------------------------------------
# Utility: load a FITS LC to a lightkurve object
# ---------------------------------------------------------------------------

def load_fits(fits_path: Path) -> Optional[lk.LightCurve]:
    """Load a saved FITS file back into a LightCurve object."""
    try:
        lc = lk.read(str(fits_path))
        return lc
    except Exception as exc:
        logger.warning(f"Could not load {fits_path}: {exc}")
        return None
