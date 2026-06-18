"""
tests/test_preprocess.py
Injects a known transit into a synthetic light curve and checks recovery.
"""
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.preprocess import preprocess_lightcurve, fold_lightcurve, compute_snr


def make_synthetic_lc(period=3.0, depth=0.01, n_pts=5000):
    """Create a simple synthetic TESS-like light curve with injected transits."""
    import lightkurve as lk
    from astropy.time import Time

    time = np.linspace(0, 27, n_pts)
    flux = np.ones(n_pts)

    # Inject transits
    duration = 0.1  # days
    t0 = 1.5
    transit_times = t0 + np.arange(0, 10) * period
    for tt in transit_times:
        mask = np.abs(time - tt) < duration / 2
        flux[mask] -= depth

    # Add noise
    rng = np.random.default_rng(42)
    flux += rng.normal(0, 0.002, n_pts)

    lc = lk.LightCurve(time=time, flux=flux, flux_err=np.ones(n_pts) * 0.002)
    return lc


def test_preprocess_returns_dataframe():
    lc = make_synthetic_lc()
    df = preprocess_lightcurve(lc, detrend=False)
    assert df is not None, "Preprocessing returned None"
    assert "time" in df.columns
    assert "flux" in df.columns
    assert len(df) > 100


def test_normalisation():
    lc = make_synthetic_lc()
    df = preprocess_lightcurve(lc, detrend=False)
    assert df is not None
    med = np.median(df["flux"])
    assert abs(med - 1.0) < 0.01, f"Median flux not near 1.0: {med}"


def test_fold_lightcurve():
    time = np.linspace(0, 27, 5000)
    flux = np.ones(5000)
    period = 3.0
    t0 = 1.5
    depth = 0.01
    duration = 0.1
    transit_times = t0 + np.arange(0, 10) * period
    for tt in transit_times:
        mask = np.abs(time - tt) < duration / 2
        flux[mask] -= depth

    phase, binned = fold_lightcurve(time, flux, period, t0)
    assert len(phase) == len(binned)
    # Transit should be near phase 0
    centre_mask = np.abs(phase) < 0.05
    assert np.min(binned[centre_mask]) < 1.0 - depth * 0.5, "Transit not found in folded LC"


def test_snr():
    flux = np.ones(1000)
    flux[400:420] = 0.99  # 1% transit
    in_mask = np.zeros(1000, dtype=bool)
    in_mask[400:420] = True
    snr = compute_snr(flux, in_mask)
    assert snr > 5, f"SNR too low: {snr}"


if __name__ == "__main__":
    test_preprocess_returns_dataframe()
    test_normalisation()
    test_fold_lightcurve()
    test_snr()
    print("All preprocess tests passed ✓")
