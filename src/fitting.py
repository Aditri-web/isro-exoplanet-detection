"""
src/fitting.py
--------------
Transit parameter estimation using the batman light-curve model.

For each PLANET candidate:
  1. Phase-fold the light curve on the TLS best-fit period
  2. Fit a batman transit model via scipy.optimize (Nelder-Mead)
  3. Optionally run emcee MCMC for posterior uncertainties
  4. Return best-fit parameters + 1σ uncertainties
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from scipy.optimize import minimize
from scipy.stats import median_abs_deviation

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class FitResult:
    """Stores transit model fit results."""
    tic_id: str
    success: bool = False

    # Best-fit parameters
    period: float = 0.0       # days
    period_err: float = 0.0
    t0: float = 0.0           # BJD/BTJD mid-transit
    t0_err: float = 0.0
    rp_rs: float = 0.0        # radius ratio Rp/Rs
    rp_rs_err: float = 0.0
    a_rs: float = 10.0        # semi-major axis / stellar radius
    a_rs_err: float = 0.0
    inc: float = 89.0         # orbital inclination (degrees)
    inc_err: float = 0.0
    depth: float = 0.0        # transit depth = (Rp/Rs)^2
    depth_err: float = 0.0
    duration: float = 0.0     # transit duration (hours)
    duration_err: float = 0.0

    # Fit quality
    chi2: float = 0.0
    reduced_chi2: float = 0.0
    bic: float = 0.0

    # MCMC posteriors (filled if run_mcmc=True)
    mcmc_samples: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Limb-darkening coefficients (TESS band, solar-type star)
# Claret 2017 quadratic LDs for Teff~5500K, logg~4.4, [Fe/H]=0
# ---------------------------------------------------------------------------
LD_U1 = 0.40
LD_U2 = 0.28


# ---------------------------------------------------------------------------
# batman model evaluation
# ---------------------------------------------------------------------------

def batman_model(
    time_phase: np.ndarray,
    rp_rs: float,
    a_rs: float,
    inc: float,
    t0_phase: float = 0.0,
    u1: float = LD_U1,
    u2: float = LD_U2,
) -> np.ndarray:
    """
    Evaluate a batman transit model on a phase-folded time array.

    Parameters
    ----------
    time_phase : np.ndarray  (time in days, relative to transit centre)
    rp_rs : float            radius ratio
    a_rs : float             semi-major axis / stellar radius
    inc : float              inclination in degrees
    t0_phase : float         mid-transit offset in days (usually ~0)
    u1, u2 : float           quadratic limb-darkening coefficients

    Returns
    -------
    flux model array
    """
    try:
        import batman

        params = batman.TransitParams()
        params.t0 = t0_phase
        params.per = 1.0          # normalised to 1 (phase-space)
        params.rp = abs(rp_rs)
        params.a = abs(a_rs)
        params.inc = float(np.clip(inc, 60.0, 90.0))
        params.ecc = 0.0
        params.w = 90.0
        params.u = [u1, u2]
        params.limb_dark = "quadratic"

        m = batman.TransitModel(params, time_phase)
        return m.light_curve(params)

    except ImportError:
        # Fallback: simple trapezoidal model
        return _simple_transit_model(time_phase, rp_rs, a_rs, inc, t0_phase)


def _simple_transit_model(
    t: np.ndarray,
    rp_rs: float,
    a_rs: float,
    inc: float,
    t0: float,
) -> np.ndarray:
    """Simple analytical approximation when batman is unavailable."""
    depth = rp_rs ** 2
    # Transit duration in phase units
    inc_rad = np.radians(inc)
    b = a_rs * np.cos(inc_rad)
    if b >= 1 + rp_rs:
        return np.ones_like(t)  # no transit
    duration = (1.0 / (np.pi * a_rs)) * np.sqrt((1 + rp_rs) ** 2 - b ** 2)
    ingress = duration * 0.1

    flux = np.ones_like(t)
    dt = np.abs(t - t0)
    # Full transit
    full_mask = dt < (duration / 2.0 - ingress / 2.0)
    flux[full_mask] = 1.0 - depth
    # Ingress/egress
    ingress_mask = (dt >= (duration / 2.0 - ingress / 2.0)) & (dt < duration / 2.0)
    frac = (dt[ingress_mask] - (duration / 2.0 - ingress / 2.0)) / ingress
    flux[ingress_mask] = 1.0 - depth * (1 - frac)
    return flux


# ---------------------------------------------------------------------------
# Objective function
# ---------------------------------------------------------------------------

def _chi2(
    params_vec: np.ndarray,
    time_phase: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
) -> float:
    rp_rs, a_rs, inc, t0_offset = params_vec
    if rp_rs <= 0 or a_rs <= 1 or inc < 60 or inc > 90:
        return 1e10

    model = batman_model(time_phase, rp_rs, a_rs, inc, t0_offset)
    residuals = (flux - model) / flux_err
    return float(np.sum(residuals ** 2))


# ---------------------------------------------------------------------------
# Main fitting function
# ---------------------------------------------------------------------------

def fit_transit(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    period: float,
    t0_init: float,
    duration_init_hr: float,
    depth_init: float,
    tic_id: str = "unknown",
    run_mcmc: bool = False,
    n_mcmc_steps: int = 2000,
    n_walkers: int = 32,
) -> FitResult:
    """
    Fit a batman transit model to a light curve.

    Parameters
    ----------
    time : np.ndarray     (BTJD, full time series)
    flux : np.ndarray     normalised flux
    flux_err : np.ndarray
    period : float        days (from TLS)
    t0_init : float       initial guess for transit centre
    duration_init_hr : float  transit duration in hours (from TLS)
    depth_init : float    fractional depth (from TLS)
    tic_id : str
    run_mcmc : bool       whether to run emcee MCMC
    n_mcmc_steps : int
    n_walkers : int

    Returns
    -------
    FitResult
    """
    result = FitResult(tic_id=tic_id)
    result.period = period

    # --- Phase-fold ---
    phase = ((time - t0_init) % period)
    # Centre around 0 (−period/2 to +period/2)
    phase[phase > period / 2] -= period

    # Keep only ±3× transit duration around centre
    duration_days = duration_init_hr / 24.0
    window = max(duration_days * 3, 0.1)
    in_window = np.abs(phase) < window
    if in_window.sum() < 10:
        logger.warning(f"  TIC {tic_id}: too few points in transit window.")
        return result

    t_win = phase[in_window]
    f_win = flux[in_window]
    e_win = flux_err[in_window]

    # Ensure flux_err is sensible
    e_win = np.where(e_win <= 0, np.nanstd(f_win) * 0.1, e_win)

    # --- Initial parameter guesses ---
    rp_rs_init = np.sqrt(max(depth_init, 1e-6))
    a_rs_init = _estimate_a_rs(period, duration_days, rp_rs_init)
    inc_init = 89.0
    t0_offset_init = 0.0

    x0 = np.array([rp_rs_init, a_rs_init, inc_init, t0_offset_init])
    bounds = [(0.001, 0.5), (2.0, 200.0), (60.0, 90.0), (-duration_days, duration_days)]

    # --- Scipy Nelder-Mead optimisation ---
    try:
        opt = minimize(
            _chi2,
            x0,
            args=(t_win, f_win, e_win),
            method="Nelder-Mead",
            options={"maxiter": 5000, "xatol": 1e-6, "fatol": 1e-8},
        )

        rp_rs, a_rs, inc, t0_off = opt.x
        rp_rs = abs(rp_rs)
        a_rs = abs(a_rs)
        inc = np.clip(inc, 60.0, 90.0)

        n_data = in_window.sum()
        n_params = 4
        dof = max(n_data - n_params, 1)
        chi2_val = float(opt.fun)
        reduced_chi2 = chi2_val / dof
        bic = chi2_val + n_params * np.log(n_data)

        # Transit duration from fitted params
        inc_rad = np.radians(inc)
        b = a_rs * np.cos(inc_rad)
        if b < 1.0:
            dur_days = (period / np.pi) * np.arcsin(
                np.sqrt(((1 + rp_rs) ** 2 - b ** 2) / a_rs ** 2) / np.sin(inc_rad)
            )
        else:
            dur_days = duration_days  # fallback

        result.success = opt.success or reduced_chi2 < 10
        result.rp_rs = float(rp_rs)
        result.a_rs = float(a_rs)
        result.inc = float(inc)
        result.t0 = float(t0_init + t0_off)
        result.depth = float(rp_rs ** 2)
        result.duration = float(dur_days * 24.0)
        result.chi2 = chi2_val
        result.reduced_chi2 = reduced_chi2
        result.bic = bic

        # Parameter uncertainties via finite-difference Hessian approximation
        errs = _param_uncertainties(opt.x, t_win, f_win, e_win, chi2_val)
        result.rp_rs_err = float(errs[0])
        result.a_rs_err = float(errs[1])
        result.inc_err = float(errs[2])
        result.t0_err = float(errs[3])
        result.depth_err = float(2 * rp_rs * result.rp_rs_err)
        result.duration_err = float(dur_days * 24.0 * 0.1)  # 10% estimate

        logger.info(
            f"  TIC {tic_id}: Rp/Rs={rp_rs:.4f}±{result.rp_rs_err:.4f}  "
            f"depth={result.depth*1e6:.0f}ppm  dur={result.duration:.2f}h  "
            f"χ²ᵣ={reduced_chi2:.2f}"
        )

        # --- Optional MCMC ---
        if run_mcmc:
            result.mcmc_samples = _run_mcmc(
                opt.x, t_win, f_win, e_win,
                n_walkers=n_walkers, n_steps=n_mcmc_steps,
            )
            if result.mcmc_samples:
                _update_from_mcmc(result)

    except Exception as exc:
        logger.error(f"  Fit failed for TIC {tic_id}: {exc}")

    return result


def _estimate_a_rs(period: float, duration: float, rp_rs: float) -> float:
    """
    Estimate a/Rs from Kepler's third law approximation.
    period in days, duration in days.
    """
    if duration <= 0 or period <= 0:
        return 10.0
    # For edge-on orbit: duration ≈ period/π * arcsin(1/a_rs)
    # → a_rs ≈ period / (π * duration) for large a
    a_rs = period / (np.pi * duration) * np.sqrt((1 + rp_rs) ** 2)
    return float(np.clip(a_rs, 3.0, 200.0))


def _param_uncertainties(
    x0: np.ndarray,
    t: np.ndarray,
    f: np.ndarray,
    e: np.ndarray,
    chi2_min: float,
    delta_chi2: float = 1.0,
) -> np.ndarray:
    """
    Estimate 1σ parameter uncertainties via finite differences.
    Uses ΔCHI2 = 1.0 criterion.
    """
    errs = np.zeros(len(x0))
    for i in range(len(x0)):
        h = max(abs(x0[i]) * 0.01, 1e-5)
        x_plus = x0.copy(); x_plus[i] += h
        x_minus = x0.copy(); x_minus[i] -= h
        chi2_plus = _chi2(x_plus, t, f, e)
        chi2_minus = _chi2(x_minus, t, f, e)
        d2chi2 = (chi2_plus - 2 * chi2_min + chi2_minus) / (h ** 2)
        if d2chi2 > 0:
            errs[i] = np.sqrt(delta_chi2 / d2chi2)
        else:
            errs[i] = abs(h * 2)
    return errs


def _run_mcmc(
    x0: np.ndarray,
    t: np.ndarray,
    f: np.ndarray,
    e: np.ndarray,
    n_walkers: int = 32,
    n_steps: int = 2000,
) -> Dict:
    """Run emcee MCMC around the best-fit solution."""
    try:
        import emcee

        ndim = len(x0)
        pos = x0 + 1e-4 * np.random.randn(n_walkers, ndim)

        def log_prob(params):
            rp, a, inc, t0 = params
            if rp <= 0 or a <= 1 or inc < 60 or inc > 90:
                return -np.inf
            chi2 = _chi2(params, t, f, e)
            return -0.5 * chi2

        sampler = emcee.EnsembleSampler(n_walkers, ndim, log_prob)
        sampler.run_mcmc(pos, n_steps, progress=False)

        # Discard burn-in (50%) and thin
        samples = sampler.get_chain(discard=n_steps // 2, thin=15, flat=True)
        return {
            "rp_rs":  samples[:, 0].tolist(),
            "a_rs":   samples[:, 1].tolist(),
            "inc":    samples[:, 2].tolist(),
            "t0_off": samples[:, 3].tolist(),
        }
    except ImportError:
        logger.warning("  emcee not installed; skipping MCMC.")
        return {}
    except Exception as exc:
        logger.warning(f"  MCMC failed: {exc}")
        return {}


def _update_from_mcmc(result: FitResult):
    """Update parameter estimates from MCMC posteriors."""
    if not result.mcmc_samples:
        return
    samples = result.mcmc_samples

    for param, attr, err_attr in [
        ("rp_rs", "rp_rs", "rp_rs_err"),
        ("a_rs", "a_rs", "a_rs_err"),
        ("inc", "inc", "inc_err"),
    ]:
        if param in samples and len(samples[param]) > 0:
            arr = np.array(samples[param])
            setattr(result, attr, float(np.median(arr)))
            setattr(result, err_attr, float(median_abs_deviation(arr)))

    if "rp_rs" in samples:
        rp = np.array(samples["rp_rs"])
        result.depth = float(np.median(rp ** 2))
        result.depth_err = float(2 * np.median(rp) * median_abs_deviation(rp))


# ---------------------------------------------------------------------------
# Batch fitting
# ---------------------------------------------------------------------------

def fit_all_planets(
    detections_df: pd.DataFrame,
    processed_dir: str = "data/processed",
    run_mcmc: bool = False,
) -> pd.DataFrame:
    """
    Run transit fitting on all PLANET candidates in detections_df.
    Updates the DataFrame in-place with fit results.
    """
    processed_dir = Path(processed_dir)
    planet_mask = detections_df["predicted_class"] == "PLANET"

    fit_cols = [
        "fit_success", "fit_rp_rs", "fit_rp_rs_err",
        "fit_depth_ppm", "fit_depth_ppm_err",
        "fit_duration_hr", "fit_duration_hr_err",
        "fit_t0", "fit_t0_err",
        "fit_a_rs", "fit_a_rs_err",
        "fit_inc_deg", "fit_chi2r",
    ]
    for col in fit_cols:
        detections_df[col] = np.nan

    for idx, row in detections_df[planet_mask].iterrows():
        tic_id = str(row["tic_id"])
        csv_files = list(processed_dir.glob(f"TIC{tic_id}*.csv"))
        if not csv_files:
            logger.warning(f"  No processed LC for TIC {tic_id}")
            continue

        try:
            lc_df = pd.read_csv(csv_files[0])
            time = lc_df["time"].values
            flux = lc_df["flux"].values
            flux_err = (
                lc_df["flux_err"].values if "flux_err" in lc_df.columns
                else np.ones_like(flux) * np.nanstd(flux) * 0.1
            )

            fr = fit_transit(
                time, flux, flux_err,
                period=float(row["period_d"]),
                t0_init=float(row["t0_btjd"]),
                duration_init_hr=float(row.get("duration_hr", 3.0)),
                depth_init=float(row.get("depth_ppm", 1000.0)) / 1e6,
                tic_id=tic_id,
                run_mcmc=run_mcmc,
            )

            detections_df.loc[idx, "fit_success"] = fr.success
            detections_df.loc[idx, "fit_rp_rs"] = fr.rp_rs
            detections_df.loc[idx, "fit_rp_rs_err"] = fr.rp_rs_err
            detections_df.loc[idx, "fit_depth_ppm"] = fr.depth * 1e6
            detections_df.loc[idx, "fit_depth_ppm_err"] = fr.depth_err * 1e6
            detections_df.loc[idx, "fit_duration_hr"] = fr.duration
            detections_df.loc[idx, "fit_duration_hr_err"] = fr.duration_err
            detections_df.loc[idx, "fit_t0"] = fr.t0
            detections_df.loc[idx, "fit_t0_err"] = fr.t0_err
            detections_df.loc[idx, "fit_a_rs"] = fr.a_rs
            detections_df.loc[idx, "fit_a_rs_err"] = fr.a_rs_err
            detections_df.loc[idx, "fit_inc_deg"] = fr.inc
            detections_df.loc[idx, "fit_chi2r"] = fr.reduced_chi2

        except Exception as exc:
            logger.error(f"  Fit error for TIC {tic_id}: {exc}")

    return detections_df
