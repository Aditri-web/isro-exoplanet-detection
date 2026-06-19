"""
src/report.py
-------------
Auto-generates a 3-page PDF report summarising the pipeline results.

Pages:
  1. Methodology & Tools
  2. Detection & Classification Results (table + figures)
  3. Parameter Estimation & Uncertainties
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

REPORT_PATH = Path("results/ISRO_Exoplanet_Report.pdf")


def generate_report(
    detections_df: pd.DataFrame,
    metrics: Dict,
    figures_dir: Path = Path("results/figures"),
    output_path: Path = REPORT_PATH,
    sector: int = 1,
    n_total: int = 0,
    sde_threshold: float = 7.0,
) -> Path:
    """
    Generate a 3-page PDF report.

    Parameters
    ----------
    detections_df : pd.DataFrame  final classified detections table
    metrics : dict                classifier training metrics
    figures_dir : Path            directory containing saved figures
    output_path : Path            where to save the PDF
    sector : int                  TESS sector processed
    n_total : int                 total targets processed
    sde_threshold : float

    Returns
    -------
    Path to the generated PDF.
    """
    try:
        from fpdf import FPDF
    except ImportError:
        logger.error("fpdf2 not installed. Run: pip install fpdf2")
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf = _build_pdf(
        detections_df, metrics, figures_dir, sector, n_total, sde_threshold
    )
    pdf.output(str(output_path))
    logger.success(f"Report saved → {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Internal PDF builder
# ---------------------------------------------------------------------------

def _build_pdf(
    df: pd.DataFrame,
    metrics: Dict,
    figures_dir: Path,
    sector: int,
    n_total: int,
    sde_threshold: float,
) -> "FPDF":
    from fpdf import FPDF

    class PDF(FPDF):
        def header(self):
            self.set_fill_color(15, 23, 42)   # #0f172a
            self.rect(0, 0, 210, 18, "F")
            self.set_font("Helvetica", "B", 11)
            self.set_text_color(241, 245, 249)
            self.set_y(4)
            self.cell(0, 10, "ISRO Hackathon · AI-Enabled Exoplanet Detection Pipeline", ln=True, align="C")
            self.set_draw_color(56, 189, 248)
            self.set_line_width(0.5)
            self.line(10, 18, 200, 18)
            self.ln(4)

        def footer(self):
            self.set_y(-12)
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(148, 163, 184)
            ts = datetime.now().strftime("%Y-%m-%d %H:%M")
            self.cell(0, 8, f"Page {self.page_no()} / 3   ·   Generated {ts}", align="C")

    pdf = PDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_margins(15, 22, 15)

    # ===========================================================
    # PAGE 1: Methodology
    # ===========================================================
    pdf.add_page()
    _section_title(pdf, "1. Methodology & Tools")

    _body(pdf,
        "This pipeline automatically detects and classifies exoplanet transit signals in TESS "
        f"Sector {sector} light curves using a multi-stage approach combining classical "
        "signal processing, machine learning, and Bayesian parameter estimation."
    )

    _subsection(pdf, "1.1  Data Acquisition")
    _body(pdf,
        f"High-cadence (2-minute) PDCSAP light curves were downloaded from MAST via the "
        "lightkurve Python package. A total of "
        f"{n_total if n_total else 'N'} targets were retrieved from Sector {sector}."
    )

    _subsection(pdf, "1.2  Preprocessing")
    steps = [
        "Quality masking: only cadences with TESS quality flag = 0 are retained.",
        "NaN / negative-flux removal.",
        "Sigma-clipping: iterative 3σ upper / 10σ lower rejection (preserves transit dips).",
        "Wotan biweight detrending (window = 0.75 d) to remove stellar variability and "
        "systematics while preserving transit signals.",
        "Normalisation to unit median flux.",
    ]
    _bullet_list(pdf, steps)

    _subsection(pdf, "1.3  Period Search (TLS + BLS)")
    _body(pdf,
        "Two complementary period search algorithms were run on each detrended light curve "
        f"over the period range 0.5-27 days. Candidates with SDE > {sde_threshold} from "
        "either algorithm were retained for further analysis."
    )
    _body(pdf,
        "TransitLeastSquares (TLS; Hippke & Heller 2019) uses optimised transit-shaped "
        "templates matched to realistic limb-darkened stellar models. "
        "Box Least Squares (BLS; Kovacs, Zucker & Mazeh 2002) uses a simple box-shaped "
        "template via astropy.timeseries.BoxLeastSquares. Both algorithms are run independently; "
        "period agreement between TLS and BLS is computed as an additional classification feature."
    )
    _body(pdf,
        "Additional diagnostics computed per candidate:"
    )
    diag = [
        "Odd-even transit depth difference (flags eclipsing binaries: |depth_diff|/depth > 0.1).",
        "Secondary eclipse depth at orbital phase 0.5 (flags EBs and blends).",
        "Transit shape score: ingress fraction (U-shape -> planet; V-shape -> EB).",
        "TLS/BLS period agreement: fractional period discrepancy (0 = perfect match).",
    ]
    _bullet_list(pdf, diag)

    _subsection(pdf, "1.4  Classification")
    _body(pdf,
        "A Random Forest classifier (300 trees, balanced class weights) was trained on the "
        "curated labelled dataset (PLANET / EB / BLEND / OTHER) using 16 engineered features. "
        "5-fold stratified cross-validation was used for model selection."
    )
    if metrics:
        cv_f1 = metrics.get("cv_f1_mean", 0)
        cv_std = metrics.get("cv_f1_std", 0)
        _body(pdf, f"CV F1 (weighted): {cv_f1:.3f} ± {cv_std:.3f}")

    _subsection(pdf, "1.5  Parameter Estimation")
    _body(pdf,
        "For PLANET candidates, a batman transit model (Kreidberg 2015) was fitted to the "
        "phase-folded light curve using Nelder-Mead optimisation. Free parameters: "
        "Rp/Rs, a/Rs, inclination i, transit time T0. Quadratic limb-darkening "
        "(u1=0.40, u2=0.28, TESS band, solar-type star; Claret 2017) was fixed. "
        "Parameter uncertainties were estimated via finite-difference approximation of the "
        "Hessian (ΔΧSQ = 1 criterion); MCMC posteriors (emcee) are optionally available."
    )

    _subsection(pdf, "1.6  Libraries & Tools")
    libs = [
        "lightkurve 2.4+ -- TESS data access and basic LC operations",
        "transitleastsquares 1.0+ -- TLS period search",
        "astropy 5.3+ -- FITS I/O, time systems, BoxLeastSquares (BLS)",
        "wotan 1.10+ -- robust biweight detrending",
        "batman-package 2.4+ -- analytic transit model",
        "scikit-learn 1.3+ -- Random Forest classifier with calibrated probabilities",
        "emcee 3.1+ -- MCMC parameter estimation (optional)",
        "scipy 1.11+ -- optimisation",
        "matplotlib / plotly -- static and interactive visualisation",
        "fpdf2 -- report generation",
    ]
    _bullet_list(pdf, libs)

    # ===========================================================
    # PAGE 2: Results
    # ===========================================================
    pdf.add_page()
    _section_title(pdf, "2. Detection & Classification Results")

    if len(df) == 0:
        _body(pdf, "No detections to report.")
    else:
        passed = df[df.get("passed_threshold", False)] if "passed_threshold" in df.columns else df
        n_candidates = len(passed)
        class_counts = df["predicted_class"].value_counts() if "predicted_class" in df.columns else pd.Series()

        _body(pdf,
            f"Of {n_total if n_total else 'N'} targets processed, "
            f"{n_candidates} passed the SDE > {sde_threshold} threshold. "
            "Classification breakdown:"
        )
        for cls, cnt in class_counts.items():
            _body(pdf, f"   • {cls}: {cnt}")

        pdf.ln(3)

        # Detection table (top 20)
        planet_df = df[df.get("predicted_class", pd.Series()) == "PLANET"] if "predicted_class" in df.columns else df
        if len(planet_df) > 0:
            _subsection(pdf, "2.1  Planet Candidates (top 20)")
            table_cols = ["tic_id", "period_d", "depth_ppm", "duration_hr", "snr", "sde", "confidence"]
            avail_cols = [c for c in table_cols if c in planet_df.columns]
            _table(pdf, planet_df[avail_cols].head(20), avail_cols)

    # Insert classification summary figure if it exists
    summary_fig = figures_dir / "classification_summary.png"
    if summary_fig.exists():
        pdf.ln(4)
        pdf.image(str(summary_fig), x=30, w=150)

    # Insert feature importance figure
    fi_fig = figures_dir / "feature_importance.png"
    if fi_fig.exists():
        pdf.ln(4)
        pdf.image(str(fi_fig), x=25, w=160)

    # ===========================================================
    # PAGE 3: Parameters & Uncertainties
    # ===========================================================
    pdf.add_page()
    _section_title(pdf, "3. Parameter Estimation & Uncertainties")

    _subsection(pdf, "3.1  Fitted Parameters")
    fit_cols = ["tic_id", "fit_rp_rs", "fit_rp_rs_err",
                "fit_depth_ppm", "fit_depth_ppm_err",
                "fit_duration_hr", "fit_duration_hr_err",
                "fit_chi2r"]
    if "fit_rp_rs" in df.columns:
        planet_fits = df[df.get("fit_success", pd.Series(False, index=df.index))]
        if len(planet_fits) > 0:
            avail = [c for c in fit_cols if c in planet_fits.columns]
            _table(pdf, planet_fits[avail].head(15), avail)
        else:
            _body(pdf, "No successful fits available.")
    else:
        _body(pdf,
            "Fitted parameters will appear here after running the fitting module "
            "(src/fitting.py) on PLANET candidates."
        )

    _subsection(pdf, "3.2  Uncertainty Estimation")
    _body(pdf,
        "1σ parameter uncertainties are estimated via finite-difference second derivatives of "
        "the chi-squared surface (ΔΧSQ = 1). This provides symmetric Gaussian-equivalent "
        "uncertainties valid in the regime of high SNR. For low-SNR targets or those with "
        "complex posteriors, MCMC sampling (emcee) is recommended."
    )
    _body(pdf,
        "Signal-to-noise ratio (SNR) for each transit event is defined as: "
        "SNR = |μ_in − μ_out| / σ_out, where μ and σ denote the mean and standard deviation "
        "of in-transit and out-of-transit flux values respectively."
    )
    _body(pdf,
        "Confidence (classification) is the probability assigned to the winning class by the "
        "Random Forest (fraction of trees voting for that class)."
    )

    _subsection(pdf, "3.3  Assumptions")
    assumptions = [
        "Circular orbits (e = 0) assumed for parameter estimation.",
        "Quadratic limb-darkening fixed to solar values (TESS band); not free-fitted.",
        "Stellar parameters (R★, M★) not used; only geometry-independent transit parameters reported.",
        "Period assumed fixed at TLS best-fit value during fitting (no free-period optimisation).",
    ]
    _bullet_list(pdf, assumptions)

    # Sample phase-fold figure
    panel_figs = sorted(figures_dir.glob("*_panel.png"))
    if panel_figs:
        _subsection(pdf, "3.4  Example Candidate")
        pdf.ln(2)
        pdf.image(str(panel_figs[0]), x=10, w=185)

    return pdf


# ---------------------------------------------------------------------------
# PDF formatting helpers
# ---------------------------------------------------------------------------

def _section_title(pdf, text: str):
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(56, 189, 248)
    pdf.cell(0, 8, text, ln=True)
    pdf.set_draw_color(56, 189, 248)
    pdf.set_line_width(0.3)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.ln(3)
    pdf.set_text_color(30, 41, 59)


def _subsection(pdf, text: str):
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(14, 116, 144)
    pdf.cell(0, 7, text, ln=True)
    pdf.set_text_color(30, 41, 59)


def _body(pdf, text: str):
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(51, 65, 85)
    pdf.multi_cell(0, 5, text)
    pdf.ln(1)


def _bullet_list(pdf, items: List[str]):
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(51, 65, 85)
    for item in items:
        pdf.cell(5)
        pdf.multi_cell(0, 5, f"• {item}")
    pdf.ln(1)


def _table(pdf, df: pd.DataFrame, cols: List[str]):
    """Simple ASCII table in PDF."""
    pdf.set_font("Courier", "B", 7)
    pdf.set_text_color(15, 23, 42)
    pdf.set_fill_color(186, 230, 253)

    col_widths = {col: max(len(col), 8) * 3.5 for col in cols}
    # Shrink to fit page
    total_w = sum(col_widths.values())
    if total_w > 175:
        scale = 175 / total_w
        col_widths = {c: w * scale for c, w in col_widths.items()}

    # Header
    for col in cols:
        pdf.cell(col_widths[col], 6, col[:18], border=1, align="C", fill=True)
    pdf.ln()

    # Rows
    pdf.set_font("Courier", "", 7)
    pdf.set_fill_color(248, 250, 252)
    for i, (_, row) in enumerate(df.iterrows()):
        fill = i % 2 == 0
        for col in cols:
            val = row.get(col, "")
            if isinstance(val, float):
                cell_text = f"{val:.3g}"
            else:
                cell_text = str(val)[:18]
            pdf.cell(col_widths[col], 5, cell_text, border=1, align="C", fill=fill)
        pdf.ln()
    pdf.ln(2)
