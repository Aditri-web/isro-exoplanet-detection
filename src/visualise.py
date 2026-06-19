"""
src/visualise.py
----------------
Publication-quality plots for the exoplanet detection pipeline.

Generates:
  1. Raw + detrended light curve overview
  2. TLS periodogram (SDE vs period)
  3. Phase-folded light curve with batman model
  4. Classification summary bar chart
  5. Feature importance plot
  6. Interactive Plotly HTML dashboard
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import AutoMinorLocator
from loguru import logger

warnings.filterwarnings("ignore")

FIGURES_DIR = Path("results/figures")

# Colour palette
COLOURS = {
    "PLANET": "#00d4ff",
    "EB":     "#ff6b35",
    "BLEND":  "#a855f7",
    "OTHER":  "#6b7280",
    "raw":    "#94a3b8",
    "detrend": "#38bdf8",
    "model":  "#f59e0b",
    "transit": "#ef4444",
    "bls":    "#22d3ee",
}

plt.style.use("dark_background")
FONT = {"family": "DejaVu Sans", "size": 11}
plt.rc("font", **FONT)


# ---------------------------------------------------------------------------
# 1. Light curve overview
# ---------------------------------------------------------------------------

def plot_lightcurve(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: Optional[np.ndarray] = None,
    transit_times: Optional[np.ndarray] = None,
    period: Optional[float] = None,
    tic_id: str = "unknown",
    save_dir: Path = FIGURES_DIR,
    title_extra: str = "",
) -> Path:
    """Plot a normalised light curve with transit markers."""
    save_dir.mkdir(parents=True, exist_ok=True)
    out = save_dir / f"TIC{tic_id}_lightcurve.png"

    fig, ax = plt.subplots(figsize=(14, 4))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")

    if flux_err is not None:
        ax.errorbar(time, flux, yerr=flux_err, fmt="none",
                    ecolor="#334155", alpha=0.4, linewidth=0.5)

    ax.scatter(time, flux, s=1.5, c=COLOURS["detrend"], alpha=0.7, linewidths=0)

    # Mark transit times
    if transit_times is not None:
        for tt in transit_times:
            ax.axvline(tt, color=COLOURS["transit"], alpha=0.5, linewidth=0.8, linestyle="--")

    ax.set_xlabel("Time (BTJD)", color="#cbd5e1", labelpad=6)
    ax.set_ylabel("Normalised Flux", color="#cbd5e1", labelpad=6)
    title = f"TIC {tic_id}  —  Detrended Light Curve"
    if title_extra:
        title += f"  |  {title_extra}"
    ax.set_title(title, color="#f1f5f9", pad=10, fontsize=12)

    ax.tick_params(colors="#94a3b8")
    for spine in ax.spines.values():
        spine.set_edgecolor("#334155")

    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.debug(f"  Saved: {out.name}")
    return out


# ---------------------------------------------------------------------------
# 2. TLS periodogram
# ---------------------------------------------------------------------------

def plot_periodogram(
    periods: np.ndarray,
    power: np.ndarray,
    best_period: float,
    sde_threshold: float = 7.0,
    tic_id: str = "unknown",
    save_dir: Path = FIGURES_DIR,
) -> Path:
    """Plot TLS SDE periodogram."""
    save_dir.mkdir(parents=True, exist_ok=True)
    out = save_dir / f"TIC{tic_id}_periodogram.png"

    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")

    ax.plot(periods, power, color=COLOURS["detrend"], linewidth=0.8, alpha=0.9)
    ax.axhline(sde_threshold, color="#f59e0b", linewidth=1.2, linestyle="--",
               label=f"SDE threshold = {sde_threshold}")
    ax.axvline(best_period, color=COLOURS["transit"], linewidth=1.5, linestyle="-",
               label=f"Best period = {best_period:.4f} d")

    ax.set_xlabel("Period (days)", color="#cbd5e1", labelpad=6)
    ax.set_ylabel("SDE (Signal Detection Efficiency)", color="#cbd5e1", labelpad=6)
    ax.set_title(f"TIC {tic_id}  —  TLS Periodogram", color="#f1f5f9", pad=10)
    ax.legend(framealpha=0.2, labelcolor="#f1f5f9", facecolor="#1e293b")
    ax.tick_params(colors="#94a3b8")
    for spine in ax.spines.values():
        spine.set_edgecolor("#334155")
    ax.set_xscale("log")

    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.debug(f"  Saved: {out.name}")
    return out


# ---------------------------------------------------------------------------
# 2b. BLS periodogram
# ---------------------------------------------------------------------------

def plot_bls_periodogram(
    periods: np.ndarray,
    power: np.ndarray,
    best_period: float,
    tic_id: str = "unknown",
    save_dir: Path = FIGURES_DIR,
) -> Path:
    """Plot BLS power periodogram."""
    save_dir.mkdir(parents=True, exist_ok=True)
    out = save_dir / f"TIC{tic_id}_bls_periodogram.png"

    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")

    ax.plot(periods, power, color=COLOURS["bls"], linewidth=0.8, alpha=0.9)
    ax.axvline(best_period, color=COLOURS["transit"], linewidth=1.5, linestyle="-",
               label=f"Best period = {best_period:.4f} d")

    ax.set_xlabel("Period (days)", color="#cbd5e1", labelpad=6)
    ax.set_ylabel("BLS Power (SR)", color="#cbd5e1", labelpad=6)
    ax.set_title(f"TIC {tic_id}  \u2014  BLS Periodogram", color="#f1f5f9", pad=10)
    ax.legend(framealpha=0.2, labelcolor="#f1f5f9", facecolor="#1e293b")
    ax.tick_params(colors="#94a3b8")
    for spine in ax.spines.values():
        spine.set_edgecolor("#334155")
    ax.set_xscale("log")

    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.debug(f"  Saved: {out.name}")
    return out


# ---------------------------------------------------------------------------
# 3. Phase-folded light curve + model
# ---------------------------------------------------------------------------

def plot_phase_fold(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    t0: float,
    duration_hr: float,
    model_phase: Optional[np.ndarray] = None,
    model_flux: Optional[np.ndarray] = None,
    depth_ppm: float = 0.0,
    tic_id: str = "unknown",
    label: str = "PLANET",
    confidence: float = 0.0,
    save_dir: Path = FIGURES_DIR,
) -> Path:
    """Phase-fold and plot with optional batman model overlay."""
    save_dir.mkdir(parents=True, exist_ok=True)
    out = save_dir / f"TIC{tic_id}_phasefold.png"

    # Phase-fold
    phase = ((time - t0) % period) / period
    phase[phase > 0.5] -= 1.0
    sort_idx = np.argsort(phase)
    phase_s = phase[sort_idx]
    flux_s = flux[sort_idx]

    # Bin for display
    n_bins = 150
    bins = np.linspace(-0.5, 0.5, n_bins + 1)
    bin_c = 0.5 * (bins[:-1] + bins[1:])
    bin_f = np.array([
        np.nanmedian(flux_s[(phase_s >= bins[i]) & (phase_s < bins[i + 1])])
        if ((phase_s >= bins[i]) & (phase_s < bins[i + 1])).sum() > 0 else np.nan
        for i in range(n_bins)
    ])
    bin_e = np.array([
        np.nanstd(flux_s[(phase_s >= bins[i]) & (phase_s < bins[i + 1])]) /
        np.sqrt(max(((phase_s >= bins[i]) & (phase_s < bins[i + 1])).sum(), 1))
        for i in range(n_bins)
    ])

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")

    # Raw data
    ax.scatter(phase_s, flux_s, s=0.8, c=COLOURS["raw"], alpha=0.25, linewidths=0)

    # Binned data
    valid = np.isfinite(bin_f)
    ax.errorbar(bin_c[valid], bin_f[valid], yerr=bin_e[valid],
                fmt="o", markersize=3, color=COLOURS["detrend"],
                ecolor=COLOURS["detrend"], alpha=0.85, linewidth=0.8,
                capsize=2, label="Binned")

    # Model overlay
    if model_phase is not None and model_flux is not None:
        ax.plot(model_phase, model_flux, color=COLOURS["model"],
                linewidth=2.0, zorder=5, label="batman model")

    # Transit duration shading
    half_dur = (duration_hr / 24.0) / period / 2.0
    ax.axvspan(-half_dur, half_dur, color=COLOURS["transit"], alpha=0.08)

    cls_colour = COLOURS.get(label, COLOURS["OTHER"])
    info = (
        f"TIC {tic_id}   |   Class: {label}   |   "
        f"P = {period:.4f} d   |   depth = {depth_ppm:.0f} ppm   |   "
        f"dur = {duration_hr:.2f} h   |   confidence = {confidence:.1%}"
    )
    ax.set_title(info, color=cls_colour, pad=8, fontsize=10, fontweight="bold")
    ax.set_xlabel("Orbital Phase", color="#cbd5e1", labelpad=6)
    ax.set_ylabel("Normalised Flux", color="#cbd5e1", labelpad=6)
    ax.legend(framealpha=0.2, labelcolor="#f1f5f9", facecolor="#1e293b", fontsize=9)
    ax.tick_params(colors="#94a3b8")
    for spine in ax.spines.values():
        spine.set_edgecolor("#334155")
    ax.set_xlim(-0.5, 0.5)

    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.debug(f"  Saved: {out.name}")
    return out


# ---------------------------------------------------------------------------
# 4. Classification summary
# ---------------------------------------------------------------------------

def plot_classification_summary(
    detections_df: pd.DataFrame,
    save_dir: Path = FIGURES_DIR,
) -> Path:
    """Bar chart of classification counts."""
    save_dir.mkdir(parents=True, exist_ok=True)
    out = save_dir / "classification_summary.png"

    counts = detections_df["predicted_class"].value_counts()
    labels = counts.index.tolist()
    values = counts.values

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")

    colours = [COLOURS.get(l, "#6b7280") for l in labels]
    bars = ax.bar(labels, values, color=colours, edgecolor="#0f172a", linewidth=1.5, width=0.6)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                str(val), ha="center", va="bottom", color="#f1f5f9", fontsize=12)

    ax.set_ylabel("Number of Targets", color="#cbd5e1", labelpad=6)
    ax.set_title("Classification Summary", color="#f1f5f9", pad=10, fontsize=14)
    ax.tick_params(colors="#94a3b8")
    for spine in ax.spines.values():
        spine.set_edgecolor("#334155")

    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.debug(f"  Saved: {out.name}")
    return out


# ---------------------------------------------------------------------------
# 5. Feature importance
# ---------------------------------------------------------------------------

def plot_feature_importance(
    importance_df: pd.DataFrame,
    save_dir: Path = FIGURES_DIR,
) -> Path:
    """Horizontal bar chart of RF feature importances."""
    save_dir.mkdir(parents=True, exist_ok=True)
    out = save_dir / "feature_importance.png"

    df = importance_df.head(16).sort_values("importance")

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")

    bars = ax.barh(df["feature"], df["importance"],
                   color="#38bdf8", edgecolor="#0f172a", linewidth=0.5)
    ax.set_xlabel("Feature Importance (Gini)", color="#cbd5e1", labelpad=6)
    ax.set_title("Random Forest Feature Importance", color="#f1f5f9", pad=10)
    ax.tick_params(colors="#94a3b8")
    ax.yaxis.set_tick_params(labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#334155")

    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.debug(f"  Saved: {out.name}")
    return out


# ---------------------------------------------------------------------------
# 6. Comprehensive candidate panel (LC + periodogram + phasefold)
# ---------------------------------------------------------------------------

def plot_candidate_panel(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    t0: float,
    duration_hr: float,
    depth_ppm: float,
    tls_periods: Optional[np.ndarray] = None,
    tls_power: Optional[np.ndarray] = None,
    bls_periods: Optional[np.ndarray] = None,
    bls_power: Optional[np.ndarray] = None,
    bls_period: Optional[float] = None,
    bls_sde: float = 0.0,
    model_phase: Optional[np.ndarray] = None,
    model_flux: Optional[np.ndarray] = None,
    tic_id: str = "unknown",
    label: str = "PLANET",
    confidence: float = 0.0,
    snr: float = 0.0,
    sde: float = 0.0,
    save_dir: Path = FIGURES_DIR,
) -> Path:
    """4-panel figure: raw LC | TLS periodogram | BLS periodogram | phase-fold."""
    save_dir.mkdir(parents=True, exist_ok=True)
    out = save_dir / f"TIC{tic_id}_panel.png"

    has_bls = (bls_periods is not None and bls_power is not None
               and len(bls_periods) > 0)
    n_cols = 4 if has_bls else 3

    fig = plt.figure(figsize=(6 * n_cols, 5))
    fig.patch.set_facecolor("#0f172a")
    gs = gridspec.GridSpec(1, n_cols, figure=fig, wspace=0.32)

    cls_colour = COLOURS.get(label, COLOURS["OTHER"])
    col_idx = 0

    # --- Panel 1: Light curve ---
    ax1 = fig.add_subplot(gs[col_idx]); col_idx += 1
    ax1.set_facecolor("#1e293b")
    ax1.scatter(time, flux, s=1, c=COLOURS["detrend"], alpha=0.6, linewidths=0)

    # Mark transit windows
    transit_times = t0 + np.arange(-50, 50) * period
    transit_times = transit_times[(transit_times >= time.min()) & (transit_times <= time.max())]
    for tt in transit_times:
        ax1.axvline(tt, color=cls_colour, alpha=0.4, linewidth=0.7)

    ax1.set_xlabel("Time (BTJD)", color="#cbd5e1", fontsize=9)
    ax1.set_ylabel("Norm. Flux", color="#cbd5e1", fontsize=9)
    ax1.set_title("Light Curve", color="#f1f5f9", fontsize=10)
    ax1.tick_params(colors="#94a3b8", labelsize=8)

    # --- Panel 2: TLS Periodogram ---
    ax2 = fig.add_subplot(gs[col_idx]); col_idx += 1
    ax2.set_facecolor("#1e293b")
    if tls_periods is not None and tls_power is not None and len(tls_periods) > 0:
        ax2.plot(tls_periods, tls_power, color=COLOURS["detrend"], linewidth=0.7)
        ax2.axhline(7.0, color="#f59e0b", linewidth=1, linestyle="--", alpha=0.8)
        ax2.axvline(period, color=cls_colour, linewidth=1.5, linestyle="-")
        ax2.set_xscale("log")
    ax2.set_xlabel("Period (days)", color="#cbd5e1", fontsize=9)
    ax2.set_ylabel("SDE", color="#cbd5e1", fontsize=9)
    ax2.set_title(f"TLS Periodogram  |  SDE={sde:.1f}", color="#f1f5f9", fontsize=10)
    ax2.tick_params(colors="#94a3b8", labelsize=8)

    # --- Panel 3: BLS Periodogram (if available) ---
    if has_bls:
        ax_bls = fig.add_subplot(gs[col_idx]); col_idx += 1
        ax_bls.set_facecolor("#1e293b")
        ax_bls.plot(bls_periods, bls_power, color=COLOURS["bls"], linewidth=0.7)
        if bls_period is not None and bls_period > 0:
            ax_bls.axvline(bls_period, color=COLOURS["transit"], linewidth=1.5,
                           linestyle="-", label=f"P={bls_period:.3f}d")
        ax_bls.axvline(period, color=cls_colour, linewidth=1, linestyle=":",
                       alpha=0.6, label=f"TLS P={period:.3f}d")
        ax_bls.set_xscale("log")
        ax_bls.set_xlabel("Period (days)", color="#cbd5e1", fontsize=9)
        ax_bls.set_ylabel("BLS Power", color="#cbd5e1", fontsize=9)
        ax_bls.set_title(f"BLS Periodogram  |  SDE={bls_sde:.1f}",
                         color="#f1f5f9", fontsize=10)
        ax_bls.tick_params(colors="#94a3b8", labelsize=8)
        ax_bls.legend(framealpha=0.2, labelcolor="#f1f5f9",
                      facecolor="#1e293b", fontsize=7, loc="upper right")

    # --- Panel N: Phase fold ---
    ax3 = fig.add_subplot(gs[col_idx])
    ax3.set_facecolor("#1e293b")

    phase = ((time - t0) % period) / period
    phase[phase > 0.5] -= 1.0
    sort_idx = np.argsort(phase)

    ax3.scatter(phase[sort_idx], flux[sort_idx], s=0.8, c=COLOURS["raw"], alpha=0.2, linewidths=0)

    # Binned
    bins = np.linspace(-0.5, 0.5, 101)
    bin_c = 0.5 * (bins[:-1] + bins[1:])
    ph_s = phase[sort_idx]
    fl_s = flux[sort_idx]
    bin_f = [
        np.nanmedian(fl_s[(ph_s >= bins[i]) & (ph_s < bins[i + 1])])
        if ((ph_s >= bins[i]) & (ph_s < bins[i + 1])).sum() > 0 else np.nan
        for i in range(100)
    ]
    valid = np.isfinite(bin_f)
    ax3.plot(bin_c[valid], np.array(bin_f)[valid], color=COLOURS["detrend"],
             linewidth=1.5, zorder=4)

    if model_phase is not None and model_flux is not None:
        ax3.plot(model_phase, model_flux, color=COLOURS["model"], linewidth=2, zorder=5)

    half_dur = (duration_hr / 24.0) / period / 2.0
    ax3.axvspan(-half_dur, half_dur, color=cls_colour, alpha=0.08)
    ax3.set_xlim(-0.25, 0.25)
    ax3.set_xlabel("Orbital Phase", color="#cbd5e1", fontsize=9)
    ax3.set_ylabel("Norm. Flux", color="#cbd5e1", fontsize=9)
    ax3.set_title(f"Phase Fold  |  SNR={snr:.1f}", color="#f1f5f9", fontsize=10)
    ax3.tick_params(colors="#94a3b8", labelsize=8)

    all_axes = [ax1, ax2, ax3]
    if has_bls:
        all_axes.insert(2, ax_bls)
    for ax in all_axes:
        for spine in ax.spines.values():
            spine.set_edgecolor("#334155")

    # Super-title
    fig.suptitle(
        f"TIC {tic_id}   ·   {label}   ·   P={period:.4f} d   ·   "
        f"depth={depth_ppm:.0f} ppm   ·   dur={duration_hr:.2f} h   ·   "
        f"confidence={confidence:.1%}",
        color=cls_colour, fontsize=11, fontweight="bold", y=1.01,
    )

    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.debug(f"  Panel saved: {out.name}")
    return out


# ---------------------------------------------------------------------------
# 7. Interactive Plotly HTML dashboard
# ---------------------------------------------------------------------------

def build_interactive_dashboard(
    detections_df: pd.DataFrame,
    save_dir: Path = Path("results"),
) -> Path:
    """Build an interactive HTML dashboard using Plotly."""
    try:
        import plotly.graph_objects as go
        import plotly.express as px
        from plotly.subplots import make_subplots
    except ImportError:
        logger.warning("plotly not installed; skipping dashboard.")
        return Path()

    save_dir.mkdir(parents=True, exist_ok=True)
    out = save_dir / "dashboard.html"

    df = detections_df.copy()
    df["predicted_class"] = df.get("predicted_class", pd.Series(["OTHER"] * len(df)))

    colour_map = {
        "PLANET": "#00d4ff",
        "EB":     "#ff6b35",
        "BLEND":  "#a855f7",
        "OTHER":  "#6b7280",
    }

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Period vs Depth (all candidates)",
            "SNR Distribution by Class",
            "Classification Breakdown",
            "Confidence vs SDE",
        ],
        vertical_spacing=0.14,
        horizontal_spacing=0.1,
    )

    # --- Plot 1: Period vs Depth ---
    for cls in ["PLANET", "EB", "BLEND", "OTHER"]:
        subset = df[df["predicted_class"] == cls]
        if len(subset) == 0:
            continue
        fig.add_trace(
            go.Scatter(
                x=subset.get("period_d", pd.Series()),
                y=subset.get("depth_ppm", pd.Series()),
                mode="markers",
                name=cls,
                marker=dict(
                    color=colour_map[cls],
                    size=8,
                    opacity=0.8,
                    line=dict(width=0.5, color="#0f172a"),
                ),
                text=subset.get("tic_id", pd.Series()).astype(str),
                hovertemplate=(
                    "<b>TIC %{text}</b><br>"
                    "Period: %{x:.3f} d<br>"
                    "Depth: %{y:.0f} ppm<br>"
                    "<extra></extra>"
                ),
            ),
            row=1, col=1,
        )

    # --- Plot 2: SNR histogram ---
    for cls in ["PLANET", "EB", "BLEND", "OTHER"]:
        subset = df[df["predicted_class"] == cls]
        if len(subset) == 0 or "snr" not in subset.columns:
            continue
        fig.add_trace(
            go.Histogram(
                x=subset["snr"],
                name=cls,
                marker_color=colour_map[cls],
                opacity=0.7,
                showlegend=False,
                nbinsx=20,
            ),
            row=1, col=2,
        )

    # --- Plot 3: Pie chart ---
    cls_counts = df["predicted_class"].value_counts()
    fig.add_trace(
        go.Pie(
            labels=cls_counts.index.tolist(),
            values=cls_counts.values.tolist(),
            marker_colors=[colour_map.get(l, "#6b7280") for l in cls_counts.index],
            hole=0.4,
            showlegend=False,
        ),
        row=2, col=1,
    )

    # --- Plot 4: Confidence vs SDE ---
    if "confidence" in df.columns and "sde" in df.columns:
        for cls in ["PLANET", "EB", "BLEND", "OTHER"]:
            subset = df[df["predicted_class"] == cls]
            if len(subset) == 0:
                continue
            fig.add_trace(
                go.Scatter(
                    x=subset["sde"],
                    y=subset["confidence"],
                    mode="markers",
                    name=cls,
                    marker=dict(color=colour_map[cls], size=8, opacity=0.8),
                    showlegend=False,
                ),
                row=2, col=2,
            )

    # Layout
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#1e293b",
        font=dict(family="Inter, sans-serif", color="#cbd5e1"),
        title=dict(
            text="🔭 ISRO Exoplanet Detection — Results Dashboard",
            font=dict(size=18, color="#f1f5f9"),
            x=0.5,
        ),
        height=750,
    )
    fig.update_xaxes(gridcolor="#1e293b", zerolinecolor="#334155")
    fig.update_yaxes(gridcolor="#1e293b", zerolinecolor="#334155")

    # Axis labels
    fig.update_xaxes(title_text="Period (days)", row=1, col=1, type="log")
    fig.update_yaxes(title_text="Depth (ppm)", row=1, col=1, type="log")
    fig.update_xaxes(title_text="SNR", row=1, col=2)
    fig.update_xaxes(title_text="SDE", row=2, col=2)
    fig.update_yaxes(title_text="Confidence", row=2, col=2)

    fig.write_html(str(out))
    logger.success(f"  Interactive dashboard saved → {out}")
    return out
