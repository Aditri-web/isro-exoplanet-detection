"""
pipeline.py
-----------
Main entry-point for the ISRO Exoplanet Detection Pipeline.

Usage examples:
  # Full run on 200 targets from Sector 1:
  python pipeline.py --sector 1 --n-targets 200 --sde-threshold 7

  # Train classifier only (requires data/labels/):
  python pipeline.py --mode train --label-csv data/labels/curated.csv

  # Classify without retraining (use saved model):
  python pipeline.py --mode detect --sector 1 --n-targets 500 --no-retrain

  # Fit transit parameters only for already-classified detections:
  python pipeline.py --mode fit --detections results/detections.csv
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import click
import numpy as np
import pandas as pd
from loguru import logger

# ──────────────────────────────────────────────────────────────────────────────
# Lazy imports (heavy libraries only loaded when needed)
# ──────────────────────────────────────────────────────────────────────────────

def _import_src():
    from src.download import ensure_dirs, download_known_planets, bulk_download, query_sector_targets
    from src.preprocess import preprocess_fits
    from src.period_search import search_all, candidates_to_dataframe
    from src.features import build_feature_dataframe, load_labelled_dataset, FEATURE_COLS
    from src.classifier import train, load_model, classify_candidates, feature_importance, evaluate_on_test
    from src.fitting import fit_all_planets
    from src.visualise import (
        plot_lightcurve, plot_periodogram, plot_phase_fold,
        plot_candidate_panel, plot_classification_summary,
        plot_feature_importance, build_interactive_dashboard,
    )
    from src.report import generate_report
    return (
        ensure_dirs, download_known_planets, bulk_download, query_sector_targets,
        preprocess_fits, search_all, candidates_to_dataframe,
        build_feature_dataframe, load_labelled_dataset, FEATURE_COLS,
        train, load_model, classify_candidates, feature_importance, evaluate_on_test,
        fit_all_planets,
        plot_lightcurve, plot_periodogram, plot_phase_fold,
        plot_candidate_panel, plot_classification_summary,
        plot_feature_importance, build_interactive_dashboard,
        generate_report,
    )


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

@click.command()
@click.option("--sector",         default=1,    help="TESS sector to process.", show_default=True)
@click.option("--n-targets",      default=100,  help="Max number of targets to download.", show_default=True)
@click.option("--sde-threshold",  default=7.0,  help="Minimum SDE to flag as candidate.", show_default=True)
@click.option("--period-min",     default=0.5,  help="Minimum period to search (days).", show_default=True)
@click.option("--period-max",     default=27.0, help="Maximum period to search (days).", show_default=True)
@click.option("--label-csv",      default=None, help="Path to labelled training CSV.")
@click.option("--detections-csv", default=None, help="Path to existing detections CSV (skip detect step).")
@click.option("--mode",
              default="full",
              type=click.Choice(["full", "train", "detect", "fit", "report"]),
              help="Pipeline mode.", show_default=True)
@click.option("--no-retrain",  is_flag=True,  help="Use saved model; skip training.")
@click.option("--run-mcmc",    is_flag=True,  help="Run emcee MCMC for parameter uncertainties (slow).")
@click.option("--model-type",
              default="rf",
              type=click.Choice(["rf", "gb", "xgb"]),
              help="Classifier type (rf=RandomForest, gb=GradientBoosting, xgb=XGBoost).", show_default=True)
@click.option("--output-dir",  default="results", help="Output directory.", show_default=True)
@click.option("--log-level",   default="INFO",   help="Log level.", show_default=True)
def main(
    sector, n_targets, sde_threshold, period_min, period_max,
    label_csv, detections_csv, mode, no_retrain, run_mcmc,
    model_type, output_dir, log_level,
):
    """🔭 ISRO Exoplanet Detection Pipeline"""

    logger.remove()
    logger.add(lambda msg: print(msg, end=""), level=log_level, colorize=True,
               format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}")

    start_time = time.time()
    logger.info("=" * 65)
    logger.info("  ISRO Exoplanet Detection Pipeline  v1.0")
    logger.info(f"  Mode: {mode.upper()}   Sector: {sector}   Targets: {n_targets}")
    logger.info("=" * 65)

    (
        ensure_dirs, download_known_planets, bulk_download, query_sector_targets,
        preprocess_fits, search_all, candidates_to_dataframe,
        build_feature_dataframe, load_labelled_dataset, FEATURE_COLS,
        train, load_model, classify_candidates, feature_importance, evaluate_on_test,
        fit_all_planets,
        plot_lightcurve, plot_periodogram, plot_phase_fold,
        plot_candidate_panel, plot_classification_summary,
        plot_feature_importance, build_interactive_dashboard,
        generate_report,
    ) = _import_src()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    ensure_dirs()
    metrics = {}

    # ──────────────────────────────────────────────────────────────────
    # STEP 1: Download
    # ──────────────────────────────────────────────────────────────────
    if mode in ("full", "detect"):
        logger.info("\n[Step 1/7] Downloading TESS light curves…")

        # Use known planet hosts as a reliable test set
        downloaded = download_known_planets(sector=sector, max_targets=min(n_targets, 20))

        # If we need more, query the sector catalog
        if len(downloaded) < n_targets:
            try:
                targets_df = query_sector_targets(sector, max_targets=n_targets - len(downloaded))
                tic_ids = targets_df["TIC"].astype(str).tolist()
                more = bulk_download(tic_ids, sector=sector)
                downloaded.extend(more)
            except Exception as exc:
                logger.warning(f"  Sector catalog query failed: {exc}. Continuing with what we have.")

        logger.success(f"  Downloaded {len(downloaded)} light curves.")

    # ──────────────────────────────────────────────────────────────────
    # STEP 2: Preprocess
    # ──────────────────────────────────────────────────────────────────
    if mode in ("full", "detect"):
        logger.info("\n[Step 2/7] Preprocessing light curves…")
        raw_dir = Path("data/raw")
        processed_dir = Path("data/processed")
        fits_files = sorted(raw_dir.glob("*.fits"))
        n_ok = 0
        for fp in fits_files:
            result = preprocess_fits(fp, processed_dir)
            if result:
                n_ok += 1
        logger.success(f"  Preprocessed {n_ok}/{len(fits_files)} light curves.")
        n_total = len(fits_files)
    else:
        n_total = n_targets

    # ──────────────────────────────────────────────────────────────────
    # STEP 3: Period search (TLS)
    # ──────────────────────────────────────────────────────────────────
    candidates = []
    feature_df = pd.DataFrame()

    if mode in ("full", "detect"):
        logger.info(f"\n[Step 3/7] Running TLS + BLS period search (SDE > {sde_threshold})…")
        candidates = search_all(
            processed_dir=Path("data/processed"),
            sde_threshold=sde_threshold,
            period_min=period_min,
            period_max=period_max,
        )

        if candidates:
            summary_df = candidates_to_dataframe(candidates)
            summary_path = output_dir / "tls_summary.csv"
            summary_df.to_csv(summary_path, index=False)
            logger.info(f"  TLS summary saved → {summary_path}")

            # Build feature matrix
            feature_df = build_feature_dataframe(candidates, Path("data/processed"))

    # ──────────────────────────────────────────────────────────────────
    # STEP 4: Train / load classifier
    # ──────────────────────────────────────────────────────────────────
    pipeline_clf = None

    if mode in ("full", "train", "detect"):
        if not no_retrain and label_csv:
            logger.info(f"\n[Step 4/7] Training classifier on {label_csv}…")
            try:
                X_train, y_train = load_labelled_dataset(Path(label_csv))
                pipeline_clf, metrics = train(
                    X_train, y_train,
                    model_type=model_type,
                )
                # Save metrics
                metrics_path = output_dir / "classifier_metrics.json"
                with open(metrics_path, "w") as f:
                    json.dump(metrics, f, indent=2, default=str)
                logger.info(f"  Metrics saved → {metrics_path}")

                # Feature importance plot
                fi_df = feature_importance(pipeline_clf)
                if len(fi_df) > 0:
                    plot_feature_importance(fi_df, figures_dir)

            except Exception as exc:
                logger.error(f"  Training failed: {exc}. Attempting to load saved model.")
                pipeline_clf = load_model()
        else:
            logger.info("\n[Step 4/7] Loading saved classifier…")
            pipeline_clf = load_model()
            if pipeline_clf is None and len(feature_df) > 0:
                logger.warning("  No saved model found. Candidates will not be classified.")

    # ──────────────────────────────────────────────────────────────────
    # STEP 5: Classify
    # ──────────────────────────────────────────────────────────────────
    detections_df = pd.DataFrame()

    if mode in ("full", "detect") and len(feature_df) > 0:
        logger.info("\n[Step 5/7] Classifying candidates…")

        if pipeline_clf is not None:
            detections_df = classify_candidates(candidates, feature_df, pipeline_clf)
        else:
            # No classifier: label all passing SDE as OTHER
            detections_df = feature_df.copy()
            detections_df["predicted_class"] = "OTHER"
            detections_df["confidence"] = 0.0

        detections_path = output_dir / "detections.csv"
        detections_df.to_csv(detections_path, index=False)
        logger.success(f"  Detections saved → {detections_path}")

    elif mode == "fit" and detections_csv:
        logger.info(f"\n  Loading detections from {detections_csv}…")
        detections_df = pd.read_csv(detections_csv)

    # ──────────────────────────────────────────────────────────────────
    # STEP 5b: Visualise candidates
    # ──────────────────────────────────────────────────────────────────
    if len(detections_df) > 0 and mode in ("full", "detect"):
        logger.info("  Generating candidate plots…")
        # Plot classification summary
        if "predicted_class" in detections_df.columns:
            plot_classification_summary(detections_df, figures_dir)

        # Plot individual PLANET candidates (top 5 by confidence)
        planet_cands = (
            detections_df[detections_df["predicted_class"] == "PLANET"]
            .sort_values("confidence", ascending=False)
            .head(5)
        ) if "predicted_class" in detections_df.columns else pd.DataFrame()

        proc_dir = Path("data/processed")
        for _, row in planet_cands.iterrows():
            tic_id = str(row["tic_id"])
            csv_files = list(proc_dir.glob(f"TIC{tic_id}*.csv"))
            if not csv_files:
                continue
            try:
                lc_df = pd.read_csv(csv_files[0])
                time_arr = lc_df["time"].values
                flux_arr = lc_df["flux"].values

                # Find matching candidate raw results
                matching = [c for c in candidates if c.tic_id == tic_id]
                tls_periods = None
                tls_power = None
                bls_periods_arr = None
                bls_power_arr = None
                bls_period_val = None
                bls_sde_val = 0.0
                if matching:
                    raw = matching[0].raw_results
                    tls_periods = np.array(raw.get("periods", []))
                    tls_power = np.array(raw.get("power", []))
                    # BLS data
                    bls_raw = getattr(matching[0], "bls_raw", {})
                    if bls_raw.get("periods"):
                        bls_periods_arr = np.array(bls_raw["periods"])
                        bls_power_arr = np.array(bls_raw["power_spectrum"])
                    bls_period_val = getattr(matching[0], "bls_period", None)
                    bls_sde_val = getattr(matching[0], "bls_sde", 0.0)

                plot_candidate_panel(
                    time=time_arr,
                    flux=flux_arr,
                    period=float(row.get("period_d", 1.0)),
                    t0=float(row.get("t0_btjd", time_arr[0])),
                    duration_hr=float(row.get("duration_hr", 3.0)),
                    depth_ppm=float(row.get("depth_ppm", 1000.0)),
                    tls_periods=tls_periods,
                    tls_power=tls_power,
                    bls_periods=bls_periods_arr,
                    bls_power=bls_power_arr,
                    bls_period=bls_period_val,
                    bls_sde=bls_sde_val,
                    tic_id=tic_id,
                    label=str(row.get("predicted_class", "PLANET")),
                    confidence=float(row.get("confidence", 0.0)),
                    snr=float(row.get("snr", 0.0)),
                    sde=float(row.get("sde", 0.0)),
                    save_dir=figures_dir,
                )
            except Exception as exc:
                logger.warning(f"  Plot failed for TIC {tic_id}: {exc}")

        # Interactive dashboard
        build_interactive_dashboard(detections_df, output_dir)

    # ──────────────────────────────────────────────────────────────────
    # STEP 6: Parameter fitting
    # ──────────────────────────────────────────────────────────────────
    if mode in ("full", "fit") and len(detections_df) > 0:
        logger.info("\n[Step 6/7] Fitting transit models for PLANET candidates…")
        detections_df = fit_all_planets(detections_df, run_mcmc=run_mcmc)
        detections_path = output_dir / "detections_fitted.csv"
        detections_df.to_csv(detections_path, index=False)
        logger.success(f"  Fitted detections saved → {detections_path}")

    # ──────────────────────────────────────────────────────────────────
    # STEP 7: Generate PDF report
    # ──────────────────────────────────────────────────────────────────
    if mode in ("full", "report"):
        logger.info("\n[Step 7/7] Generating PDF report…")
        report_path = generate_report(
            detections_df=detections_df if len(detections_df) > 0 else pd.DataFrame(),
            metrics=metrics,
            figures_dir=figures_dir,
            output_path=output_dir / "ISRO_Exoplanet_Report.pdf",
            sector=sector,
            n_total=n_total,
            sde_threshold=sde_threshold,
        )
        logger.success(f"  Report saved → {report_path}")

    # ──────────────────────────────────────────────────────────────────
    # Summary
    # ──────────────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    logger.info("\n" + "=" * 65)
    logger.success(f"  Pipeline complete in {elapsed/60:.1f} min")

    if len(detections_df) > 0 and "predicted_class" in detections_df.columns:
        for cls in ["PLANET", "EB", "BLEND", "OTHER"]:
            n = (detections_df["predicted_class"] == cls).sum()
            logger.info(f"    {cls:10s}: {n}")

    logger.info("=" * 65)


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
