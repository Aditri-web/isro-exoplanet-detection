# 🔭 ISRO Exoplanet Detection Pipeline

> AI-enabled detection and classification of exoplanet transit signals from noisy TESS astronomical light curves.

---

## Pipeline Overview

```
TESS MAST  →  Preprocess  →  TLS Period Search  →  ML Classifier  →  Transit Fitting  →  Report
```

| Stage | Module | Description |
|---|---|---|
| Download | `src/download.py` | TESS PDCSAP LCs via lightkurve / MAST |
| Preprocess | `src/preprocess.py` | Quality mask, sigma-clip, Wotan detrending, normalise |
| Period Search | `src/period_search.py` | TransitLeastSquares (TLS), SDE threshold, BLS, EB diagnostics |
| Features | `src/features.py` | 20-D feature vector construction |
| Classify | `src/classifier.py` | Random Forest + XGBoost → PLANET / EB / BLEND / OTHER |
| Fit | `src/fitting.py` | batman transit model, Nelder-Mead + optional emcee MCMC |
| Visualise | `src/visualise.py` | LC plot, periodogram, phase-fold, Plotly dashboard, BLS periodogram |
| Report | `src/report.py` | Auto-generated 3-page PDF |

---

## Quick Start

### 1. Set up environment
```bash
cd isro-exoplanet-detection
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Generate synthetic training data (if curated CSV not yet available)
```bash
python scripts/generate_synthetic_labels.py --n-samples 2000 --output data/labels/synthetic_labels.csv
```

### 3. Run full pipeline (small test: 50 targets from Sector 1)
```bash
python pipeline.py \
  --sector 1 \
  --n-targets 50 \
  --sde-threshold 7 \
  --label-csv data/labels/synthetic_labels.csv \
  --output-dir results/
```

### 4. Run on full sector (~500+ targets)
```bash
python pipeline.py \
  --sector 1 \
  --n-targets 500 \
  --sde-threshold 7 \
  --label-csv data/labels/curated.csv \
  --run-mcmc \
  --output-dir results/
```

### 5. Run tests
```bash
pytest tests/ -v
```

---

## Pipeline Modes

| Mode | Description |
|---|---|
| `full` | Run all steps end-to-end (default) |
| `train` | Train classifier only |
| `detect` | Download + preprocess + period search + classify |
| `fit` | Parameter fitting only (load existing detections) |
| `report` | Generate PDF report only |

---

## Output Files

```
results/
├── tls_summary.csv          ← all TLS results
├── detections.csv           ← classified candidates
├── detections_fitted.csv    ← with batman fit parameters
├── classifier_metrics.json  ← training metrics
├── dashboard.html           ← interactive Plotly dashboard
├── ISRO_Exoplanet_Report.pdf
└── figures/
    ├── TIC*_panel.png       ← 3-panel candidate figure
    ├── TIC*_lightcurve.png
    ├── TIC*_periodogram.png
    ├── TIC*_phasefold.png
    ├── classification_summary.png
    └── feature_importance.png
```

---

## Classification Features

| Feature | Physical meaning |
|---|---|
| `sde` | Signal Detection Efficiency (TLS SNR proxy) |
| `snr` | Transit SNR |
| `depth_ppm` | Transit depth in ppm |
| `duration_ratio` | Duration / Period (geometric constraint) |
| `rp_rs` | Radius ratio estimate |
| `n_transits` | Number of observed transits |
| `odd_even_diff` | Odd vs even transit depth mismatch (EB flag) |
| `secondary_depth` | Depth at phase 0.5 (EB/blend flag) |
| `shape_score` | U-shape (planet) vs V-shape (EB) |
| `scatter_ratio` | In-transit / out-of-transit scatter |

---

## Using the Official Curated Dataset

When ISRO provides the curated labelled dataset, place it at `data/labels/curated.csv`.

Required columns:
- Feature columns (see `src/features.py` → `FEATURE_COLS`) **or** raw TIC IDs (run preprocessing first)
- A `label` column with values: `PLANET`, `EB`, `BLEND`, `OTHER`

Then run:
```bash
python pipeline.py --label-csv data/labels/curated.csv --no-retrain false
```

---

## Dependencies

- `lightkurve` — TESS data access
- `transitleastsquares` — TLS period search
- `wotan` — biweight detrending
- `batman-package` — transit model
- `scikit-learn` — Random Forest
- `emcee` — MCMC (optional)
- `plotly` — interactive dashboard
- `fpdf2` — PDF report
