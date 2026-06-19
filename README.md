# 🔭 ISRO Exoplanet Detection Pipeline

> **AI-driven detection and classification of exoplanet transit signals from noisy TESS light curves.**

An end-to-end pipeline that downloads TESS photometric data, identifies periodic dip features using dual period-search algorithms (BLS + TLS), classifies candidates into astrophysical categories using machine learning, fits physical transit models for parameter estimation, and generates publication-ready visualisations and reports.

---

## 🚀 Pipeline Architecture

```
TESS MAST ─→ Preprocess ─→ BLS + TLS Search ─→ ML Classifier ─→ Transit Fitting ─→ Report
              (Wotan)       (dual algorithm)    (RF / XGBoost)    (batman + MCMC)    (PDF)
```

| Stage | Module | Description |
|---|---|---|
| **Download** | `src/download.py` | TESS PDCSAP light curves via `lightkurve` / MAST, local FITS caching |
| **Preprocess** | `src/preprocess.py` | Quality mask, sigma-clip, Wotan biweight detrending, normalise to unit flux |
| **Period Search** | `src/period_search.py` | **BLS** (Box Least Squares) + **TLS** (Transit Least Squares) run in parallel with cross-validation |
| **Features** | `src/features.py` | 20-dimensional feature vector (transit metrics + EB diagnostics + BLS features) |
| **Classify** | `src/classifier.py` | Random Forest / XGBoost → PLANET / EB / BLEND / OTHER with calibrated confidence |
| **Fit** | `src/fitting.py` | `batman` transit model, 5-parameter Nelder-Mead + optional `emcee` MCMC |
| **Visualise** | `src/visualise.py` | 4-panel candidate figures, BLS + TLS periodograms, interactive Plotly dashboard |
| **Report** | `src/report.py` | Auto-generated PDF report with methodology, results, and parameter tables |

---

## 📐 Estimated Parameters

The pipeline estimates the following transit parameters with uncertainties:

| Parameter | Method | Uncertainty |
|---|---|---|
| **Orbital Period** | BLS + TLS grid search → refined via Nelder-Mead fitting (±0.5%) | Finite-difference Hessian / MCMC posterior |
| **Transit Duration** | Computed from fitted a/R★, inclination, Rp/R★ | Numerical error propagation |
| **Transit Depth** | (Rp/R★)² from batman model fit | Propagated from Rp/R★ uncertainty |
| **Rp/R★ (radius ratio)** | batman transit model fit | Hessian σ or MCMC 1σ |
| **a/R★ (scaled semi-major axis)** | batman transit model fit | Hessian σ or MCMC 1σ |
| **Inclination** | batman transit model fit | Hessian σ or MCMC 1σ |
| **Mid-transit time (T₀)** | BLS/TLS initial → refined in fitting | Hessian σ or MCMC 1σ |

---

## 🔬 Detection Algorithms

### Box Least Squares (BLS)
- **Implementation:** `astropy.timeseries.BoxLeastSquares` (Kovács, Zucker & Mazeh 2002)
- **Method:** Searches for periodic box-shaped dips across a grid of trial periods and durations
- **Output:** BLS periodogram, best-fit period, depth, duration, SDE, SNR

### Transit Least Squares (TLS)
- **Implementation:** `transitleastsquares` (Hippke & Heller 2019)
- **Method:** Uses optimised transit-shaped templates matched to realistic limb-darkened stellar models
- **Output:** TLS periodogram, best-fit period, depth, duration, SDE, SNR

Both algorithms run independently on every light curve. **Period agreement** between BLS and TLS (including harmonic checking at P, 2P, P/2, P/3) is computed as an additional classification feature. If TLS misses a signal but BLS detects it above the SDE threshold, the candidate is still flagged.

---

## 🤖 Classification

### Feature Vector (20 dimensions)

| Feature | Physical Meaning |
|---|---|
| `sde` | TLS Signal Detection Efficiency |
| `snr` | Transit signal-to-noise ratio |
| `depth_ppm` | Transit depth in parts per million |
| `log_depth_ppm` | Log₁₀ of transit depth |
| `duration_hr` | Transit duration in hours |
| `duration_ratio` | Duration / Period (geometric constraint) |
| `rp_rs` | Radius ratio estimate (Rp/R★) |
| `log_rp_rs` | Log₁₀ of radius ratio |
| `n_transits` | Number of observed transits in the light curve |
| `odd_even_diff` | Odd vs even transit depth mismatch (EB diagnostic) |
| `secondary_depth` | Flux depth at phase 0.5 (secondary eclipse → EB flag) |
| `secondary_depth_ratio` | Secondary / primary depth ratio |
| `shape_score` | U-shape (planet) vs V-shape (EB) ingress fraction |
| `scatter_in` | Std. deviation of in-transit flux |
| `scatter_out` | Std. deviation of out-of-transit flux |
| `scatter_ratio` | In/out scatter ratio |
| `bls_sde` | BLS Signal Detection Efficiency |
| `bls_snr` | BLS signal-to-noise ratio |
| `bls_power` | BLS peak power (SR statistic) |
| `period_agreement` | BLS/TLS period agreement (0 = perfect match) |

### Classifiers

| Model | Flag | Description |
|---|---|---|
| **Random Forest** | `--model-type rf` | 300 trees, balanced class weights (default) |
| **Gradient Boosting** | `--model-type gb` | sklearn GradientBoosting, 200 estimators |
| **XGBoost** | `--model-type xgb` | L1/L2 regularised, recommended for production |

All classifiers are wrapped with **`CalibratedClassifierCV`** (Platt scaling) to produce true calibrated probability estimates rather than raw voting fractions.

### Output Classes
- **PLANET** — candidate exoplanet transit
- **EB** — eclipsing binary
- **BLEND** — background eclipsing binary / blended source
- **OTHER** — instrumental artefact, noise, or stellar variability

---

## ⚡ Quick Start

### 1. Set up environment

```bash
git clone https://github.com/Shourya1507/isro-exoplanet-detection.git
cd isro-exoplanet-detection
python -m venv .venv

# Linux/Mac
source .venv/bin/activate

# Windows
.venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Generate synthetic training data

```bash
python scripts/generate_synthetic_labels.py --n-samples 2000 --output data/labels/synthetic_labels.csv
```

### 3. Run the pipeline (small test: 50 targets)

```bash
python pipeline.py \
  --sector 1 \
  --n-targets 50 \
  --sde-threshold 7 \
  --label-csv data/labels/synthetic_labels.csv \
  --output-dir results/
```

### 4. Run with XGBoost + MCMC (production)

```bash
python pipeline.py \
  --sector 1 \
  --n-targets 500 \
  --sde-threshold 7 \
  --model-type xgb \
  --run-mcmc \
  --label-csv data/labels/synthetic_labels.csv \
  --output-dir results/
```

### 5. Run tests

```bash
pytest tests/ -v
```

---

## 🎛️ CLI Options

| Option | Default | Description |
|---|---|---|
| `--sector` | `1` | TESS sector number |
| `--n-targets` | `50` | Number of targets to process |
| `--sde-threshold` | `7.0` | Minimum SDE for candidate detection |
| `--period-min` | `0.5` | Minimum search period (days) |
| `--period-max` | `27.0` | Maximum search period (days) |
| `--label-csv` | — | Path to labelled training CSV |
| `--model-type` | `rf` | Classifier: `rf`, `gb`, or `xgb` |
| `--mode` | `full` | Pipeline mode: `full`, `train`, `detect`, `fit`, `report` |
| `--run-mcmc` | `false` | Enable emcee MCMC for parameter uncertainties |
| `--no-retrain` | `false` | Skip classifier retraining, use saved model |
| `--output-dir` | `results` | Output directory |

---

## 📊 Pipeline Modes

| Mode | Steps Executed |
|---|---|
| `full` | Download → Preprocess → Search → Classify → Fit → Visualise → Report |
| `train` | Train classifier only (requires `--label-csv`) |
| `detect` | Download → Preprocess → Search → Classify |
| `fit` | Transit fitting only (loads existing detections) |
| `report` | Generate PDF report only |

---

## 📁 Project Structure

```
isro-exoplanet-detection/
├── pipeline.py                          # Main CLI entry point (7-step orchestrator)
├── requirements.txt                     # Python dependencies
├── .gitignore
│
├── src/
│   ├── __init__.py
│   ├── download.py                      # TESS data download (lightkurve + MAST)
│   ├── preprocess.py                    # Light curve preprocessing (Wotan detrending)
│   ├── period_search.py                 # BLS + TLS period search
│   ├── features.py                      # 20-D feature vector construction
│   ├── classifier.py                    # RF / XGBoost classifier with calibration
│   ├── fitting.py                       # batman transit model fitting
│   ├── visualise.py                     # Matplotlib + Plotly visualisations
│   └── report.py                        # PDF report generation (fpdf2)
│
├── scripts/
│   └── generate_synthetic_labels.py     # Synthetic training data generator
│
├── data/
│   ├── raw/                             # Downloaded FITS files
│   ├── processed/                       # Preprocessed CSVs
│   └── labels/
│       └── synthetic_labels.csv         # 8000 synthetic training samples
│
├── tests/
│   ├── test_classifier.py              # Classifier unit tests
│   └── test_preprocess.py              # Preprocessing unit tests
│
├── models/                              # Saved classifier pickles
│
└── results/
    ├── tls_summary.csv                  # All TLS + BLS results
    ├── detections.csv                   # Classified candidates
    ├── detections_fitted.csv            # With fitted transit parameters
    ├── classifier_metrics.json          # Training metrics
    ├── dashboard.html                   # Interactive Plotly dashboard
    ├── ISRO_Exoplanet_Report.pdf        # Auto-generated report
    └── figures/
        ├── TIC*_panel.png              # 4-panel candidate figure
        ├── TIC*_lightcurve.png
        ├── TIC*_periodogram.png
        ├── TIC*_bls_periodogram.png
        ├── TIC*_phasefold.png
        ├── classification_summary.png
        └── feature_importance.png
```

---

## 🔧 Dependencies

| Package | Purpose |
|---|---|
| `lightkurve` ≥ 2.4 | TESS data access and light curve operations |
| `astropy` ≥ 5.3 | FITS I/O, time systems, **BoxLeastSquares (BLS)** |
| `transitleastsquares` ≥ 1.0 | TLS period search |
| `wotan` ≥ 1.10 | Robust biweight detrending |
| `batman-package` ≥ 2.4 | Analytic transit model (Kreidberg 2015) |
| `scikit-learn` ≥ 1.3 | Random Forest, StandardScaler, calibration |
| `xgboost` ≥ 2.0 | XGBoost classifier (optional, recommended) |
| `emcee` ≥ 3.1 | MCMC posterior sampling (optional) |
| `scipy` ≥ 1.11 | Nelder-Mead optimisation, statistics |
| `matplotlib` ≥ 3.7 | Static publication-quality plots |
| `plotly` ≥ 5.17 | Interactive HTML dashboard |
| `fpdf2` ≥ 2.7 | PDF report generation |
| `loguru` ≥ 0.7 | Structured logging |

---

## 📄 Using a Curated Dataset

When a curated labelled dataset is available, place it at `data/labels/curated.csv`.

**Required columns:**
- Feature columns matching `FEATURE_COLS` in `src/features.py` **or** raw TIC IDs (run preprocessing first)
- A `label` column with values: `PLANET`, `EB`, `BLEND`, `OTHER`

```bash
python pipeline.py --label-csv data/labels/curated.csv --model-type xgb --output-dir results/
```

---

## 📚 References

- Kovács, G., Zucker, S., & Mazeh, T. (2002). *A box-fitting algorithm in the search for periodic transits.* A&A, 391, 369–377. — **BLS algorithm**
- Hippke, M., & Heller, R. (2019). *Optimized transit detection algorithm to search for periodic transits of small planets.* A&A, 623, A39. — **TLS algorithm**
- Kreidberg, L. (2015). *batman: BAsic Transit Model cAlculatioN in Python.* PASP, 127, 1161. — **Transit model**
- Foreman-Mackey, D., et al. (2013). *emcee: The MCMC Hammer.* PASP, 125, 306. — **MCMC sampling**

---
