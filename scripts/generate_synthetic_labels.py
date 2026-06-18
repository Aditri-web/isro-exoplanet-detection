"""
scripts/generate_synthetic_labels.py
-------------------------------------
Generates a synthetic labelled training CSV to bootstrap the classifier
when the official curated dataset is not yet available.

Creates 4 classes with realistic feature distributions based on
published TESS statistics (Sullivan et al. 2015; Shporer et al. 2017).

Usage:
    python scripts/generate_synthetic_labels.py --n-samples 2000 --output data/labels/synthetic_labels.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
rng = np.random.default_rng(SEED)


def generate_class(n: int, cls: str) -> pd.DataFrame:
    """Generate synthetic feature vectors for a given class."""

    if cls == "PLANET":
        # Hot Jupiters to Earths: shallow (<1%), short period
        depth_ppm   = rng.lognormal(np.log(3000), 1.2, n).clip(50, 30000)
        period_d    = rng.lognormal(np.log(5), 1.0, n).clip(0.5, 27)
        dur_hr      = (period_d / np.pi / 10) * 24 * rng.uniform(0.7, 1.3, n)
        snr         = rng.lognormal(np.log(30), 0.8, n).clip(5, 500)
        sde         = snr * rng.uniform(0.3, 0.7, n)
        n_transits  = np.floor(27 / period_d).clip(2, 50).astype(int)
        odd_even    = rng.exponential(0.02, n).clip(0, 0.08)
        sec_depth   = rng.exponential(50, n).clip(0, 200)
        shape       = rng.beta(8, 2, n)          # U-shaped → high score
        scatter_in  = depth_ppm / 1e6 * rng.uniform(0.8, 1.2, n)
        scatter_out = rng.uniform(1e-4, 3e-4, n)

    elif cls == "EB":
        # Deep dips, secondary eclipse, V-shape, odd-even difference
        depth_ppm   = rng.lognormal(np.log(50000), 0.8, n).clip(5000, 300000)
        period_d    = rng.lognormal(np.log(3), 0.8, n).clip(0.5, 15)
        dur_hr      = (period_d / np.pi / 5) * 24 * rng.uniform(0.7, 1.3, n)
        snr         = rng.lognormal(np.log(200), 0.5, n).clip(20, 2000)
        sde         = snr * rng.uniform(0.4, 0.9, n)
        n_transits  = np.floor(27 / period_d).clip(2, 100).astype(int)
        odd_even    = rng.uniform(0.1, 0.5, n)  # large!
        sec_depth   = rng.uniform(3000, 30000, n)  # large secondary
        shape       = rng.beta(2, 5, n)          # V-shaped → low score
        scatter_in  = depth_ppm / 1e6 * rng.uniform(0.9, 1.1, n)
        scatter_out = rng.uniform(2e-4, 5e-4, n)

    elif cls == "BLEND":
        # Background EB: shallow but odd-even, secondary still present
        depth_ppm   = rng.lognormal(np.log(2000), 1.0, n).clip(200, 20000)
        period_d    = rng.lognormal(np.log(4), 0.9, n).clip(0.5, 20)
        dur_hr      = (period_d / np.pi / 8) * 24 * rng.uniform(0.6, 1.4, n)
        snr         = rng.lognormal(np.log(15), 0.9, n).clip(5, 100)
        sde         = snr * rng.uniform(0.3, 0.7, n)
        n_transits  = np.floor(27 / period_d).clip(2, 40).astype(int)
        odd_even    = rng.uniform(0.05, 0.3, n)
        sec_depth   = rng.lognormal(np.log(500), 1.2, n).clip(50, 5000)
        shape       = rng.beta(4, 4, n)
        scatter_in  = depth_ppm / 1e6 * rng.uniform(0.8, 1.3, n)
        scatter_out = rng.uniform(1.5e-4, 4e-4, n)

    else:  # OTHER (noise, systematics, etc.)
        depth_ppm   = rng.uniform(50, 5000, n)
        period_d    = rng.uniform(0.5, 27, n)
        dur_hr      = rng.uniform(0.5, 10, n)
        snr         = rng.uniform(1, 15, n)
        sde         = rng.uniform(0.5, 9, n)
        n_transits  = rng.integers(1, 20, n)
        odd_even    = rng.uniform(0, 0.2, n)
        sec_depth   = rng.uniform(0, 300, n)
        shape       = rng.uniform(0.1, 0.9, n)
        scatter_in  = rng.uniform(1e-4, 5e-4, n)
        scatter_out = rng.uniform(1e-4, 5e-4, n)

    rp_rs = np.sqrt(np.clip(depth_ppm / 1e6, 1e-8, 1.0))

    df = pd.DataFrame({
        "sde":                  np.clip(sde, 0.5, 999),
        "snr":                  np.clip(snr, 1, 9999),
        "depth_ppm":            np.clip(depth_ppm, 10, 1e6),
        "log_depth_ppm":        np.log10(np.clip(depth_ppm, 10, 1e6)),
        "duration_hr":          np.clip(dur_hr, 0.1, 24),
        "duration_ratio":       np.clip(dur_hr / 24 / period_d, 1e-4, 0.5),
        "rp_rs":                rp_rs,
        "log_rp_rs":            np.log10(rp_rs + 1e-6),
        "n_transits":           n_transits.astype(float),
        "odd_even_diff":        np.clip(odd_even, 0, 1),
        "secondary_depth":      np.clip(sec_depth, 0, 1e6),
        "secondary_depth_ratio": sec_depth / np.clip(depth_ppm, 1, 1e6),
        "shape_score":          np.clip(shape, 0, 1),
        "scatter_in":           scatter_in,
        "scatter_out":          scatter_out,
        "scatter_ratio":        scatter_in / (scatter_out + 1e-8),
        "label":                cls,
    })
    return df


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic training labels.")
    parser.add_argument("--n-samples", type=int, default=2000, help="Samples per class")
    parser.add_argument("--output", type=str, default="data/labels/synthetic_labels.csv")
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    dfs = [generate_class(args.n_samples, cls) for cls in ["PLANET", "EB", "BLEND", "OTHER"]]
    combined = pd.concat(dfs, ignore_index=True).sample(frac=1, random_state=SEED)
    combined.to_csv(args.output, index=False)

    print(f"Saved {len(combined)} samples to {args.output}")
    print(combined["label"].value_counts())


if __name__ == "__main__":
    main()
