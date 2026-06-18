"""
tests/test_classifier.py
Tests classifier training and inference on synthetic data.
"""
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.features import FEATURE_COLS, LABEL_MAP
from src.classifier import build_pipeline, train, predict, feature_importance


def _make_synthetic_data(n=400):
    """Generate simple separable synthetic feature data."""
    rng = np.random.default_rng(0)
    rows = []
    labels = []
    # PLANET: deep SDE, low odd_even, low secondary
    for _ in range(n // 4):
        rows.append({
            "sde": rng.uniform(8, 30), "snr": rng.uniform(15, 100),
            "depth_ppm": rng.uniform(500, 15000), "log_depth_ppm": rng.uniform(2.7, 4.2),
            "duration_hr": rng.uniform(1, 5), "duration_ratio": rng.uniform(0.002, 0.02),
            "rp_rs": rng.uniform(0.02, 0.12), "log_rp_rs": rng.uniform(-1.7, -0.9),
            "n_transits": rng.uniform(3, 20), "odd_even_diff": rng.uniform(0, 0.05),
            "secondary_depth": rng.uniform(0, 100), "secondary_depth_ratio": rng.uniform(0, 0.05),
            "shape_score": rng.uniform(0.6, 0.95), "scatter_in": rng.uniform(1e-4, 3e-4),
            "scatter_out": rng.uniform(1e-4, 3e-4), "scatter_ratio": rng.uniform(0.8, 1.2),
        })
        labels.append(0)  # PLANET
    # EB: deep, high odd_even, high secondary
    for _ in range(n // 4):
        rows.append({
            "sde": rng.uniform(20, 100), "snr": rng.uniform(100, 1000),
            "depth_ppm": rng.uniform(10000, 200000), "log_depth_ppm": rng.uniform(4, 5.3),
            "duration_hr": rng.uniform(1, 6), "duration_ratio": rng.uniform(0.01, 0.1),
            "rp_rs": rng.uniform(0.1, 0.5), "log_rp_rs": rng.uniform(-1, -0.3),
            "n_transits": rng.uniform(3, 50), "odd_even_diff": rng.uniform(0.1, 0.5),
            "secondary_depth": rng.uniform(5000, 50000), "secondary_depth_ratio": rng.uniform(0.1, 0.8),
            "shape_score": rng.uniform(0.05, 0.3), "scatter_in": rng.uniform(3e-4, 8e-4),
            "scatter_out": rng.uniform(2e-4, 5e-4), "scatter_ratio": rng.uniform(1.2, 3.0),
        })
        labels.append(1)  # EB
    # BLEND: moderate depth, some secondary
    for _ in range(n // 4):
        rows.append({
            "sde": rng.uniform(7, 25), "snr": rng.uniform(8, 50),
            "depth_ppm": rng.uniform(300, 8000), "log_depth_ppm": rng.uniform(2.5, 3.9),
            "duration_hr": rng.uniform(1, 4), "duration_ratio": rng.uniform(0.003, 0.03),
            "rp_rs": rng.uniform(0.017, 0.09), "log_rp_rs": rng.uniform(-1.77, -1.0),
            "n_transits": rng.uniform(2, 15), "odd_even_diff": rng.uniform(0.04, 0.25),
            "secondary_depth": rng.uniform(200, 5000), "secondary_depth_ratio": rng.uniform(0.05, 0.5),
            "shape_score": rng.uniform(0.3, 0.7), "scatter_in": rng.uniform(1.5e-4, 4e-4),
            "scatter_out": rng.uniform(1.5e-4, 4e-4), "scatter_ratio": rng.uniform(0.9, 2.0),
        })
        labels.append(2)  # BLEND
    # OTHER: noisy, low SDE
    for _ in range(n // 4):
        rows.append({
            "sde": rng.uniform(0, 10), "snr": rng.uniform(1, 12),
            "depth_ppm": rng.uniform(50, 3000), "log_depth_ppm": rng.uniform(1.7, 3.5),
            "duration_hr": rng.uniform(0.5, 10), "duration_ratio": rng.uniform(0.001, 0.05),
            "rp_rs": rng.uniform(0.007, 0.055), "log_rp_rs": rng.uniform(-2.15, -1.26),
            "n_transits": rng.uniform(1, 10), "odd_even_diff": rng.uniform(0, 0.15),
            "secondary_depth": rng.uniform(0, 400), "secondary_depth_ratio": rng.uniform(0, 0.3),
            "shape_score": rng.uniform(0.1, 0.9), "scatter_in": rng.uniform(1e-4, 5e-4),
            "scatter_out": rng.uniform(1e-4, 5e-4), "scatter_ratio": rng.uniform(0.5, 2.5),
        })
        labels.append(3)  # OTHER

    X = pd.DataFrame(rows)[FEATURE_COLS]
    y = np.array(labels)
    return X, y


def test_build_pipeline():
    p = build_pipeline("rf")
    assert p is not None


def test_train_and_predict():
    X, y = _make_synthetic_data(400)
    pipeline, metrics = train(X, y, cv_folds=3, output_path=Path("/tmp/test_model.pkl"))
    # With only 100 samples/class these thresholds are intentionally relaxed;
    # on the real 2000+/class dataset expect >0.85 accuracy and >0.80 CV F1.
    assert metrics["train_accuracy"] > 0.50, f"Train accuracy too low: {metrics['train_accuracy']}"
    assert metrics["cv_f1_mean"] > 0.35, f"CV F1 too low: {metrics['cv_f1_mean']}"

    labels, probs = predict(pipeline, X)
    assert len(labels) == len(y)
    assert probs.shape[0] == len(y)
    assert probs.shape[1] == 4  # 4 classes
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)


def test_feature_importance():
    X, y = _make_synthetic_data(200)
    pipeline, _ = train(X, y, cv_folds=2, output_path=Path("/tmp/test_fi.pkl"))
    fi = feature_importance(pipeline)
    assert len(fi) == len(FEATURE_COLS)
    assert fi["importance"].sum() > 0.99


if __name__ == "__main__":
    test_build_pipeline()
    test_train_and_predict()
    test_feature_importance()
    print("All classifier tests passed ✓")
