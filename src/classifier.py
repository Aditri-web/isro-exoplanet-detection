"""
src/classifier.py
-----------------
Trains and applies the transit classification model.

Classes:  0=PLANET  1=EB  2=BLEND  3=OTHER

Pipeline:
  - Random Forest (primary) with class balancing
  - Optional XGBoost ensemble
  - Exports feature importance
  - Saves/loads model with joblib
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features import FEATURE_COLS, CLASS_LABELS, LABEL_MAP

warnings.filterwarnings("ignore")

MODEL_PATH = Path("models/rf_classifier.pkl")


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def build_pipeline(model_type: str = "rf") -> Pipeline:
    """
    Build a scikit-learn Pipeline with scaler + classifier.

    Parameters
    ----------
    model_type : str
        "rf" (Random Forest) or "gb" (Gradient Boosting / XGBoost-like)
    """
    if model_type == "rf":
        clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
    elif model_type == "gb":
        clf = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", clf),
    ])
    return pipeline


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    X: pd.DataFrame,
    y: np.ndarray,
    model_type: str = "rf",
    cv_folds: int = 5,
    output_path: Path = MODEL_PATH,
) -> Tuple[Pipeline, Dict]:
    """
    Train the classifier with cross-validation.

    Parameters
    ----------
    X : pd.DataFrame  feature matrix
    y : np.ndarray    integer class labels
    model_type : str
    cv_folds : int
    output_path : Path  where to save the trained model

    Returns
    -------
    (fitted Pipeline, metrics dict)
    """
    X_arr = X[FEATURE_COLS].values.astype(np.float32)

    # Replace NaN/Inf with 0
    X_arr = np.nan_to_num(X_arr, nan=0.0, posinf=0.0, neginf=0.0)

    logger.info(f"Training {model_type.upper()} classifier on {len(y)} samples…")
    class_counts = pd.Series(y).value_counts().sort_index()
    for cls_id, count in class_counts.items():
        logger.info(f"  Class {CLASS_LABELS.get(cls_id, cls_id)}: {count} samples")

    pipeline = build_pipeline(model_type)

    # Cross-validation
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X_arr, y, cv=cv, scoring="f1_weighted", n_jobs=-1)
    logger.info(f"  CV F1 (weighted): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # Final fit on all training data
    pipeline.fit(X_arr, y)

    # Training metrics
    y_pred = pipeline.predict(X_arr)
    train_acc = accuracy_score(y, y_pred)
    train_f1 = f1_score(y, y_pred, average="weighted")

    metrics = {
        "cv_f1_mean": float(cv_scores.mean()),
        "cv_f1_std":  float(cv_scores.std()),
        "train_accuracy": float(train_acc),
        "train_f1_weighted": float(train_f1),
        "n_samples": int(len(y)),
        "class_distribution": class_counts.to_dict(),
    }

    logger.success(
        f"  Train accuracy: {train_acc:.3f}  |  Train F1: {train_f1:.3f}"
    )

    # Print classification report
    report = classification_report(
        y, y_pred,
        target_names=[CLASS_LABELS.get(i, str(i)) for i in sorted(set(y))],
    )
    logger.info(f"\n{report}")

    # Save model
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output_path)
    logger.success(f"  Model saved → {output_path}")

    return pipeline, metrics


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def load_model(model_path: Path = MODEL_PATH) -> Optional[Pipeline]:
    """Load a previously trained model."""
    if not model_path.exists():
        logger.warning(f"Model not found at {model_path}. Train first.")
        return None
    return joblib.load(model_path)


def predict(
    pipeline: Pipeline,
    X: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Classify candidates.

    Returns
    -------
    labels : np.ndarray of int
    probabilities : np.ndarray of shape (N, n_classes)
    """
    X_arr = X[FEATURE_COLS].values.astype(np.float32)
    X_arr = np.nan_to_num(X_arr, nan=0.0, posinf=0.0, neginf=0.0)

    labels = pipeline.predict(X_arr)
    probs = pipeline.predict_proba(X_arr)
    return labels, probs


def classify_candidates(
    candidates: List,  # List[TransitCandidate]
    feature_df: pd.DataFrame,
    pipeline: Pipeline,
) -> pd.DataFrame:
    """
    Apply classifier to a list of transit candidates.

    Returns a DataFrame with classification results appended.
    """
    # Only classify those that passed SDE threshold
    mask = feature_df["passed_threshold"].fillna(False)
    if not mask.any():
        logger.warning("No candidates passed the SDE threshold. Nothing to classify.")
        feature_df["predicted_class"] = "OTHER"
        feature_df["class_label"] = 3
        feature_df["confidence"] = 0.0
        for cls in CLASS_LABELS.values():
            feature_df[f"prob_{cls}"] = 0.0
        return feature_df

    X_cand = feature_df.loc[mask, FEATURE_COLS]
    labels, probs = predict(pipeline, X_cand)

    # Write results back
    feature_df["class_label"] = 3  # default OTHER
    feature_df["predicted_class"] = "OTHER"
    feature_df["confidence"] = 0.0
    for cls in CLASS_LABELS.values():
        feature_df[f"prob_{cls}"] = 0.0

    idx = feature_df[mask].index
    feature_df.loc[idx, "class_label"] = labels
    feature_df.loc[idx, "predicted_class"] = [CLASS_LABELS.get(l, "OTHER") for l in labels]
    feature_df.loc[idx, "confidence"] = probs.max(axis=1)

    for i, cls in CLASS_LABELS.items():
        if i < probs.shape[1]:
            feature_df.loc[idx, f"prob_{cls}"] = probs[:, i]

    # Log summary
    classified = feature_df.loc[mask]
    for cls_name in CLASS_LABELS.values():
        count = (classified["predicted_class"] == cls_name).sum()
        logger.info(f"  {cls_name}: {count} targets")

    return feature_df


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

def feature_importance(pipeline: Pipeline) -> pd.DataFrame:
    """Extract feature importance from the trained Random Forest."""
    clf = pipeline.named_steps["clf"]
    if not hasattr(clf, "feature_importances_"):
        return pd.DataFrame()

    importance = clf.feature_importances_
    df = pd.DataFrame({
        "feature": FEATURE_COLS,
        "importance": importance,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    return df


def evaluate_on_test(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
) -> Dict:
    """Evaluate model on a held-out test set."""
    labels, probs = predict(pipeline, X_test)
    acc = accuracy_score(y_test, labels)
    f1 = f1_score(y_test, labels, average="weighted")
    cm = confusion_matrix(y_test, labels)
    report = classification_report(
        y_test, labels,
        target_names=[CLASS_LABELS.get(i, str(i)) for i in sorted(set(y_test))],
        output_dict=True,
    )
    logger.success(f"Test accuracy: {acc:.3f}  |  Test F1: {f1:.3f}")
    return {
        "accuracy": float(acc),
        "f1_weighted": float(f1),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }
