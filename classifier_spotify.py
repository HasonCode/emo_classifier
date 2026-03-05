"""Spotify-based emo music classifier using audio features."""

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import joblib

from config import DATA_DIR, MODELS_DIR, SPOTIFY_ALL_FEATURES


def load_dataset() -> pd.DataFrame:
    """Load the Spotify emo dataset."""
    path = DATA_DIR / "spotify_emo_dataset.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run: python -m spotify_data "
            "(or python -m generate_seed_data if rate-limited)"
        )
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        raise FileNotFoundError(
            f"Dataset at {path} is empty. Run: python -m spotify_data first."
        )
    if df.empty or "is_emo" not in df.columns:
        raise FileNotFoundError(
            f"Dataset at {path} is empty or invalid. Run: python -m spotify_data first."
        )
    return df


def prepare_features(df: pd.DataFrame):
    """Prepare feature matrix and labels."""
    # Use available features (full or fallback)
    feature_cols = [c for c in SPOTIFY_ALL_FEATURES if c in df.columns]
    X = df[feature_cols].fillna(0)
    y = df["is_emo"]
    return X, y, feature_cols


def train_classifier(df: pd.DataFrame = None, save: bool = True):
    """Train the emo classifier."""
    if df is None:
        df = load_dataset()

    X, y, feature_cols = prepare_features(df)
    if not feature_cols:
        raise ValueError(
            "No feature columns found. Dataset may be corrupted or from an unsupported source."
        )
    if len(y.unique()) < 2:
        raise ValueError(
            "Need both emo and non-emo tracks. Collect more data with: python -m spotify_data"
        )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Use simpler, well-calibrated model for few features (fallback case)
    n_features = len(feature_cols)
    if n_features <= 4:
        base_clf = LogisticRegression(
            max_iter=1000, random_state=42, class_weight="balanced", C=0.5
        )
        clf = CalibratedClassifierCV(base_clf, method="sigmoid", cv=5)
    else:
        base_clf = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=6,
            min_samples_leaf=10,
            class_weight="balanced",
        )
        clf = CalibratedClassifierCV(base_clf, method="isotonic", cv=5)
    clf.fit(X_train_scaled, y_train)

    y_pred = clf.predict(X_test_scaled)
    print("Spotify Emo Classifier - Performance")
    print("=" * 40)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")
    print(classification_report(y_test, y_pred, target_names=["Not Emo", "Emo"]))

    if save:
        model_path = MODELS_DIR / "emo_classifier_spotify.joblib"
        scaler_path = MODELS_DIR / "emo_scaler_spotify.joblib"
        meta_path = MODELS_DIR / "emo_classifier_spotify_meta.json"
        joblib.dump(clf, model_path)
        joblib.dump(scaler, scaler_path)
        with open(meta_path, "w") as f:
            json.dump({"feature_columns": feature_cols}, f, indent=2)
        print(f"\nModel saved to {model_path}")

    return clf, scaler, feature_cols


def load_classifier():
    """Load trained classifier and scaler."""
    model_path = MODELS_DIR / "emo_classifier_spotify.joblib"
    scaler_path = MODELS_DIR / "emo_scaler_spotify.joblib"
    meta_path = MODELS_DIR / "emo_classifier_spotify_meta.json"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found. Train first: python -m classifier_spotify"
        )

    clf = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    with open(meta_path) as f:
        meta = json.load(f)
    return clf, scaler, meta["feature_columns"]


# Require higher confidence to predict Emo (reduces false positives like pop hits)
EMO_DECISION_THRESHOLD = 0.75


def predict(features: dict) -> tuple[str, float]:
    """
    Predict if a song is emo from Spotify audio features.
    Returns (label, probability) e.g. ("Emo", 0.85)
    Uses EMO_DECISION_THRESHOLD to reduce false emo predictions.
    """
    import numpy as np
    clf, scaler, feature_cols = load_classifier()
    row = {}
    for c in feature_cols:
        v = features.get(c, 0)
        arr = np.asarray(v)
        row[c] = float(arr.flat[0]) if arr.size > 0 else 0.0
    X = pd.DataFrame([row])
    X_scaled = scaler.transform(X)
    proba = clf.predict_proba(X_scaled)[0]
    emo_prob = float(np.asarray(proba[1]).flat[0])
    pred = 1 if emo_prob >= EMO_DECISION_THRESHOLD else 0
    label = "Emo" if pred == 1 else "Not Emo"
    return label, emo_prob


def main():
    """Train and save the classifier."""
    train_classifier()


if __name__ == "__main__":
    main()
