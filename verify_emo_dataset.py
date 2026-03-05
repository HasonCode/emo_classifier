"""
Verify emo-labeled audio: flag potential mislabels for manual review.

Uses two methods (cannot actually listen to audio):
1. LocalOutlierFactor: emo samples that are statistical outliers in feature space
2. Cross-validation: emo samples predicted "Not Emo" with high confidence

Run: poetry run python -m verify_emo_dataset
Output: data/audio_verification_report.txt (files to manually listen to)
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import LocalOutlierFactor

from classifier_audio import (
    build_audio_dataset,
    extract_audio_features,
    scan_audio_directory,
)
from config import RAW_AUDIO_DIR


def get_feature_matrix(audio_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    """Build feature matrix and return (df with path, label, features), feature_cols."""
    file_df = scan_audio_directory(audio_dir)
    if file_df.empty:
        raise FileNotFoundError(f"No audio in {audio_dir}")

    feature_cols = [
        "mfcc_mean", "mfcc_std", "chroma_mean", "chroma_std",
        "spectral_centroid_mean", "spectral_rolloff_mean",
        "spectral_bandwidth_mean", "zero_crossing_rate_mean",
        "rms_mean", "tempo",
    ]
    rows = []
    for i, (_, row) in enumerate(file_df.iterrows()):
        try:
            feats = extract_audio_features(row["path"])
            feats["label"] = row["label"]
            feats["path"] = row["path"]
            rows.append(feats)
        except Exception as e:
            print(f"  Skip {row['path']}: {e}")
        if (i + 1) % 200 == 0:
            print(f"  Processed {i + 1}/{len(file_df)} files...")
    return pd.DataFrame(rows), feature_cols


def flag_outliers(df: pd.DataFrame, feature_cols: list[str], n: int = 30) -> set[str]:
    """Flag top N emo samples that are outliers in emo feature space."""
    emo_df = df[df["label"] == "emo"]
    if len(emo_df) < 20:
        return set()
    X = emo_df[feature_cols].fillna(0)
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)
    lof = LocalOutlierFactor(n_neighbors=min(20, len(emo_df) // 2), novelty=False)
    lof.fit(X_sc)
    # Negative scores = outliers (higher magnitude = more outlier-like)
    scores = lof.negative_outlier_factor_
    # Top N most negative = most outlier-like
    worst_idx = np.argsort(scores)[:n]
    return set(emo_df.iloc[worst_idx]["path"].tolist())


def flag_cv_mispredictions(
    df: pd.DataFrame, feature_cols: list[str], n_splits: int = 5, prob_threshold: float = 0.25
) -> set[str]:
    """Flag emo samples repeatedly predicted Not Emo with low emo probability across folds."""
    X = df[feature_cols].fillna(0)
    y = (df["label"] == "emo").astype(int)
    paths = df["path"].values
    emo_indices = np.where(y == 1)[0]

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    misprediction_counts = {i: 0 for i in emo_indices}

    for train_idx, test_idx in kf.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc = scaler.transform(X_test)
        clf = RandomForestClassifier(n_estimators=50, random_state=42, class_weight="balanced")
        clf.fit(X_train_sc, y_train)
        probs = clf.predict_proba(X_test_sc)[:, 1]
        for j, idx in enumerate(test_idx):
            if idx in emo_indices and probs[j] < prob_threshold:
                misprediction_counts[idx] += 1

    # Flag emo samples mispredicted in at least half the folds
    flagged = set()
    for idx, count in misprediction_counts.items():
        if count >= n_splits // 2:
            flagged.add(paths[idx])
    return flagged


def main():
    print("Emo Dataset Verification")
    print("=" * 50)
    print("Loading audio and extracting features...")
    df, feature_cols = get_feature_matrix(RAW_AUDIO_DIR)
    emo_count = (df["label"] == "emo").sum()
    print(f"  Emo: {emo_count}, Not emo: {len(df) - emo_count}")

    print("\n1. Outlier detection (emo samples that don't match emo cluster)...")
    outlier_paths = flag_outliers(df, feature_cols, n=40)
    print(f"   Flagged {len(outlier_paths)} potential mislabels (outliers)")

    print("\n2. Cross-validation (emo predicted Not Emo in multiple folds)...")
    cv_paths = flag_cv_mispredictions(df, feature_cols, n_splits=5, prob_threshold=0.25)
    print(f"   Flagged {len(cv_paths)} potential mislabels (CV)")

    # Union of both methods
    all_flagged = sorted(outlier_paths | cv_paths)

    report_path = RAW_AUDIO_DIR.parent / "audio_verification_report.txt"
    with open(report_path, "w") as f:
        f.write("Emo samples flagged for manual verification (listen to these)\n")
        f.write("=" * 60 + "\n")
        f.write(f"Total flagged: {len(all_flagged)} of {emo_count} emo files\n\n")
        for p in all_flagged:
            f.write(f"{p}\n")

    print(f"\nReport written to {report_path}")
    print(f"Please manually listen to the {len(all_flagged)} flagged files.")
    print("Move any confirmed non-emo files from emo/ to not_emo/ and retrain.")


if __name__ == "__main__":
    main()
