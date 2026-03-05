"""Audio-based emo music classifier using Librosa features."""

import io
import json
import tempfile
from pathlib import Path
from typing import BinaryIO, Union

import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import joblib

from config import DATA_DIR, MODELS_DIR, RAW_AUDIO_DIR


AudioSource = Union[str, Path, bytes, BinaryIO]


def _suffix_from_magic(data: bytes) -> str:
    """Infer file extension from magic bytes for librosa format detection."""
    if data[:3] == b"ID3" or (len(data) >= 2 and data[:2] == b"\xff\xfb"):
        return ".mp3"
    if data[:4] == b"RIFF":
        return ".wav"
    if data[:4] == b"fLaC":
        return ".flac"
    if data[:4] == b"OggS":
        return ".ogg"
    if len(data) >= 8 and data[4:8] == b"ftyp":
        return ".m4a"
    return ".mp3"  # fallback


def _resolve_audio_source(source: AudioSource, sr: int):
    """
    Load audio from path, file-like object, or bytes.
    Returns (y, sr) for use with feature extraction.
    Uses temp file for bytes/file-like to support all formats (incl. MP3).
    """
    if isinstance(source, (str, Path)):
        return librosa.load(source, sr=sr, mono=True, duration=30)
    # Bytes or file-like: write to temp file for full format support (incl. MP3)
    if isinstance(source, bytes):
        data = source
    else:
        if hasattr(source, "seek"):
            source.seek(0)
        data = source.read() if hasattr(source, "read") else bytes(source)
    suffix = _suffix_from_magic(data)
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
        tmp.write(data)
        tmp.flush()
        return librosa.load(tmp.name, sr=sr, mono=True, duration=30)


def _scalar(x) -> float:
    """Convert numpy scalar or 0-d array to Python float."""
    arr = np.asarray(x)
    return float(arr.flat[0]) if arr.size > 0 else 0.0


def extract_audio_features(audio_source: AudioSource, sr: int = 22050) -> dict:
    """Extract features from an audio file using Librosa.
    Accepts: file path (str/Path), bytes, or file-like object (e.g. upload stream).
    """
    y, sr = _resolve_audio_source(audio_source, sr)
    if y is None or len(y) < sr:  # need at least 1 second
        raise ValueError("Audio too short or empty")

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)[0]
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    return {
        "mfcc_mean": _scalar(np.mean(mfcc)),
        "mfcc_std": _scalar(np.std(mfcc)),
        "chroma_mean": _scalar(np.mean(chroma)),
        "chroma_std": _scalar(np.std(chroma)),
        "spectral_centroid_mean": _scalar(np.mean(spectral_centroid)),
        "spectral_rolloff_mean": _scalar(np.mean(spectral_rolloff)),
        "spectral_bandwidth_mean": _scalar(np.mean(spectral_bandwidth)),
        "zero_crossing_rate_mean": _scalar(np.mean(zcr)),
        "rms_mean": _scalar(np.mean(rms)),
        "tempo": _scalar(tempo),
    }


def scan_audio_directory(audio_dir: Path) -> pd.DataFrame:
    """
    Scan directory for audio files.
    Expected structure:
      audio_dir/
        emo/
          song1.mp3
          song2.wav
        not_emo/
          song1.mp3
    """
    rows = []
    for label, subdir in [("emo", "emo"), ("not_emo", "not_emo")]:
        subpath = audio_dir / subdir
        if not subpath.exists():
            continue
        for ext in ["*.mp3", "*.wav", "*.flac", "*.ogg", "*.m4a"]:
            for f in subpath.glob(ext):
                rows.append({"path": str(f), "label": label})

    return pd.DataFrame(rows)


def build_audio_dataset(audio_dir: Path = None) -> pd.DataFrame:
    """Build dataset from audio directory."""
    audio_dir = audio_dir or RAW_AUDIO_DIR
    file_df = scan_audio_directory(audio_dir)
    if file_df.empty:
        raise FileNotFoundError(
            f"No audio files found. Create directories:\n"
            f"  {audio_dir}/emo/   - Put emo songs here\n"
            f"  {audio_dir}/not_emo/ - Put non-emo songs here"
        )

    rows = []
    total = len(file_df)
    for i, (_, row) in enumerate(file_df.iterrows()):
        try:
            feats = extract_audio_features(row["path"])
            feats["is_emo"] = 1 if row["label"] == "emo" else 0
            feats["path"] = row["path"]
            rows.append(feats)
        except Exception as e:
            print(f"Skipping {row['path']}: {e}")
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{total} files...")

    if not rows:
        raise ValueError(
            "No audio files could be processed. Check file formats (MP3, WAV, FLAC) and try again."
        )
    print(f"Extracted features from {len(rows)} files")
    return pd.DataFrame(rows)


def train_classifier(df: pd.DataFrame = None, save: bool = True):
    """Train the audio-based emo classifier."""
    if df is None:
        df = build_audio_dataset()

    feature_cols = [
        "mfcc_mean", "mfcc_std", "chroma_mean", "chroma_std",
        "spectral_centroid_mean", "spectral_rolloff_mean",
        "spectral_bandwidth_mean", "zero_crossing_rate_mean",
        "rms_mean", "tempo",
    ]
    X = df[feature_cols].fillna(0)
    y = df["is_emo"]

    if len(np.unique(y)) < 2:
        raise ValueError("Need both emo and not_emo samples to train")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Hyperparameter tuning for better accuracy
    param_grid = {
        "n_estimators": [100, 150, 200],
        "max_depth": [8, 12, 16],
        "min_samples_leaf": [1, 2, 4],
    }
    base_rf = RandomForestClassifier(random_state=42, class_weight="balanced")
    grid = GridSearchCV(
        base_rf, param_grid, cv=4, scoring="accuracy", n_jobs=-1, verbose=0
    )
    grid.fit(X_train_scaled, y_train)
    clf = grid.best_estimator_
    print(f"Best params: {grid.best_params_}")

    y_pred = clf.predict(X_test_scaled)
    print("Audio Emo Classifier - Performance")
    print("=" * 40)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")
    print(classification_report(y_test, y_pred, target_names=["Not Emo", "Emo"]))

    if save:
        model_path = MODELS_DIR / "emo_classifier_audio.joblib"
        scaler_path = MODELS_DIR / "emo_scaler_audio.joblib"
        meta_path = MODELS_DIR / "emo_classifier_audio_meta.json"
        joblib.dump(clf, model_path)
        joblib.dump(scaler, scaler_path)
        with open(meta_path, "w") as f:
            json.dump({"feature_columns": feature_cols}, f, indent=2)
        print(f"\nModel saved to {model_path}")

    return clf, scaler, feature_cols


def load_classifier():
    """Load trained audio classifier."""
    model_path = MODELS_DIR / "emo_classifier_audio.joblib"
    scaler_path = MODELS_DIR / "emo_scaler_audio.joblib"
    meta_path = MODELS_DIR / "emo_classifier_audio_meta.json"

    if not model_path.exists():
        raise FileNotFoundError(
            "Audio model not found. Add audio files to data/audio/emo/ and "
            "data/audio/not_emo/ then run: python -m classifier_audio"
        )

    clf = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    with open(meta_path) as f:
        meta = json.load(f)
    return clf, scaler, meta["feature_columns"]


def predict(audio_source: AudioSource) -> tuple[str, float]:
    """
    Predict if an audio file is emo.
    Accepts: path (str/Path), bytes, or file-like object (e.g. uploaded file).
    Returns (label, probability) e.g. ("Emo", 0.78)
    """
    clf, scaler, feature_cols = load_classifier()
    feats = extract_audio_features(audio_source)
    row = [feats.get(c, 0) for c in feature_cols]
    X = np.array([row])
    X_scaled = scaler.transform(X)
    proba = clf.predict_proba(X_scaled)[0]
    pred = clf.predict(X_scaled)[0]
    label = "Emo" if pred == 1 else "Not Emo"
    emo_prob = _scalar(proba[1])
    return label, emo_prob


def main():
    """Train the audio classifier."""
    import argparse
    from config import SPOTIFY_PREVIEW_DIR
    parser = argparse.ArgumentParser(description="Train emo classifier from audio files")
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=None,
        help="Audio directory with emo/ and not_emo/ subdirs (default: data/audio). Use data/audio/spotify_previews for Spotify previews.",
    )
    args = parser.parse_args()
    audio_dir = args.audio_dir
    if audio_dir:
        path = Path(audio_dir)
        if not path.is_absolute():
            path = path.resolve()
        df = build_audio_dataset(path)
    else:
        df = None
    train_classifier(df=df)


if __name__ == "__main__":
    main()
