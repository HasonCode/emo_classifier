"""
Extract Spotify-like audio features from raw audio files.
Used to run the Spotify-trained classifier on uploaded MP3/audio when
Spotify API features are not available.
"""

from pathlib import Path
from typing import BinaryIO, Union

import librosa
import numpy as np

from classifier_audio import _resolve_audio_source
from config import SPOTIFY_ALL_FEATURES

AudioSource = Union[str, Path, bytes, BinaryIO]


def _scalar(x) -> float:
    """Convert numpy array/scalar to Python float."""
    arr = np.asarray(x)
    return float(arr.flatten()[0]) if arr.size > 0 else 0.0


def extract_spotify_like_features(audio_source: AudioSource, sr: int = 22050) -> dict:
    """
    Extract approximations of Spotify audio features from raw audio.
    Returns a dict with keys matching SPOTIFY_ALL_FEATURES for use with
    classifier_spotify.predict(). Approximations are heuristic; accuracy
    varies vs. true Spotify features.
    """
    y, sr = _resolve_audio_source(audio_source, sr)

    # Duration
    duration_ms = _scalar(len(y) / sr * 1000)

    # Tempo (BPM) - librosa 0.10+ may return array
    tempo_raw, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo = _scalar(np.clip(tempo_raw, 50, 200))

    # RMS energy → energy (0-1) and loudness (dB)
    # Use conservative scaling: typical RMS 0.01-0.15 maps to energy 0.2-0.85
    rms = librosa.feature.rms(y=y)[0]
    rms_mean = _scalar(np.mean(rms))
    energy = _scalar(np.clip(0.2 + 4.5 * rms_mean, 0.2, 0.9))
    loudness = _scalar(np.clip(20 * np.log10(rms_mean + 1e-10), -60, 0))

    # Spectral features
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    spectral_flatness = librosa.feature.spectral_flatness(y=y)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)

    # Acousticness: high spectral contrast + low flatness ≈ more acoustic
    flatness_mean = _scalar(np.mean(spectral_flatness))
    acousticness = _scalar(np.clip(0.5 + (0.5 - flatness_mean) * 0.5, 0, 1))

    # Chroma for key (0-11)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    key = int(np.argmax(chroma_mean))

    # Mode: 1=major, 0=minor (simplified: use chroma profile)
    try:
        minor_profile = np.array([1.0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0])
        major_profile = np.array([1.0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0])
        if np.std(chroma_mean) > 1e-6:
            cm = np.nan_to_num(chroma_mean, nan=0.0)
            minor_corr = np.corrcoef(cm, minor_profile)[0, 1]
            major_corr = np.corrcoef(cm, major_profile)[0, 1]
            mode = 0 if (minor_corr > major_corr and not np.isnan(minor_corr)) else 1
        else:
            mode = 1
    except Exception:
        mode = 1

    # Danceability: tempogram consistency + moderate tempo
    try:
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr)
        tempogram_std = _scalar(np.std(tempogram))
        danceability = _scalar(np.clip(0.3 + 0.4 * (1 - np.clip(tempogram_std / 50, 0, 1)), 0, 1))
    except Exception:
        danceability = 0.5

    # Speechiness: we can't detect speech in music; use low default
    speechiness = 0.05

    # Instrumentalness: cannot reliably detect; use mid value
    instrumentalness = 0.3

    # Liveness: cannot detect audience; use low
    liveness = 0.08

    # Valence (positivity): cannot infer from audio; use mid
    valence = 0.5

    # popularity, explicit: unknown for local files
    # Vary popularity slightly by audio complexity to avoid constant bias
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    bw_mean = _scalar(np.mean(spectral_bandwidth))
    popularity = int(np.clip(40 + (bw_mean / 2000) * 30, 25, 75))  # 25-75 range
    explicit = 0

    features = {
        "duration_ms": duration_ms,
        "tempo": tempo,
        "energy": energy,
        "loudness": loudness,
        "acousticness": acousticness,
        "key": key,
        "mode": mode,
        "danceability": danceability,
        "speechiness": speechiness,
        "instrumentalness": instrumentalness,
        "liveness": liveness,
        "valence": valence,
        "popularity": popularity,
        "explicit": explicit,
    }
    # Ensure all values are Python natives (not numpy)
    result = {}
    for k, v in features.items():
        if k not in SPOTIFY_ALL_FEATURES:
            continue
        if hasattr(v, "item"):
            result[k] = v.item()
        elif isinstance(v, (int, float)):
            result[k] = v
        else:
            result[k] = _scalar(v)
    return result
