"""Configuration for the emo music classifier."""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RAW_AUDIO_DIR = DATA_DIR / "audio"
EMO_AUDIO_DIR = RAW_AUDIO_DIR / "emo"
NOT_EMO_AUDIO_DIR = RAW_AUDIO_DIR / "not_emo"
SPOTIFY_PREVIEW_DIR = RAW_AUDIO_DIR / "spotify_previews"

# Create directories
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
RAW_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
EMO_AUDIO_DIR.mkdir(exist_ok=True)
NOT_EMO_AUDIO_DIR.mkdir(exist_ok=True)
(SPOTIFY_PREVIEW_DIR / "emo").mkdir(parents=True, exist_ok=True)
(SPOTIFY_PREVIEW_DIR / "not_emo").mkdir(parents=True, exist_ok=True)

# Spotify config
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID", "")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET", "")

# Genres for dataset building
EMO_GENRES = ["emo", "emo-pop", "pop-punk", "post-hardcore", "screamo"]
# Curated playlists = human-verified real emo (Spotify official, Emo Nite, etc.)
EMO_PLAYLIST_IDS = [
    "37i9dQZF1DX9wa6XirBPv8",  # Emo Forever (Spotify)
    "4eC7Sa1Xcy33lKn53gfiZb",  # The Sound of Emo
]
NON_EMO_GENRES = [
    "pop", "rock", "hip-hop", "electronic", "country", "jazz",
    "classical", "r-n-b", "reggae", "metal", "indie", "dance",
    "dance pop", "disco", "soul", "funk",  # distinctly non-emo
]
# Artist genres that almost never overlap with emo (for Spotify URL override)
# Match if artist genre equals or contains any of these (e.g. "classic soul" contains "soul")
DEF_NOT_EMO_GENRES = frozenset({
    "disco", "funk", "soul", "motown", "gospel", "reggaeton",
    "bossa nova", "latin", "samba", "reggae", "dancehall",
    "country", "bluegrass", "honky-tonk", "classical", "opera",
    "r-n-b", "r&b", "rhythm and blues",
})

# Spotify audio features used for classification (when audio-features API is available)
SPOTIFY_FEATURES = [
    "danceability", "energy", "key", "loudness", "mode",
    "speechiness", "acousticness", "instrumentalness", "liveness",
    "valence", "tempo", "duration_ms"
]
# Fallback features from sp.track() when audio-features is deprecated (403)
SPOTIFY_FALLBACK_FEATURES = ["duration_ms", "popularity", "explicit"]
# All feature names the Spotify classifier may use
SPOTIFY_ALL_FEATURES = list(dict.fromkeys(SPOTIFY_FEATURES + SPOTIFY_FALLBACK_FEATURES))

# Librosa features (for audio-based classification)
LIBROSA_FEATURE_NAMES = [
    "mfcc_mean", "mfcc_std", "chroma_mean", "chroma_std",
    "spectral_centroid_mean", "spectral_rolloff_mean",
    "spectral_bandwidth_mean", "zero_crossing_rate_mean",
    "rms_mean", "tempo",
]
