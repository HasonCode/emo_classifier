"""
HTTP API for emo music classification.
Supports multipart/form-data file upload for audio classification.

Run: uvicorn api:app --reload
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(
    title="Emo Music Classifier API",
    description="Classify whether a song is emo or not via audio file upload.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".webm"}


@app.get("/")
async def index():
    """Serve the upload UI."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"message": "Emo Classifier API. POST /classify with a file to classify."}


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class PredictionResponse(BaseModel):
    label: str
    probability: float
    is_emo: bool


class SpotifyClassifyRequest(BaseModel):
    url: str


def _check_extension(filename: str) -> bool:
    ext = Path(filename).suffix.lower()
    return ext in ALLOWED_EXTENSIONS


def _calibrate_probability(prob: float) -> float:
    """Dampen extreme probabilities to reduce overconfidence."""
    return max(0.12, min(0.88, prob))


@app.post("/classify", response_model=PredictionResponse)
async def classify_audio(file: UploadFile = File(...)):
    """
    Upload an audio file and get emo/not emo classification.
    Prefers audio model when available; else uses Spotify model with approximated features.
    Accepts: MP3, WAV, FLAC, OGG, M4A, WebM
    """
    if not _check_extension(file.filename or ""):
        raise HTTPException(
            400,
            detail=f"Unsupported format. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    try:
        data = await file.read()
    except Exception as e:
        raise HTTPException(422, detail=f"Could not read upload: {e}")

    if not data:
        raise HTTPException(422, detail="Empty file")

    try:
        # Prefer audio model (trained on real audio features) when available
        from config import MODELS_DIR
        audio_model = MODELS_DIR / "emo_classifier_audio.joblib"
        if audio_model.exists():
            from classifier_audio import predict as audio_predict
            label, prob = audio_predict(data)
        else:
            from spotify_from_audio import extract_spotify_like_features
            from classifier_spotify import predict as spotify_predict
            features = extract_spotify_like_features(data)
            label, prob = spotify_predict(features)
    except FileNotFoundError as e:
        raise HTTPException(503, detail=str(e))
    except Exception as e:
        raise HTTPException(422, detail=f"Could not process audio: {e}")

    prob = _calibrate_probability(prob)
    return PredictionResponse(
        label=label,
        probability=round(prob, 4),
        is_emo=(label == "Emo"),
    )


@app.post("/classify/spotify", response_model=PredictionResponse)
async def classify_spotify_url(body: SpotifyClassifyRequest = Body(...)):
    """
    Classify a song by Spotify track URL. Uses the Spotify model.
    """
    url = (body.url or "").strip()
    if not url:
        raise HTTPException(400, detail="Spotify URL required")

    parts = url.split("/")
    track_id = None
    for i, p in enumerate(parts):
        if p == "track" and i + 1 < len(parts):
            track_id = parts[i + 1].split("?")[0]
            break
    if not track_id:
        raise HTTPException(400, detail="Invalid Spotify track URL")

    try:
        import spotipy
        from spotipy.oauth2 import SpotifyClientCredentials
        from config import SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET
        from classifier_spotify import predict as spotify_predict
    except FileNotFoundError as e:
        raise HTTPException(503, detail=str(e))
    except ImportError as e:
        raise HTTPException(503, detail=f"Dependency error: {e}")

    if not SPOTIFY_CLIENT_ID or not SPOTIFY_CLIENT_SECRET:
        raise HTTPException(503, detail="Spotify credentials not configured")

    auth = SpotifyClientCredentials(
        client_id=SPOTIFY_CLIENT_ID,
        client_secret=SPOTIFY_CLIENT_SECRET,
    )
    sp = spotipy.Spotify(auth_manager=auth)
    features = None
    try:
        feats_list = sp.audio_features([track_id])
        if feats_list and feats_list[0]:
            features = feats_list[0]
    except Exception:
        pass
    track = None
    if not features:
        track = sp.track(track_id)
        features = {
            "id": track_id,
            "duration_ms": track.get("duration_ms", 0),
            "popularity": track.get("popularity", 0),
            "explicit": 1 if track.get("explicit") else 0,
        }
    else:
        track = sp.track(track_id)

    # Genre-based override: if artist is in "definitely not emo" genres, require very high confidence
    from config import DEF_NOT_EMO_GENRES
    def_not_emo = False
    if track and track.get("artists"):
        try:
            art = sp.artist(track["artists"][0]["id"])
            genre_str = " ".join(art.get("genres") or []).lower()
            def_not_emo = any(g in genre_str for g in DEF_NOT_EMO_GENRES)
        except Exception:
            pass

    try:
        label, prob = spotify_predict(features)
    except Exception as e:
        raise HTTPException(422, detail=f"Classification failed: {e}")
    if def_not_emo and label == "Emo" and prob < 0.92:
        label, prob = "Not Emo", 1.0 - prob

    prob = _calibrate_probability(prob)
    return PredictionResponse(
        label=label,
        probability=round(prob, 4),
        is_emo=(label == "Emo"),
    )


@app.get("/health")
async def health():
    return {"status": "ok"}
