"""
HTTP API for emo music classification.
Audio upload only — uses the trained audio model (Librosa features).

Run: uvicorn api:app --reload
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, File, UploadFile, HTTPException
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
    return {"message": "Emo Classifier API. POST /classify with a file."}


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class PredictionResponse(BaseModel):
    label: str
    probability: float
    is_emo: bool


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
    Uses the trained audio model (Librosa: MFCC, chroma, spectral, tempo, etc.).
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
        from classifier_audio import predict as audio_predict

        label, prob = audio_predict(data)
    except FileNotFoundError as e:
        raise HTTPException(
            503,
            detail="Audio model not found. Train with: poetry run python -m classifier_audio",
        )
    except Exception as e:
        raise HTTPException(422, detail=f"Could not process audio: {e}")

    prob = _calibrate_probability(prob)
    return PredictionResponse(
        label=label,
        probability=round(prob, 4),
        is_emo=(label == "Emo"),
    )


@app.get("/health")
async def health():
    return {"status": "ok"}
