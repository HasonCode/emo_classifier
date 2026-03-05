# Emo Music Classifier 🎸

A machine learning model to classify whether a song is **emo** or **not emo**. Supports two approaches:

1. **Spotify-based** – Uses Spotify's audio features (energy, valence, etc.) – good when you have Spotify links
2. **Audio-based** – Analyzes raw audio files with Librosa (MFCC, chroma, etc.) – works with any audio file

## Quick Start

### 1. Install dependencies

**Using Poetry (recommended):**
```bash
poetry install
poetry run python -m spotify_data  # use poetry run for commands
```

**Or using pip:**
```bash
pip install -r requirements.txt
```

### 2. Spotify classifier (recommended)

**Get Spotify API credentials** (free):
1. Go to [Spotify Developer Dashboard](https://developer.spotify.com/dashboard)
2. Create an app, copy Client ID and Client Secret
3. Create `.env`:
   ```
   SPOTIFY_CLIENT_ID=your_id
   SPOTIFY_CLIENT_SECRET=your_secret
   ```

**Collect data and train:**
```bash
# Fetch emo + non-emo tracks from Spotify
python -m spotify_data

# Train the classifier
python -m classifier_spotify
```

**Classify a song:**
```bash
python predict.py spotify "https://open.spotify.com/track/3n3Ppam7vgaVa1iaRUc9Lp"
```

### 3. Audio-based classifier

If you have local audio files:

1. Organize your files:
   ```
   data/audio/
     emo/        # Put emo songs here (.mp3, .wav, .flac)
     not_emo/    # Put non-emo songs here
   ```

2. Train:
   ```bash
   python -m classifier_audio
   ```

3. Classify a file:
   ```bash
   python predict.py audio path/to/song.mp3
   ```

### 4. Spotify 30-second previews (training from previews)

Use Spotify's preview URLs to fetch training data without local audio files:

```bash
# 1. Fetch previews (requires Spotify credentials in .env)
poetry run python -m spotify_preview_fetch

# 2. Train on the downloaded previews
poetry run python -m classifier_audio --audio-dir data/audio/spotify_previews
```

Previews are saved to `data/audio/spotify_previews/emo/` and `.../not_emo/`. Note: not all tracks have preview_url available.

### 5. Web app (upload UI)

Start the app and open http://localhost:8000 (or 8001 if 8000 is in use) in your browser to upload MP3s and get classifications:

```bash
poetry run uvicorn api:app --reload --port 8000
```

- Dark-themed upload page with drag-and-drop
- Click or drop an MP3 to classify
- Shows **Emo** or **Not Emo** with confidence %

**API endpoint** `POST /classify` (for curl/Swagger):

```bash
curl -X POST -F "file=@song.mp3" http://localhost:8000/classify
```

- Swagger docs: http://localhost:8000/docs
- Accepts: MP3, WAV, FLAC, OGG, M4A, WebM

### 6. Standalone app (downloadable)

Run a local app that opens in your browser—upload audio and get prediction + confidence:

```bash
poetry run python standalone_app.py
```

Opens http://localhost:8765. To build a **distributable** (no Python required):

```bash
poetry run pyinstaller emo_classifier.spec
```

Zip `dist/EmoClassifier/` and share it. Users run `EmoClassifier.exe` (Windows) or `./EmoClassifier` (Mac/Linux); the browser opens automatically. See [docs/STANDALONE_APP.md](docs/STANDALONE_APP.md).

## Project Structure

```
classifier/
├── config.py           # Paths, genres, feature config
├── spotify_data.py     # Fetch training data from Spotify
├── classifier_spotify.py  # Spotify-based model
├── classifier_audio.py    # Audio-based model (Librosa)
├── predict.py            # CLI for predictions
├── api.py                # HTTP API + upload web app
├── static/
│   └── index.html        # Upload UI
├── data/               # Datasets
├── models/             # Saved models
└── requirements.txt
```

## How It Works

- **Spotify**: Searches for emo-related genres (emo, emo-pop, pop-punk, post-hardcore, screamo) vs others (pop, rock, metal, etc.), extracts acoustic features, trains a Random Forest.
- **Audio**: Extracts MFCC, chroma, spectral centroid, tempo from raw audio, trains a Random Forest.

Both models use scikit-learn's `RandomForestClassifier` with standardized features.
