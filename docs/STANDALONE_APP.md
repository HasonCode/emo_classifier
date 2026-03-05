# Emo Classifier Standalone App

A downloadable app that lets users upload audio files and get emo/not emo predictions with confidence scores. Runs locally—no account or internet required for audio classification.

## Quick Start (Developers)

```bash
# Install dependencies
poetry install

# Train the model first (if not already done)
poetry run python -m classifier_audio

# Run the standalone app
poetry run python standalone_app.py
```

Opens http://localhost:8765 in your browser. Upload an MP3, WAV, FLAC, OGG, or M4A file to get the prediction and confidence.

## Building a Distributable

Package the app into a folder that users can download and run without installing Python:

```bash
poetry install  # Ensure pyinstaller is installed
poetry run pyinstaller emo_classifier.spec
```

Output: `dist/EmoClassifier/` containing:
- `EmoClassifier.exe` (Windows) or `EmoClassifier` (macOS/Linux)
- Required libraries and the trained model

### Distributing

1. Zip the entire `dist/EmoClassifier` folder
2. Share the zip (e.g. via GitHub Releases)
3. Users: unzip, run `EmoClassifier.exe` (or `./EmoClassifier` on Mac/Linux)
4. Browser opens automatically; upload audio to classify

### Requirements for Build

- A trained model in `models/` (run `poetry run python -m classifier_audio` first)
- PyInstaller: `poetry add --group dev pyinstaller`

### Platform Notes

- Build on each target OS: Windows build produces `.exe`, macOS produces Unix executable, etc.
- The app bundles the audio model; no Spotify or internet required
