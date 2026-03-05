"""
Standalone Emo Classifier App
Launch a local server for uploading audio and getting emo/not emo predictions.

Run: python standalone_app.py
Or use the built executable (after: poetry run pyinstaller emo_classifier.spec)

Opens http://localhost:8765 - upload an MP3/WAV/etc to get prediction + confidence.
"""

import sys
import webbrowser
from pathlib import Path

# Support PyInstaller frozen bundle: add _MEIPASS to path
if getattr(sys, "frozen", False):
    _APP_DIR = Path(sys._MEIPASS)
else:
    _APP_DIR = Path(__file__).parent.resolve()

sys.path.insert(0, str(_APP_DIR))


def main():
    import uvicorn

    host = "127.0.0.1"
    port = 8765
    url = f"http://{host}:{port}"

    print("Emo Music Classifier - Standalone App")
    print("=" * 40)
    print(f"Opening {url}")
    print("Upload an audio file to get prediction + confidence.")
    print("Press Ctrl+C to exit.")
    print()

    # Open browser after a short delay (give server time to start)
    def open_browser():
        import time

        time.sleep(1.2)
        webbrowser.open(url)

    import threading

    t = threading.Thread(target=open_browser, daemon=True)
    t.start()

    # Ensure api is importable (helps PyInstaller trace dependencies)
    import api  # noqa: F401

    uvicorn.run(
        api.app,
        host=host,
        port=port,
        log_level="warning",
    )


if __name__ == "__main__":
    main()
