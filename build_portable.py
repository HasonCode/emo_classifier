#!/usr/bin/env python3
"""
Create a portable zip for distribution (no PyInstaller required).
Use when PyInstaller isn't available (e.g. Python 3.14+) or for quick distribution.

Usage: python build_portable.py
Output: dist/EmoClassifier-Portable.zip

Users: unzip, run run.bat (Windows) or ./run.sh (Mac/Linux)
"""

import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
DIST_DIR = PROJECT_ROOT / "dist"
OUTPUT_NAME = "EmoClassifier-Portable"


def main():
    print("Building portable Emo Classifier package...")
    out_path = DIST_DIR / OUTPUT_NAME
    if out_path.exists():
        shutil.rmtree(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    # Copy application files
    to_copy = [
        ("standalone_app.py", "standalone_app.py"),
        ("api.py", "api.py"),
        ("config.py", "config.py"),
        ("classifier_audio.py", "classifier_audio.py"),
        ("requirements.txt", "requirements.txt"),
    ]
    for src, dst in to_copy:
        src_path = PROJECT_ROOT / src
        if src_path.exists():
            shutil.copy2(src_path, out_path / dst)

    # Copy directories
    for dirname in ["static", "models"]:
        src = PROJECT_ROOT / dirname
        if src.exists():
            shutil.copytree(src, out_path / dirname, dirs_exist_ok=True)

    # Create run script for Windows
    run_bat = out_path / "run.bat"
    run_bat.write_text("""@echo off
echo Emo Music Classifier - Portable
echo.
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)
call venv\\Scripts\\activate.bat
pip install -q -r requirements.txt 2>nul
echo.
echo Starting app - browser will open...
echo Press Ctrl+C to exit.
python standalone_app.py
pause
""", encoding="utf-8")

    # Create run script for Unix
    run_sh = out_path / "run.sh"
    run_sh.write_text("""#!/bin/sh
echo "Emo Music Classifier - Portable"
echo
if [ ! -d venv ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi
. venv/bin/activate
pip install -q -r requirements.txt 2>/dev/null
echo
echo "Starting app - browser will open..."
echo "Press Ctrl+C to exit."
python standalone_app.py
""", encoding="utf-8")
    run_sh.chmod(0o755)

    # README for the package
    readme = out_path / "README.txt"
    readme.write_text("""Emo Music Classifier - Portable

1. Extract this folder anywhere.

2. Run the app:
   - Windows: Double-click run.bat
   - Mac/Linux: ./run.sh (or: chmod +x run.sh && ./run.sh)

3. Your browser will open. Upload an MP3, WAV, FLAC, OGG, or M4A to get
   emo/not emo prediction with confidence.

Requirements: Python 3.10+ must be installed on your system.
If not installed: https://www.python.org/downloads/
""", encoding="utf-8")

    # Create zip
    zip_path = DIST_DIR / f"{OUTPUT_NAME}.zip"
    if zip_path.exists():
        zip_path.unlink()
    shutil.make_archive(str(DIST_DIR / OUTPUT_NAME), "zip", DIST_DIR, OUTPUT_NAME)
    print(f"Created: {zip_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
