# PyInstaller spec for Emo Classifier Standalone App
# Build: poetry run pyinstaller emo_classifier.spec
# Output: dist/EmoClassifier/ with EmoClassifier.exe + dependencies
# Distribute: zip the dist/EmoClassifier folder; users run EmoClassifier.exe

from pathlib import Path

project_root = Path(SPECPATH)

a = Analysis(
    ["standalone_app.py"],
    pathex=[str(project_root)],
    binaries=[],
    datas=[
        (str(project_root / "static"), "static"),
        (str(project_root / "models"), "models"),
    ],
    hiddenimports=[
        "api",
        "config",
        "classifier_audio",
        "classifier_spotify",
        "spotify_from_audio",
        "uvicorn.logging",
        "uvicorn.loops",
        "uvicorn.loops.auto",
        "uvicorn.protocols",
        "uvicorn.protocols.http",
        "uvicorn.protocols.http.auto",
        "uvicorn.protocols.websockets",
        "uvicorn.protocols.websockets.auto",
        "uvicorn.lifespan",
        "uvicorn.lifespan.on",
        "fastapi",
        "starlette.routing",
        "starlette.staticfiles",
        "sklearn.utils._cython_blas",
        "sklearn.neighbors._typedefs",
        "sklearn.neighbors._quad_tree",
        "sklearn.tree._utils",
        "sklearn.tree._splitter",
        "joblib",
        "numpy",
        "librosa",
        "soundfile",
        "scipy.fft",
        "sklearn.ensemble._forest",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "matplotlib",
        "tkinter",
        "PyQt5",
        "PyQt6",
        "pytest",
        "IPython",
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="EmoClassifier",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="EmoClassifier",
)
