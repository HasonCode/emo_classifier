"""CLI for predicting if a song is emo."""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def predict_spotify(track_url: str):
    """Predict using Spotify features (fetches from Spotify API)."""
    try:
        import spotipy
        from spotipy.oauth2 import SpotifyClientCredentials
        from config import SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET
        from classifier_spotify import predict
    except ImportError as e:
        print(f"Missing dependency: {e}")
        sys.exit(1)

    if not SPOTIFY_CLIENT_ID or not SPOTIFY_CLIENT_SECRET:
        print("Set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET in .env")
        sys.exit(1)

    # Parse track ID from URL
    # e.g. https://open.spotify.com/track/3n3Ppam7vgaVa1iaRUc9Lp
    parts = track_url.strip().split("/")
    track_id = None
    for i, p in enumerate(parts):
        if p == "track" and i + 1 < len(parts):
            track_id = parts[i + 1].split("?")[0]
            break
    if not track_id:
        print("Invalid Spotify track URL")
        sys.exit(1)

    auth = SpotifyClientCredentials(
        client_id=SPOTIFY_CLIENT_ID,
        client_secret=SPOTIFY_CLIENT_SECRET,
    )
    sp = spotipy.Spotify(auth_manager=auth)
    features = sp.audio_features([track_id])[0]
    if not features:
        print("Could not fetch track features")
        sys.exit(1)

    track = sp.track(track_id)
    name = track["name"]
    artist = track["artists"][0]["name"]

    label, prob = predict(features)
    print(f"\n  {name} - {artist}")
    print(f"  Prediction: {label} ({prob:.1%} emo)")
    return label, prob


def predict_audio(audio_path: str):
    """Predict using raw audio file (path or '-' for stdin)."""
    from classifier_audio import predict

    if audio_path == "-":
        data = sys.stdin.buffer.read()
        if not data:
            print("No audio data received from stdin")
            sys.exit(1)
        label, prob = predict(data)
        print(f"\n  (stdin)")
    else:
        path = Path(audio_path)
        if not path.exists():
            print(f"File not found: {audio_path}")
            sys.exit(1)
        label, prob = predict(path)
        print(f"\n  {path.name}")
    print(f"  Prediction: {label} ({prob:.1%} emo)")
    return label, prob


def main():
    parser = argparse.ArgumentParser(
        description="Classify if a song is emo or not emo"
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # Spotify mode
    spotify_p = subparsers.add_parser("spotify", help="Classify via Spotify track URL")
    spotify_p.add_argument(
        "url",
        help="Spotify track URL (e.g. https://open.spotify.com/track/...)",
    )

    # Audio mode
    audio_p = subparsers.add_parser("audio", help="Classify from local audio file")
    audio_p.add_argument(
        "path",
        help="Path to audio file (.mp3, .wav, .flac, etc.) or '-' for stdin",
    )

    args = parser.parse_args()
    if args.mode == "spotify":
        predict_spotify(args.url)
    else:
        predict_audio(args.path)


if __name__ == "__main__":
    main()
