"""
Fetch Spotify 30-second previews and use them for audio-based training.

Uses Spotify's preview_url (30-second MP3 clips) to build a training dataset.
Run: poetry run python -m spotify_preview_fetch
Then train: poetry run python -m classifier_audio --audio-dir data/audio/spotify_previews

Note: preview_url can be null for some tracks. Spotify policy restricts bulk download;
use for personal/research purposes only.
"""

import re
import time
from pathlib import Path

import requests
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

from config import (
    EMO_GENRES,
    NON_EMO_GENRES,
    SPOTIFY_CLIENT_ID,
    SPOTIFY_CLIENT_SECRET,
    SPOTIFY_PREVIEW_DIR,
)

PREVIEW_DELAY = 0.3
BATCH_DELAY = 0.5
SEARCH_DELAY = 0.6
SEARCH_LIMIT = 10


def get_spotify_client():
    """Create authenticated Spotify client."""
    if not SPOTIFY_CLIENT_ID or not SPOTIFY_CLIENT_SECRET:
        raise ValueError(
            "Spotify credentials required. Set SPOTIFY_CLIENT_ID and "
            "SPOTIFY_CLIENT_SECRET in .env"
        )
    auth = SpotifyClientCredentials(
        client_id=SPOTIFY_CLIENT_ID,
        client_secret=SPOTIFY_CLIENT_SECRET,
    )
    return spotipy.Spotify(auth_manager=auth)


def fetch_tracks_by_genre(sp, genre: str, limit: int) -> list[dict]:
    """Fetch tracks by genre search. Includes preview_url from search response.
    Avoids sp.tracks() which returns 403 for new Spotify apps.
    """
    tracks = []
    offset = 0
    while len(tracks) < limit:
        try:
            results = sp.search(
                q=f"genre:{genre}",
                type="track",
                limit=min(SEARCH_LIMIT, limit - len(tracks)),
                offset=offset,
            )
            items = results["tracks"]["items"]
            for item in items:
                if item["id"] not in [t["id"] for t in tracks]:
                    preview = item.get("preview_url")
                    if preview and isinstance(preview, str) and preview.startswith("http"):
                        tracks.append({
                            "id": item["id"],
                            "name": item.get("name", ""),
                            "artist": item["artists"][0]["name"] if item.get("artists") else "",
                            "preview_url": preview,
                        })
                    if len(tracks) >= limit:
                        break
            if not items:
                break
            offset += SEARCH_LIMIT
            time.sleep(SEARCH_DELAY)
        except Exception as e:
            err = str(e).lower()
            if "429" in err or "rate" in err:
                m = re.search(r"retry[-\s]after[:\s]+(\d+)", str(e), re.I)
                wait = min(int(m.group(1)), 300) if m else 60
                print(f"  Rate limited. Waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"  Genre {genre} error: {e}")
                break
    return tracks[:limit]


PREVIEW_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; EmoClassifier/1.0)",
    "Accept": "audio/mpeg,*/*",
}


def download_preview(url: str) -> bytes | None:
    """Download preview MP3 from URL."""
    try:
        r = requests.get(url, timeout=15, headers=PREVIEW_HEADERS)
        r.raise_for_status()
        data = r.content
        return data if data and len(data) > 500 else None
    except Exception:
        return None


def fetch_and_download_previews(
    per_class: int = 80,
    emo_from_playlists: bool = False,
) -> tuple[int, int]:
    """
    Fetch emo/non-emo tracks from Spotify via genre search, download 30-second previews.
    Uses preview_url from search results (no sp.tracks() - returns 403 for new apps).
    Playlists disabled by default (require user OAuth, client creds get 401).
    """
    sp = get_spotify_client()
    seen = set()
    emo_with_preview = {}
    not_emo_with_preview = {}

    # Emo: genre search only (playlists need user auth → 401)
    per_genre_emo = max(15, per_class // len(EMO_GENRES))
    for genre in EMO_GENRES:
        if len(emo_with_preview) >= per_class:
            break
        print(f"Fetching emo genre: {genre}...")
        for t in fetch_tracks_by_genre(sp, genre, per_genre_emo):
            if t["id"] not in seen:
                seen.add(t["id"])
                emo_with_preview[t["id"]] = t
            if len(emo_with_preview) >= per_class:
                break
        time.sleep(SEARCH_DELAY)

    # Non-emo: genre search
    per_genre = max(10, per_class // len(NON_EMO_GENRES))
    for genre in NON_EMO_GENRES:
        if len(not_emo_with_preview) >= per_class:
            break
        print(f"Fetching non-emo genre: {genre}...")
        for t in fetch_tracks_by_genre(sp, genre, per_genre):
            if t["id"] not in seen:
                seen.add(t["id"])
                not_emo_with_preview[t["id"]] = t
            if len(not_emo_with_preview) >= per_class:
                break
        time.sleep(SEARCH_DELAY)

    # Download and save
    emo_dir = SPOTIFY_PREVIEW_DIR / "emo"
    not_emo_dir = SPOTIFY_PREVIEW_DIR / "not_emo"
    emo_dir.mkdir(parents=True, exist_ok=True)
    not_emo_dir.mkdir(parents=True, exist_ok=True)
    emo_count = 0
    not_emo_count = 0

    total_with_preview = len(emo_with_preview) + len(not_emo_with_preview)
    if total_with_preview == 0:
        print("\nNo tracks with preview_url found.")
        print("  Possible causes: network/proxy blocking Spotify, invalid credentials, or rate limits.")
        print("  Try: unset http_proxy https_proxy (if using proxy) and run again.")
        return 0, 0
    print(f"\nDownloading previews... (emo: {len(emo_with_preview)} with preview, non-emo: {len(not_emo_with_preview)})")
    for tid, info in emo_with_preview.items():
        data = download_preview(info["preview_url"])
        if data and len(data) > 1000:
            out = emo_dir / f"{tid}.mp3"
            out.write_bytes(data)
            emo_count += 1
        time.sleep(PREVIEW_DELAY)
        if emo_count % 20 == 0 and emo_count > 0:
            print(f"  Emo: {emo_count}")

    for tid, info in not_emo_with_preview.items():
        data = download_preview(info["preview_url"])
        if data and len(data) > 1000:
            out = not_emo_dir / f"{tid}.mp3"
            out.write_bytes(data)
            not_emo_count += 1
        time.sleep(PREVIEW_DELAY)
        if not_emo_count % 20 == 0 and not_emo_count > 0:
            print(f"  Non-emo: {not_emo_count}")

    return emo_count, not_emo_count


def main():
    """Fetch Spotify previews and save for training."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    print("Spotify Preview Fetch - 30-second previews for training")
    print("=" * 50)
    try:
        emo_n, not_emo_n = fetch_and_download_previews(per_class=80)
        print(f"\nDone. Saved to {SPOTIFY_PREVIEW_DIR}")
        print(f"  Emo: {emo_n} previews")
        print(f"  Non-emo: {not_emo_n} previews")
        if emo_n or not_emo_n:
            print("\nTrain with:")
            print("  poetry run python -m classifier_audio --audio-dir data/audio/spotify_previews")
        else:
            print("\nNo previews downloaded. Try again later or check Spotify credentials.")
    except ValueError as e:
        print(f"\nError: {e}")
        print("Add SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET to .env")
    except Exception as e:
        print(f"\nError: {e}")
        if "proxy" in str(e).lower() or "connection" in str(e).lower():
            print("  If using a proxy, try: unset http_proxy https_proxy")


if __name__ == "__main__":
    main()
