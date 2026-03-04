"""Collect training data from Spotify API for emo vs non-emo classification."""

import json
import time
from pathlib import Path

import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

from config import (
    DATA_DIR,
    EMO_GENRES,
    NON_EMO_GENRES,
    SPOTIFY_CLIENT_ID,
    SPOTIFY_CLIENT_SECRET,
    SPOTIFY_FEATURES,
    SPOTIFY_FALLBACK_FEATURES,
)


def get_spotify_client():
    """Create authenticated Spotify client."""
    if not SPOTIFY_CLIENT_ID or not SPOTIFY_CLIENT_SECRET:
        raise ValueError(
            "Spotify credentials required. Set SPOTIFY_CLIENT_ID and "
            "SPOTIFY_CLIENT_SECRET in your environment or .env file. "
            "Get credentials at https://developer.spotify.com/dashboard"
        )
    auth = SpotifyClientCredentials(
        client_id=SPOTIFY_CLIENT_ID,
        client_secret=SPOTIFY_CLIENT_SECRET,
    )
    return spotipy.Spotify(auth_manager=auth)


# Spotify Search API limit is 1-10 per request (documented max)
SEARCH_LIMIT = 10


def fetch_tracks_by_genre(sp, genre: str, limit: int = 200) -> list[dict]:
    """Fetch track IDs for a given genre using Spotify's search."""
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
            if not items:
                break

            for item in items:
                if item["id"] not in [t["id"] for t in tracks]:
                    tracks.append({
                        "id": item["id"],
                        "name": item["name"],
                        "artist": item["artists"][0]["name"],
                        "genre": genre,
                    })
                    if len(tracks) >= limit:
                        break

            offset += SEARCH_LIMIT
            time.sleep(0.2)  # Rate limiting
        except Exception as e:
            print(f"Error fetching {genre}: {e}")
            break

    return tracks[:limit]


def fetch_audio_features(sp, track_ids: list[str]) -> list[dict] | None:
    """
    Fetch audio features for a batch of track IDs.
    Returns list of feature dicts, or None if deprecated (403).
    Spotify deprecated audio-features for new apps (~Nov 2024).
    """
    all_features = []
    for i in range(0, len(track_ids), 50):  # 50 per batch (URL length, rate limits)
        batch = track_ids[i : i + 50]
        try:
            features = sp.audio_features(batch)
            if features is None:
                return None
            valid = [f for f in features if f is not None]
            if not valid:
                return None
            all_features.extend(valid)
            time.sleep(0.2)
        except Exception as e:
            err_str = str(e).lower()
            if "403" in err_str or "forbidden" in err_str:
                print(
                    "Note: audio-features API returned 403. Spotify deprecated this "
                    "for new apps. Using fallback (duration, popularity)..."
                )
                return None
            print(f"Error fetching features: {e}")
    return all_features


def fetch_track_details(sp, track_ids: list[str]) -> dict[str, dict]:
    """Fallback: fetch basic track details when audio-features is unavailable."""
    details = {}
    for i in range(0, len(track_ids), 50):
        batch = track_ids[i : i + 50]
        for tid in batch:
            try:
                t = sp.track(tid)
                details[tid] = {
                    "id": tid,
                    "duration_ms": t.get("duration_ms", 0),
                    "popularity": t.get("popularity", 0),
                    "explicit": 1 if t.get("explicit") else 0,
                }
            except Exception:
                pass
            time.sleep(0.05)
    return details


def build_dataset(tracks_per_genre: int = 100) -> pd.DataFrame:
    """Build emo vs non-emo dataset from Spotify."""
    sp = get_spotify_client()

    rows = []
    seen_ids = set()

    # Emo tracks
    for genre in EMO_GENRES:
        print(f"Fetching emo genre: {genre}...")
        tracks = fetch_tracks_by_genre(sp, genre, limit=tracks_per_genre)
        for t in tracks:
            if t["id"] in seen_ids:
                continue
            seen_ids.add(t["id"])
            rows.append({**t, "is_emo": 1})

    emo_count = len(rows)

    # Non-emo tracks
    for genre in NON_EMO_GENRES:
        if len(rows) >= emo_count * 2:  # Balance roughly 1:1
            break
        print(f"Fetching non-emo genre: {genre}...")
        tracks = fetch_tracks_by_genre(sp, genre, limit=tracks_per_genre // len(NON_EMO_GENRES))
        for t in tracks:
            if t["id"] in seen_ids:
                continue
            seen_ids.add(t["id"])
            rows.append({**t, "is_emo": 0})

    # Fetch audio features for all tracks (or fallback to track details)
    track_ids = [r["id"] for r in rows]
    print(f"Fetching data for {len(track_ids)} tracks...")
    features_list = fetch_audio_features(sp, track_ids)
    use_fallback = features_list is None
    feature_cols = SPOTIFY_FALLBACK_FEATURES if use_fallback else SPOTIFY_FEATURES

    if use_fallback:
        print("Using fallback: fetching duration, popularity, explicit...")
        features_by_id = fetch_track_details(sp, track_ids)
    else:
        features_by_id = {f["id"]: f for f in features_list if f and "id" in f}

    # Combine
    df_rows = []
    for r in rows:
        feat = features_by_id.get(r["id"])
        if feat is None:
            continue
        row = {
            "id": r["id"],
            "name": r["name"],
            "artist": r["artist"],
            "genre": r["genre"],
            "is_emo": r["is_emo"],
        }
        for key in feature_cols:
            if key in feat:
                row[key] = feat[key]
        df_rows.append(row)

    df = pd.DataFrame(df_rows)
    return df


def main():
    """Build and save the dataset."""
    output_path = DATA_DIR / "spotify_emo_dataset.csv"
    print("Building emo vs non-emo dataset from Spotify...")
    df = build_dataset(tracks_per_genre=80)
    if df.empty:
        print("\nNo tracks collected. Possible causes:")
        print("  - Spotify credentials invalid or missing")
        print("  - audio-features API deprecated for new apps (use fallback)")
        print("  - Rate limiting or network issues")
        print("\nTry the audio-based classifier instead:")
        print("  Put MP3s in data/audio/emo/ and data/audio/not_emo/")
        print("  Then run: python -m classifier_audio")
        if output_path.exists():
            output_path.unlink()  # Remove stale empty/invalid file
        return
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} tracks to {output_path}")
    print(f"  Emo: {df['is_emo'].sum()}")
    print(f"  Non-emo: {len(df) - df['is_emo'].sum()}")


if __name__ == "__main__":
    main()
