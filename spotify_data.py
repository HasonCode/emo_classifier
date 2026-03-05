"""Collect training data from Spotify API for emo vs non-emo classification."""

import json
import re
import time
from pathlib import Path

import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

from config import (
    DATA_DIR,
    EMO_GENRES,
    EMO_PLAYLIST_IDS,
    NON_EMO_GENRES,
    SPOTIFY_CLIENT_ID,
    SPOTIFY_CLIENT_SECRET,
    SPOTIFY_FEATURES,
    SPOTIFY_FALLBACK_FEATURES,
)

# Rate-limit avoidance
SEARCH_DELAY = 0.6
BATCH_DELAY = 0.5
PHASE_DELAY = 2.0
RETRY_BASE_DELAY = 5
RETRY_MAX_DELAY = 60


def _retry_until(delay_sec: float | None = None):
    """Sleep. If delay_sec looks like rate-limit retry (e.g. 83545), cap at 300s."""
    if delay_sec is not None and delay_sec > 0:
        wait = min(delay_sec, 300) if delay_sec > 60 else delay_sec
        print(f"  Rate limited. Waiting {wait:.0f}s...")
        time.sleep(wait)


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


SEARCH_LIMIT = 10


def fetch_tracks_from_playlist(sp, playlist_id: str, limit: int = 100) -> list[dict]:
    """Fetch tracks from a curated playlist (real emo verified by curators)."""
    tracks = []
    offset = 0
    while len(tracks) < limit:
        try:
            result = sp.playlist_tracks(playlist_id, limit=min(50, limit - len(tracks)), offset=offset)
            items = result.get("items", [])
            if not items:
                break
            for item in items:
                t = item.get("track")
                if not t or not t.get("id"):
                    continue
                tracks.append({
                    "id": t["id"],
                    "name": t["name"],
                    "artist": t["artists"][0]["name"] if t.get("artists") else "",
                    "genre": "playlist",
                })
                if len(tracks) >= limit:
                    break
            offset += len(items)
            time.sleep(BATCH_DELAY)
            if not result.get("next"):
                break
        except Exception as e:
            err = str(e).lower()
            if "429" in err or "rate" in err:
                m = re.search(r"retry after[:\s]+(\d+)", str(e), re.I)
                retry_after = min(int(m.group(1)), 300) if m else 60
                _retry_until(retry_after)
            else:
                print(f"  Playlist {playlist_id}: {e}")
                break
    return tracks[:limit]


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
            time.sleep(SEARCH_DELAY)
        except Exception as e:
            err = str(e).lower()
            if "429" in err or "rate" in err:
                m = re.search(r"retry after[:\s]+(\d+)", str(e), re.I)
                retry_after = min(int(m.group(1)), 300) if m else 60
                _retry_until(retry_after)
            else:
                print(f"Error fetching {genre}: {e}")
                break

    return tracks[:limit]


def _probe_audio_features(sp, track_ids: list[str]) -> bool:
    """Try a small batch to see if audio-features API works (not 403)."""
    try:
        batch = track_ids[:5]
        features = sp.audio_features(batch)
        if features and any(f for f in features if f and f.get("danceability") is not None):
            return True
    except Exception as e:
        if "403" in str(e).lower() or "forbidden" in str(e).lower():
            return False
    return False


def fetch_audio_features(sp, track_ids: list[str]) -> list[dict] | None:
    """
    Fetch audio features (danceability, energy, valence, etc.) - actual music properties.
    Returns None if deprecated (403). Uses longer delays to avoid rate limits.
    """
    all_features = []
    for i in range(0, len(track_ids), 50):
        batch = track_ids[i : i + 50]
        for attempt in range(5):
            try:
                features = sp.audio_features(batch)
                if features is None:
                    return None
                valid = [f for f in features if f is not None and f.get("id")]
                if not valid:
                    return None
                all_features.extend(valid)
                time.sleep(BATCH_DELAY)
                break
            except Exception as e:
                err_str = str(e).lower()
                if "403" in err_str or "forbidden" in err_str:
                    print("Note: audio-features returned 403. Using fallback (duration, popularity)...")
                    return None
                if "429" in err_str or "rate" in err_str:
                    retry_after = 60
                    m = re.search(r"retry after[:\s]+(\d+)", str(e), re.I)
                    if m:
                        retry_after = min(int(m.group(1)), 300)
                    _retry_until(retry_after)
                else:
                    print(f"Error fetching features: {e}")
                    return None
    return all_features


def fetch_track_details(sp, track_ids: list[str]) -> dict[str, dict]:
    """Fallback: fetch basic track details when audio-features is unavailable.
    Uses batch sp.tracks() (max 50 per request). Retries on 429.
    """
    details = {}
    for i in range(0, len(track_ids), 50):
        batch = track_ids[i : i + 50]
        for attempt in range(5):
            try:
                result = sp.tracks(batch)
                for t in result.get("tracks", []) or []:
                    if t and t.get("id"):
                        details[t["id"]] = {
                            "id": t["id"],
                            "duration_ms": t.get("duration_ms", 0),
                            "popularity": t.get("popularity", 0),
                            "explicit": 1 if t.get("explicit") else 0,
                        }
                time.sleep(BATCH_DELAY)
                break
            except Exception as e:
                err = str(e).lower()
                if "429" in err or "rate" in err:
                    m = re.search(r"retry after[:\s]+(\d+)", str(e), re.I)
                    retry_after = min(int(m.group(1)), 300) if m else 60
                    _retry_until(retry_after)
                else:
                    print(f"Error fetching track batch: {e}")
                    break
    return details


def build_dataset(
    tracks_per_genre: int = 35,
    emo_from_playlists: bool = True,
) -> pd.DataFrame:
    """Build emo vs non-emo dataset from Spotify.
    Emo: curated playlists (real emo) first, then genre search. Non-emo: genre search.
    """
    sp = get_spotify_client()
    rows = []
    seen_ids = set()

    # Phase 1: Emo from curated playlists (human-verified real emo)
    if emo_from_playlists and EMO_PLAYLIST_IDS:
        per_playlist = max(20, tracks_per_genre * 2)
        for pid in EMO_PLAYLIST_IDS:
            print(f"Fetching emo playlist {pid}...")
            tracks = fetch_tracks_from_playlist(sp, pid, limit=per_playlist)
            for t in tracks:
                if t["id"] in seen_ids:
                    continue
                seen_ids.add(t["id"])
                rows.append({**t, "is_emo": 1})
            time.sleep(PHASE_DELAY)
        print(f"  Emo from playlists: {sum(1 for r in rows if r['is_emo'])}")

    # Phase 2: Emo from genre search (fill remaining)
    for genre in EMO_GENRES:
        print(f"Fetching emo genre: {genre}...")
        tracks = fetch_tracks_by_genre(sp, genre, limit=tracks_per_genre)
        for t in tracks:
            if t["id"] in seen_ids:
                continue
            seen_ids.add(t["id"])
            rows.append({**t, "is_emo": 1})
        time.sleep(PHASE_DELAY)

    emo_count = len(rows)
    # Ensure we fetch enough non-emo to balance (same total as emo)
    non_emo_per_genre = max(15, emo_count // len(NON_EMO_GENRES))

    # Phase 3: Non-emo tracks
    time.sleep(PHASE_DELAY)
    for genre in NON_EMO_GENRES:
        if len(rows) >= emo_count * 2:
            break
        print(f"Fetching non-emo genre: {genre}...")
        tracks = fetch_tracks_by_genre(sp, genre, limit=non_emo_per_genre)
        for t in tracks:
            if t["id"] in seen_ids:
                continue
            seen_ids.add(t["id"])
            rows.append({**t, "is_emo": 0})
        time.sleep(SEARCH_DELAY)

    # Phase 4: Fetch features - try audio-features first (actual music properties)
    time.sleep(PHASE_DELAY)
    track_ids = [r["id"] for r in rows]
    print(f"Fetching features for {len(track_ids)} tracks...")
    use_fallback = True
    if track_ids:
        if _probe_audio_features(sp, track_ids):
            print("Using audio-features (danceability, energy, valence, etc.)")
            features_list = fetch_audio_features(sp, track_ids)
            use_fallback = features_list is None
        else:
            print("audio-features unavailable (403). Using fallback...")
        if use_fallback:
            features_list = None
    feature_cols = SPOTIFY_FALLBACK_FEATURES if use_fallback else SPOTIFY_FEATURES

    if use_fallback:
        print("Fetching duration, popularity, explicit...")
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
    df = build_dataset(tracks_per_genre=35, emo_from_playlists=True)
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
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} tracks to {output_path}")
    print(f"  Emo: {df['is_emo'].sum()}")
    print(f"  Non-emo: {len(df) - df['is_emo'].sum()}")


if __name__ == "__main__":
    main()
