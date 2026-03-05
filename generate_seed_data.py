"""
Generate a synthetic seed dataset when Spotify API is rate-limited.
Creates data/spotify_emo_dataset.csv so you can train immediately:
    poetry run python -m generate_seed_data
    poetry run python -m classifier_spotify
"""

import random
from pathlib import Path

from config import DATA_DIR, SPOTIFY_FALLBACK_FEATURES


def generate_seed_dataset(n_emo: int = 150, n_non_emo: int = 150) -> None:
    """Generate synthetic emo/non-emo data with plausible distributions."""
    random.seed(42)
    rows = []

    # Emo: 2.5–4 min, often explicit, moderate popularity (not mega-hits)
    for _ in range(n_emo):
        duration_ms = random.randint(150_000, 270_000)
        popularity = random.randint(20, 65)
        explicit = 1 if random.random() < 0.55 else 0
        rows.append({"duration_ms": duration_ms, "popularity": popularity, "explicit": explicit, "is_emo": 1})

    # Non-emo: includes pop hits (very high popularity, rarely explicit), broader range
    for _ in range(n_non_emo):
        duration_ms = random.randint(120_000, 360_000)
        popularity = random.randint(25, 95)
        explicit = 1 if random.random() < 0.25 else 0
        rows.append({"duration_ms": duration_ms, "popularity": popularity, "explicit": explicit, "is_emo": 0})

    random.shuffle(rows)

    import pandas as pd
    df = pd.DataFrame(rows)
    out = DATA_DIR / "spotify_emo_dataset.csv"
    df.to_csv(out, index=False)
    print(f"Generated {len(df)} rows → {out}")
    print(f"  Emo: {df['is_emo'].sum()}, Non-emo: {len(df) - df['is_emo'].sum()}")
    print("Run: poetry run python -m classifier_spotify")


if __name__ == "__main__":
    generate_seed_dataset()
