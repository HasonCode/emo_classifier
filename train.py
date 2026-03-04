#!/usr/bin/env python3
"""
One-click training script for the emo classifier.
Run: python train.py [--audio | --spotify]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def main():
    parser = argparse.ArgumentParser(description="Train the emo music classifier")
    parser.add_argument(
        "--spotify",
        action="store_true",
        help="Use Spotify API (requires credentials, fetches data automatically)",
    )
    parser.add_argument(
        "--audio",
        action="store_true",
        help="Use local audio files from data/audio/emo/ and data/audio/not_emo/",
    )
    args = parser.parse_args()

    if args.audio:
        from classifier_audio import train_classifier
        train_classifier()
    elif args.spotify:
        from spotify_data import build_dataset
        from config import DATA_DIR
        import pandas as pd

        df = build_dataset(tracks_per_genre=80)
        df.to_csv(DATA_DIR / "spotify_emo_dataset.csv", index=False)
        print(f"Saved {len(df)} tracks")

        from classifier_spotify import train_classifier
        train_classifier(df=df)
    else:
        print("Choose --spotify or --audio")
        sys.exit(1)


if __name__ == "__main__":
    main()
