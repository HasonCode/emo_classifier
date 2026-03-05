#!/bin/bash
# Train emo classifier on data in data/audio/

set -e
cd "$(dirname "$0")"

echo "=== Emo Audio Classifier - Training ==="
echo "Data: data/audio/emo/ and data/audio/not_emo/"
echo ""

if [ ! -d "data/audio/emo" ] || [ ! -d "data/audio/not_emo" ]; then
    echo "Error: Create data/audio/emo/ and data/audio/not_emo/ and add audio files."
    exit 1
fi

emo_count=$(find data/audio/emo -type f \( -name "*.mp3" -o -name "*.wav" -o -name "*.flac" \) 2>/dev/null | wc -l)
not_count=$(find data/audio/not_emo -type f \( -name "*.mp3" -o -name "*.wav" -o -name "*.flac" \) 2>/dev/null | wc -l)
echo "Found: $emo_count emo, $not_count non-emo"
echo ""

poetry run python -m classifier_audio
