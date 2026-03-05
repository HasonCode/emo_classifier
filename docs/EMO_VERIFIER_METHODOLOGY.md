# Emo Dataset Verifier: Approach and Methodology

## Overview

The emo dataset verifier (`verify_emo_dataset.py`) identifies hand-labeled emo tracks that may be mislabeled—i.e., tracks in `emo/` that are actually non-emo. Because we cannot listen to audio programmatically, the verifier uses two complementary statistical methods to flag candidates for manual review.

## Problem Statement

Training data quality directly affects classifier accuracy. If non-emo tracks are placed in `emo/`, the model learns incorrect patterns and produces more false positives. The verifier helps surface potential mislabels so a human can listen and correct them.

## Method 1: Local Outlier Factor (LOF)

### Idea

Emo music shares certain acoustic characteristics (e.g., guitar-driven, dynamic range, typical tempo and timbre). Tracks labeled emo that differ strongly from the emo cluster in feature space may be mislabeled.

### How It Works

1. **Features**: For each emo sample, we extract Librosa audio features (MFCC, chroma, spectral centroid, rolloff, bandwidth, zero-crossing rate, RMS energy, tempo).
2. **Space**: We build a feature matrix of all emo samples only (no non-emo).
3. **LOF**: We fit `LocalOutlierFactor` on this emo-only feature space. LOF assigns each sample an "outlier score."
4. **Selection**: Samples with the most negative scores are the strongest outliers—their audio features diverge the most from the typical emo cluster.

### Interpretation

- **Outlier ≠ guaranteed mislabel**. Outliers may be legitimately emo but stylistically different (e.g., acoustic emo vs. post-hardcore).
- **Outlier = worth checking**. These are the best candidates for manual listening and potential re-labeling.

### Parameters

- `n_neighbors`: Set to `min(20, len(emo)//2)` to adapt to dataset size.
- `n`: Top N most outlier-like emo samples are flagged (default 40).

---

## Method 2: Cross-Validation Misprediction

### Idea

If a model trained on most of the data repeatedly predicts an emo sample as "Not Emo" with high confidence, that sample may be mislabeled.

### How It Works

1. **Stratified K-Fold**: Split the full dataset (emo + not_emo) into K folds while preserving class balance.
2. **Per fold**: Train a RandomForest on K−1 folds, predict on the held-out fold.
3. **Focus on emo samples**: For each emo sample, record how often it was predicted "Not Emo" with emo probability below a threshold (e.g., 0.25).
4. **Flagging**: Emo samples mispredicted in at least half of the folds are flagged.

### Interpretation

- A track labeled emo but consistently predicted Not Emo suggests it matches the non-emo distribution in feature space.
- This method uses the learned boundary between emo and not_emo, so it complements LOF (which uses only emo internal structure).

### Parameters

- `n_splits`: 5 folds.
- `prob_threshold`: Emo probability below 0.25 counts as a misprediction.

---

## Combining Both Methods

The verifier takes the **union** of samples flagged by either method. A file can be flagged because:

1. It is an outlier in the emo cluster (LOF), or  
2. It is often predicted as Not Emo in cross-validation (CV), or  
3. Both.

The combined list is written to `data/audio_verification_report.txt` for manual review.

---

## Workflow

```
┌─────────────────────────────────────────────────────────┐
│  data/audio/emo/  (hand-labeled emo tracks)              │
│  data/audio/not_emo/  (hand-labeled non-emo tracks)      │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│  Extract Librosa features (MFCC, chroma, spectral, etc.) │
└─────────────────────────────────────────────────────────┘
                            │
            ┌───────────────┴───────────────┐
            ▼                             ▼
┌──────────────────────┐      ┌──────────────────────────┐
│  LOF on emo-only     │      │  Stratified K-Fold CV     │
│  → Flag outliers     │      │  → Flag emo → Not Emo     │
└──────────────────────┘      └──────────────────────────┘
            │                             │
            └───────────────┬───────────────┘
                            ▼
┌─────────────────────────────────────────────────────────┐
│  Union of flagged paths → audio_verification_report.txt  │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│  Human listens, moves confirmed non-emo to not_emo/     │
│  Retrain: python -m classifier_audio                    │
└─────────────────────────────────────────────────────────┘
```

---

## Limitations

1. **No ground truth**: We infer possible mislabels from statistics, not from listening.
2. **Feature coverage**: Emo is partly defined by lyrics and emotional content, which Librosa features do not capture. Some mislabels may not be detected.
3. **Edge cases**: Stylistically unusual but correct emo (e.g., slowcore, acoustic) may appear as outliers and be flagged.
4. **Circularity in CV**: The model is trained on the same labels we are verifying. If many emo samples are mislabeled, the boundary is skewed and CV may miss some errors.

---

## Usage

```bash
poetry run python -m verify_emo_dataset
```

Then:

1. Open `data/audio_verification_report.txt`
2. Listen to the listed files
3. Move any confirmed non-emo files from `emo/` to `not_emo/`
4. Retrain: `poetry run python -m classifier_audio`

---

## References

- Local Outlier Factor: Breunig et al., "LOF: Identifying Density-Based Local Outliers" (SIGMOD 2000)
- Librosa: McFee et al., "librosa: Audio and Music Signal Analysis in Python" (SciPy 2015)
