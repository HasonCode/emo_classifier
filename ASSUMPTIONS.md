# Assumptions

This document lists the key assumptions underlying the emo music classifier.

## Emo Music Definition

- **Emo music is included** in this project as a distinct, classifiable category.
- Emo is treated as a **binary label** (emo vs not emo) rather than a spectrum or multi-label taxonomy.
- The classifier conflates related subgenres: **emo**, **emo-pop**, **pop-punk**, **post-hardcore**, and **screamo** are all treated as positive (emo) examples for training.
- Genre boundaries are assumed to be meaningful enough that acoustic features can distinguish emo from non-emo tracks.
- Spotify’s genre tags (e.g. `genre:emo`, `genre:pop-punk`) are assumed to be reasonably accurate for building training data.
- Non-emo tracks are drawn from broad genres (pop, rock, hip-hop, electronic, country, jazz, classical, R&B, reggae, metal, indie, dance); overlap with emo-adjacent styles (e.g. some indie, metal) may exist.

## Data Assumptions

- Spotify’s search by genre returns representative tracks for that genre.
- Spotify’s acoustic features (energy, valence, danceability, etc.) capture aspects relevant to emo vs non-emo discrimination.
- For the audio-based model, user-provided labels in `data/audio/emo/` and `data/audio/not_emo/` are correct.
- Training data is reasonably balanced between emo and non-emo.
- Artist and release-date bias in the training set is not explicitly addressed.
- No manual review or correction of Spotify search results is assumed.

## Model Assumptions

- A **Random Forest** classifier with the chosen feature sets is sufficient for this task.
- Feature standardization (zero mean, unit variance) improves performance.
- The relationship between acoustic features and emo/non-emo is learnable from the available data.
- Overfitting is partially mitigated by `max_depth=10` and similar hyperparameters; no tuning is assumed.
- Model performance on held-out data is assumed to generalise to new, unseen tracks.

## Technical Assumptions

- **Spotify model**: Spotify API credentials are available and rate limits are respected.
- **Audio model**: Input audio is stereo or mono; only the first 30 seconds are used for feature extraction.
- Supported formats include MP3, WAV, FLAC, OGG, M4A, WebM.
- Sample rate is normalised to 22,050 Hz for audio feature extraction.
- Extracted features (MFCC, chroma, spectral centroid, etc.) are sufficiently informative for classification.
- The project runs in a Python 3.10+ environment with the specified dependencies.

## Use-Case Assumptions

- The classifier is intended for exploratory or auxiliary use, not as a definitive authority on genre.
- Predictions are treated as probabilities, not hard truths; edge cases and misclassifications are expected.
- The system is used in good faith; output is not assumed to be free from bias or errors.
