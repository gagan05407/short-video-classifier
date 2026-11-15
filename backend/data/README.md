# Dataset Directory

This directory stores dataset manifests, extracted multimodal feature tables, and video samples:

- `manifest.csv`: Stratified split manifest (train / val / test) across classes (`safe`, `dance`, `adult_content`, `violence`).
- `features.csv`: Extracted 12-dimensional multimodal feature representations for all splits.
- `speech_texts.csv`: Audio speech transcripts used for synthetic speech generation.
- `features_corrupted_*.csv`: Extracted features under physical Gaussian Blur, Darkening, and Noise corruptions.
- `test_data/`: Contains unseen test video clips (excluded from git due to >3 GB size).
- `speech_data/`: Contains synthetic audio-muxed test videos (excluded from git).
