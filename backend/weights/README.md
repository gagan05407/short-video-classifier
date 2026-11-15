# Pretrained Model Weights Directory

This folder contains model checkpoints for the Multimodal Short-Video Classifier:

| Model File | Architecture / Task | Size | Included in Git |
| :--- | :--- | :--- | :--- |
| `gate_frozen.pt` | Adaptive Reliability Gate (12-in MLP) | ~3.8 KB | **Yes** (tracked) |
| `gate_config.json` | Gate hyperparameter & normalization config | ~2.0 KB | **Yes** (tracked) |
| `best_model.pth` | Temporal ResNeXt50 + BiLSTM Action Classifier | ~160 MB | **No** (exceeds GitHub 100MB limit) |
| `nsfw_resnet18_small_state.pth` | Static Frame ResNet18 NSFW Classifier | ~45 MB | **No** (optional download) |

### Note on Large Checkpoints:
If cloning fresh, place `best_model.pth` and `nsfw_resnet18_small_state.pth` directly into this `backend/weights/` directory before running inference.
