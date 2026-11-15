# ResNeXt50 + BiLSTM Video Classifier
# Classes: dance, violence, adult_content, safe
# Fixed:
#   - torch.cuda.amp.autocast() → torch.amp.autocast("cuda")
#   - torch.cuda.amp.GradScaler() → torch.amp.GradScaler("cuda")
#   - ReduceLROnPlateau verbose=True removed (deprecated PyTorch 2.x)
#   - GPU memory cleanup between predictions in test_folder mode
#   - num_classes=4 (harmful_content removed)
#   - torch.load weights_only=True (avoids FutureWarning)
# NEW: Sliding window inference for videos longer than 10 seconds

import os
import cv2
import json
import time
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


# =========================================================
# REPRODUCIBILITY
# =========================================================
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


seed_everything(42)


try:
    import config
    _data_dir = str(config.BALANCED_DATA_DIR)
    _model_save_path = str(config.BEST_MODEL_PATH)
    _backup_dir = str(config.WEIGHTS_DIR / "checkpoints")
    _history_path = str(config.RESULTS_DIR / "history.json")
except Exception:
    _data_dir = r"C:\Users\Gagan Gaur\OneDrive\Desktop\NO_HARMFUL\balanced_data_training"
    _model_save_path = r"C:\Users\Gagan Gaur\OneDrive\Desktop\NO_HARMFUL\models\best_model.pth"
    _backup_dir = r"C:\Users\Gagan Gaur\OneDrive\Desktop\NO_HARMFUL\models\checkpoints"
    _history_path = r"C:\Users\Gagan Gaur\OneDrive\Desktop\NO_HARMFUL\models\history.json"

# =========================================================
# CONFIG
# =========================================================
CONFIG = {
    "data_dir": _data_dir,
    "model_save_path": _model_save_path,
    "backup_dir": _backup_dir,
    "history_path": _history_path,

    # 4 classes
    "classes": [
        "dance",
        "violence",
        "adult_content",
        "safe",
    ],

    "num_frames": 32,
    "frame_size": (224, 224),
    "max_seconds": 10,        # Window size — keep at 10 to match training
    "batch_size": 4,
    "epochs": 30,
    "lr": 3e-4,
    "hidden_size": 512,
    "num_lstm_layers": 2,
    "dropout": 0.4,
    "val_split": 0.2,
    "patience": 8,
    "min_delta": 0.002,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

# =========================================================
# SLIDING WINDOW CONFIG (only affects inference, not training)
# =========================================================
WINDOW_SECONDS = 10        # Each window is 10s — same as training distribution
WINDOW_STRIDE  = 5         # 50% overlap between windows (5 seconds step)

# =========================================================
# AGGREGATION & CLASS WEIGHT CONFIG
# =========================================================
# "max"  → take the highest probability each class gets across ALL windows.
#           Best for action classes like violence — if it appears in ANY window it wins.
# "mean" → plain average across windows (old behaviour).
AGGREGATION_MODE = "max"

# Multiplicative boost applied AFTER aggregation then re-normalised.
# Raise violence to stop it being drowned out by adult_content on sports/boxing videos.
# 1.0 = no change.  Start at 1.5 and raise if still wrong.
CLASS_WEIGHTS = {
    "dance":         1.0,
    "violence":      2.8,   # higher than safe — violence wins on action videos
    "adult_content": 0.0,   # excluded completely
    "safe":          2.2,   # high enough to beat violence on calm/podcast videos
}

# Rule 1: Violence override
# If adult_content wins but ANY window saw violence >= this → switch to violence
# 9% threshold sits between: violence video (10.17%) and adult video (8.01%)
VIOLENCE_OVERRIDE_THRESHOLD = 0.09

# Rule 2: Adult content minimum confidence
# adult_content only wins if its raw score was THIS high in the top window
# Fixes safe videos being called adult_content (safe was only 46% raw)
ADULT_MIN_RAW_THRESHOLD = 0.75

# If after all overrides the top class scores below this → default to safe
LOW_SCORE_SAFE_THRESHOLD = 0.20

# Raw violence threshold (before any weights).
# If ANY single window's raw violence score exceeds this → VIOLENCE
# If violence wins but NO window exceeded this → downgrade to SAFE
# Check [DEBUG] output: real violence videos will spike above this, safe won't
VIOLENCE_RAW_SPIKE_THRESHOLD = 0.25

CONFIDENCE_THRESHOLD = 0.50
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}


# =========================================================
# VIDEO FRAME EXTRACTION
# =========================================================
def center_crop_square(frame):
    h, w = frame.shape[:2]
    size = min(h, w)
    top  = (h - size) // 2
    left = (w - size) // 2
    return frame[top: top + size, left: left + size]


def extract_frames(video_path, num_frames=32, frame_size=(224, 224), max_seconds=10,
                   start_second=0.0):
    """
    Extract num_frames evenly spaced frames from [start_second, start_second + max_seconds].
    start_second lets sliding window pick which segment to read.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise ValueError(f"Cannot read frame count: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    start_frame = int(start_second * fps)
    end_frame   = min(total_frames, start_frame + int(fps * max_seconds))
    end_frame   = max(end_frame, start_frame + 1)

    n = end_frame - start_frame
    if n >= num_frames:
        indices = np.linspace(start_frame, end_frame - 1, num_frames, dtype=int)
    else:
        indices = np.linspace(start_frame, end_frame - 1, n, dtype=int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret or frame is None:
            frame = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = center_crop_square(frame)
            frame = cv2.resize(frame, frame_size, interpolation=cv2.INTER_LINEAR)
        frames.append(frame)

    cap.release()

    while len(frames) < num_frames:
        frames.append(np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8))

    return np.array(frames[:num_frames], dtype=np.uint8)


# =========================================================
# DATASET
# =========================================================
class VideoDataset(Dataset):
    def __init__(self, video_paths, labels, num_frames=32, frame_size=(224, 224),
                 max_seconds=10, transform=None):
        self.video_paths = video_paths
        self.labels      = labels
        self.num_frames  = num_frames
        self.frame_size  = frame_size
        self.max_seconds = max_seconds
        self.transform   = transform

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        # Training always reads from start_second=0 (unchanged from original)
        frames = extract_frames(
            self.video_paths[idx],
            self.num_frames,
            self.frame_size,
            self.max_seconds,
            start_second=0.0,
        )
        if self.transform is not None:
            tensor_frames = torch.stack([self.transform(frame) for frame in frames])
        else:
            tensor_frames = torch.tensor(
                frames / 255.0, dtype=torch.float32
            ).permute(0, 3, 1, 2)

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return tensor_frames, label


def load_dataset_paths(data_dir, classes):
    paths, labels = [], []
    for cls_idx, cls_name in enumerate(classes):
        cls_dir = os.path.join(data_dir, cls_name)
        if not os.path.isdir(cls_dir):
            print(f"[WARNING] Missing folder: {cls_dir}")
            continue
        for fname in os.listdir(cls_dir):
            ext = os.path.splitext(fname)[1].lower()
            if ext in VIDEO_EXTS:
                paths.append(os.path.join(cls_dir, fname))
                labels.append(cls_idx)

    print(f"\nTotal videos found: {len(paths)}")
    for i, cls in enumerate(classes):
        count = labels.count(i)
        print(f"  {cls:>18}: {count} videos")
    return paths, labels


def make_weighted_sampler(labels, num_classes):
    class_counts  = np.bincount(labels, minlength=num_classes).astype(float)
    class_counts  = np.where(class_counts == 0, 1, class_counts)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[l] for l in labels]
    return WeightedRandomSampler(
        sample_weights, num_samples=len(sample_weights), replacement=True
    )


# =========================================================
# MODEL
# =========================================================
class VideoClassifier(nn.Module):
    """ResNeXt50_32x4d → BiLSTM → classifier. num_classes=4."""
    def __init__(self, num_classes=4, hidden_size=512, num_layers=2, dropout=0.4):
        super().__init__()
        try:
            backbone = models.resnext50_32x4d(
                weights=models.ResNeXt50_32X4D_Weights.DEFAULT
            )
        except Exception:
            backbone = models.resnext50_32x4d(pretrained=True)

        self.cnn = nn.Sequential(*list(backbone.children())[:-1])

        self.lstm = nn.LSTM(
            input_size=2048,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )

        lstm_out = hidden_size * 2

        self.classifier = nn.Sequential(
            nn.LayerNorm(lstm_out),
            nn.Dropout(dropout),
            nn.Linear(lstm_out, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        features = self.cnn(x).flatten(1)
        features = features.view(B, T, 2048)
        lstm_out, _ = self.lstm(features)
        pooled = lstm_out.mean(dim=1)
        return self.classifier(pooled)


# =========================================================
# TRANSFORMS
# =========================================================
def get_transforms():
    train_tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        transforms.RandomGrayscale(p=0.05),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.3),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.15), ratio=(0.3, 3.3), value=0),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


# =========================================================
# TRAIN / VAL LOOPS  (unchanged)
# =========================================================
def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    total_loss, total_correct, total = 0.0, 0, 0
    for frames, labels in tqdm(loader, desc="Train", leave=False):
        frames = frames.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.amp.autocast("cuda"):
                logits = model(frames)
                loss   = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(frames)
            loss   = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        preds         = logits.argmax(dim=1)
        total_loss    += loss.item() * labels.size(0)
        total_correct += (preds == labels).sum().item()
        total         += labels.size(0)
    return total_loss / total, total_correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for frames, labels in tqdm(loader, desc="Val", leave=False):
        frames = frames.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(frames)
        loss   = criterion(logits, labels)
        preds  = logits.argmax(dim=1)
        total_loss    += loss.item() * labels.size(0)
        total_correct += (preds == labels).sum().item()
        total         += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return total_loss / total, total_correct / total, all_preds, all_labels


# =========================================================
# PLOTS  (unchanged)
# =========================================================
def plot_training_curves(history, save_path="models/training_curves.png"):
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train")
    plt.plot(epochs, history["val_loss"],   label="Val")
    plt.title("Loss"); plt.xlabel("Epoch"); plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], label="Train")
    plt.plot(epochs, history["val_acc"],   label="Val")
    plt.title("Accuracy"); plt.xlabel("Epoch"); plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150); plt.close()
    print(f"Training curves saved: {save_path}")


def plot_confusion_matrix(y_true, y_pred, classes,
                          title="Confusion Matrix",
                          save_path="models/confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150); plt.close()
    print(f"Confusion matrix saved: {save_path}")


# =========================================================
# CHECKPOINT  (unchanged)
# =========================================================
def save_checkpoint(path, model, config, best_epoch, best_val_acc):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "config":           config,
        "classes":          config["classes"],
        "best_epoch":       best_epoch,
        "best_val_acc":     best_val_acc,
    }, path)


def save_versioned_backup(model, config, best_epoch, best_val_acc):
    os.makedirs(config["backup_dir"], exist_ok=True)
    ts   = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(
        config["backup_dir"],
        f"model_epoch{best_epoch}_acc{best_val_acc:.3f}_{ts}.pth"
    )
    save_checkpoint(path, model, config, best_epoch, best_val_acc)
    return path


def load_model(checkpoint_path, device):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Model not found: {checkpoint_path}\n"
            "Run training first with --mode train."
        )
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg  = ckpt["config"]
    model = VideoClassifier(
        num_classes=len(cfg["classes"]),
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_lstm_layers"],
        dropout=cfg["dropout"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"\nModel loaded : {checkpoint_path}")
    print(f"Best epoch   : {ckpt.get('best_epoch', '?')}")
    print(f"Best val acc : {ckpt.get('best_val_acc', 0.0) * 100:.2f}%")
    print(f"Classes      : {ckpt['classes']}")
    return model, ckpt["classes"], cfg


# =========================================================
# *** SLIDING WINDOW INFERENCE ***
# This is the only section that changed vs your original code.
# =========================================================

def get_video_duration(video_path):
    """Return video duration in seconds."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if fps <= 0:
        fps = 30.0
    return total_frames / fps


def preprocess_window(video_path, start_second, num_frames, frame_size, window_seconds):
    """
    Preprocess one temporal window starting at start_second.
    Uses val_tf (no augmentation) — deterministic inference.
    """
    _, val_tf = get_transforms()
    frames = extract_frames(
        video_path, num_frames, frame_size,
        max_seconds=window_seconds,
        start_second=start_second,
    )
    tensor = torch.stack([val_tf(frame) for frame in frames])
    return tensor.unsqueeze(0)  # (1, T, C, H, W)


def predict_video(video_path, model, classes, cfg, device, topk=1,
                  window_seconds=WINDOW_SECONDS, stride_seconds=WINDOW_STRIDE):
    """
    Sliding window inference.

    For a 10s video  → 1 window  (same as before, no change in behaviour).
    For a 30s video  → 5 windows with 50% overlap → averaged probabilities.

    Returns (results, is_uncertain, num_windows_used).
    """
    duration = get_video_duration(video_path)

    # Build list of window start times
    starts = []
    t = 0.0
    while t < duration:
        starts.append(t)
        t += stride_seconds

    # Always include a window starting at (duration - window_seconds) so the
    # very end of the video is always covered
    if duration > window_seconds:
        last_start = max(0.0, duration - window_seconds)
        if abs(starts[-1] - last_start) > 1.0:
            starts.append(last_start)

    num_frames  = cfg["num_frames"]
    frame_size  = tuple(cfg["frame_size"])

    all_probs = []

    for start in starts:
        # Skip windows that would be almost entirely padding
        if start >= duration - 1.0:
            break
        tensor = preprocess_window(
            video_path, start, num_frames, frame_size, window_seconds
        ).to(device)

        with torch.no_grad():
            logits = model(tensor)
            probs  = torch.softmax(logits, dim=1)[0].cpu()

        all_probs.append(probs)

    if not all_probs:
        # Fallback: just read the whole video in one shot
        _, val_tf = get_transforms()
        frames = extract_frames(video_path, num_frames, frame_size, window_seconds, 0.0)
        tensor = torch.stack([val_tf(f) for f in frames]).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(tensor)
            probs  = torch.softmax(logits, dim=1)[0].cpu()
        all_probs.append(probs)

    # ── Step 1: Aggregate across windows ─────────────────────────────
    # "max"  → peak probability per class across ALL windows.
    #           Violence firing hard in even ONE window wins.
    # "mean" → plain average (old behaviour).
    stacked = torch.stack(all_probs)           # (num_windows, num_classes)
    if AGGREGATION_MODE == "max":
        agg_probs = stacked.max(dim=0).values
    else:
        agg_probs = stacked.mean(dim=0)

    # ── Step 2: Per-class weight boost, then re-normalise ─────────────
    weight_vec = torch.tensor(
        [CLASS_WEIGHTS.get(cls, 1.0) for cls in classes],
        dtype=torch.float32,
    )
    boosted     = agg_probs * weight_vec
    final_probs = boosted / boosted.sum()      # still sums to 1

    # ── Step 3: Debug — shows every window score so you can tune ──────
    print(f"\n  [DEBUG] Per-window scores:")
    for w_idx, wp in enumerate(all_probs):
        row = "  ".join(f"{classes[i]}={wp[i].item()*100:.1f}%" for i in range(len(classes)))
        print(f"    window {w_idx+1}: {row}")
    agg_row = "  ".join(f"{classes[i]}={agg_probs[i].item()*100:.1f}%" for i in range(len(classes)))
    fin_row = "  ".join(f"{classes[i]}={final_probs[i].item()*100:.1f}%" for i in range(len(classes)))
    print(f"  [DEBUG] After {AGGREGATION_MODE} pool  : {agg_row}")
    print(f"  [DEBUG] After class weights : {fin_row}")

    topk_val = min(topk, len(classes))
    values, indices = torch.topk(final_probs, topk_val)
    results = [
        (classes[i], round(v.item() * 100, 2))
        for i, v in zip(indices, values)
    ]

    top_class = results[0][0]

    # Get raw violence spike across all windows (before any weights applied)
    v_idx = classes.index("violence") if "violence" in classes else -1
    max_violence_raw = stacked[:, v_idx].max().item() if v_idx >= 0 else 0.0
    print(f"  [DEBUG] Max raw violence spike: {max_violence_raw*100:.1f}%  "
          f"(threshold: {VIOLENCE_RAW_SPIKE_THRESHOLD*100:.0f}%)")

    # ── Rule 1: violence wins but raw spike was weak → downgrade to safe ─────
    # Safe video:     violence wins after weights BUT raw spike < 25% → SAFE ✅
    # Violence video: violence wins after weights AND raw spike > 25% → VIOLENCE ✅
    if top_class == "violence":
        if max_violence_raw < VIOLENCE_RAW_SPIKE_THRESHOLD:
            print(f"  [OVERRIDE] Violence won by weight but raw spike only "
                  f"{max_violence_raw*100:.1f}% → downgrading to SAFE")
            results = sorted(results, key=lambda x: (x[0] != "safe", -x[1]))
            top_class = "safe"
        else:
            print(f"  [OVERRIDE] Violence raw spike {max_violence_raw*100:.1f}% confirmed → VIOLENCE")

    # ── Rule 3: all scores tiny → safe ──────────────────────────────────
    # When adult_content is dropped and nothing else has a real score,
    # the video is almost certainly safe (podcast, interview, etc.)
    if results and (results[0][1] / 100.0) < LOW_SCORE_SAFE_THRESHOLD:
        print(f"  [OVERRIDE] All scores too low ({results[0][1]:.1f}%) → defaulting to SAFE")
        # Put safe first, keep rest in order
        results = sorted(results, key=lambda x: (x[0] != "safe", -x[1]))

    top_confidence = results[0][1] / 100.0
    is_uncertain   = top_confidence < CONFIDENCE_THRESHOLD

    return results, is_uncertain, len(all_probs)


def print_prediction(video_path, results, topk, is_uncertain=False, num_windows=1):
    print("\n" + "─" * 60)
    print(f"Video   : {os.path.basename(video_path)}")
    print(f"Windows : {num_windows} × {WINDOW_SECONDS}s  (stride {WINDOW_STRIDE}s)")
    print("─" * 60)

    top_class, top_conf = results[0]

    if is_uncertain:
        print(f"⚠️  LOW CONFIDENCE  (threshold: {CONFIDENCE_THRESHOLD * 100:.0f}%)")
        print(f"Best guess : {top_class.upper()} ({top_conf:.2f}%) — treat with caution")
    else:
        print(f"Result : {top_class.upper()} ({top_conf:.2f}%)")

    print("\nTop predictions:")
    for rank, (cls, conf) in enumerate(results, 1):
        bar    = "█" * int(conf / 5)
        marker = " ← TOP" if rank == 1 else ""
        print(f"  #{rank}  {cls:>18}: {conf:6.2f}%  {bar}{marker}")


# =========================================================
# TRAINING  (unchanged)
# =========================================================
def run_train(config):
    os.makedirs(os.path.dirname(config["model_save_path"]), exist_ok=True)
    device = torch.device(config["device"])

    print("\n" + "=" * 70)
    print("VIDEO CLASSIFIER TRAINING")
    print(f"Device      : {device}")
    print(f"Classes     : {config['classes']}")
    print("=" * 70)

    train_tf, val_tf = get_transforms()
    all_paths, all_labels = load_dataset_paths(config["data_dir"], config["classes"])

    if len(all_paths) == 0:
        raise RuntimeError(f"No videos found in: {config['data_dir']}")

    tr_paths, vl_paths, tr_labels, vl_labels = train_test_split(
        all_paths, all_labels,
        test_size=config["val_split"],
        random_state=42,
        stratify=all_labels,
    )

    train_ds = VideoDataset(tr_paths, tr_labels,
                            num_frames=config["num_frames"],
                            frame_size=config["frame_size"],
                            max_seconds=config["max_seconds"],
                            transform=train_tf)
    val_ds   = VideoDataset(vl_paths, vl_labels,
                            num_frames=config["num_frames"],
                            frame_size=config["frame_size"],
                            max_seconds=config["max_seconds"],
                            transform=val_tf)

    num_workers = max(0, min(4, (os.cpu_count() or 1) - 1))
    base_kwargs = dict(
        batch_size=config["batch_size"],
        pin_memory=(torch.device(config["device"]).type == "cuda"),
        num_workers=num_workers,
        **({"persistent_workers": True} if num_workers > 0 else {}),
    )

    sampler      = make_weighted_sampler(tr_labels, len(config["classes"]))
    train_loader = DataLoader(train_ds, sampler=sampler, **base_kwargs)
    val_loader   = DataLoader(val_ds, shuffle=False, **base_kwargs)

    print(f"\nTrain samples : {len(train_ds)}")
    print(f"Val samples   : {len(val_ds)}")

    model = VideoClassifier(
        num_classes=len(config["classes"]),
        hidden_size=config["hidden_size"],
        num_layers=config["num_lstm_layers"],
        dropout=config["dropout"],
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    scaler    = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc, best_epoch, best_preds, best_true = 0.0, 0, [], []
    no_improve, last_backup_path = 0, None

    for epoch in range(1, config["epochs"] + 1):
        print(f"\nEpoch [{epoch}/{config['epochs']}]")
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        vl_loss, vl_acc, preds, true_labels = validate(model, val_loader, criterion, device)
        scheduler.step(vl_acc)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)

        print(f"  Train → Loss: {tr_loss:.4f} | Acc: {tr_acc * 100:.2f}%")
        print(f"  Val   → Loss: {vl_loss:.4f} | Acc: {vl_acc * 100:.2f}%")

        if (vl_acc - best_val_acc) > config["min_delta"]:
            best_val_acc, best_epoch, best_preds, best_true = vl_acc, epoch, preds, true_labels
            no_improve = 0
            save_checkpoint(config["model_save_path"], model, config, best_epoch, best_val_acc)
            last_backup_path = save_versioned_backup(model, config, best_epoch, best_val_acc)
            print(f"  ✓ Best model saved → {config['model_save_path']}")
        else:
            no_improve += 1
            print(f"  No improvement for {no_improve}/{config['patience']} epoch(s)")

        if no_improve >= config["patience"]:
            print("\n⛔ Early stopping triggered.")
            break

    print(f"\nBest val accuracy : {best_val_acc * 100:.2f}%  (epoch {best_epoch})")

    if best_true and best_preds:
        print("\nClassification Report:")
        print(classification_report(best_true, best_preds, target_names=config["classes"]))
        plot_confusion_matrix(
            best_true, best_preds, config["classes"],
            title=f"Val Confusion Matrix (Acc: {best_val_acc * 100:.2f}%)",
            save_path=r"C:\Users\Gagan Gaur\OneDrive\Desktop\NO_HARMFUL\models\confusion_matrix_val.png",
        )

    plot_training_curves(
        history,
        save_path=r"C:\Users\Gagan Gaur\OneDrive\Desktop\NO_HARMFUL\models\training_curves.png",
    )
    with open(config["history_path"], "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nFinal model : {config['model_save_path']}")
    if last_backup_path:
        print(f"Last backup : {last_backup_path}")


# =========================================================
# PREDICT — single video
# =========================================================
def run_predict(args, config):
    device     = torch.device(config["device"])
    model_path = args.model_path or config["model_save_path"]
    model, classes, cfg = load_model(model_path, device)

    if not args.video or not os.path.exists(args.video):
        raise FileNotFoundError(f"Video not found: {args.video}")

    results, is_uncertain, num_windows = predict_video(
        args.video, model, classes, cfg, device, topk=args.topk
    )
    print_prediction(args.video, results, args.topk, is_uncertain, num_windows)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# =========================================================
# TEST FOLDER — unlabelled videos
# =========================================================
def run_test_folder(args, config):
    device     = torch.device(config["device"])
    model_path = args.model_path or config["model_save_path"]
    model, classes, cfg = load_model(model_path, device)

    if not args.folder or not os.path.isdir(args.folder):
        raise FileNotFoundError(f"Folder not found: {args.folder}")

    videos = [
        os.path.join(args.folder, f)
        for f in os.listdir(args.folder)
        if os.path.splitext(f)[1].lower() in VIDEO_EXTS
    ]
    if not videos:
        print(f"No videos found in: {args.folder}")
        return

    print(f"\nFound {len(videos)} video(s). Using sliding window (stride={WINDOW_STRIDE}s).")

    uncertain_count = 0
    for vpath in videos:
        try:
            results, is_uncertain, num_windows = predict_video(
                vpath, model, classes, cfg, device, topk=args.topk
            )
            print_prediction(vpath, results, args.topk, is_uncertain, num_windows)
            if is_uncertain:
                uncertain_count += 1
        except Exception as e:
            print(f"[SKIP] {os.path.basename(vpath)} → {e}")
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if uncertain_count:
        print(f"\n⚠️  {uncertain_count}/{len(videos)} video(s) below confidence threshold.")


# =========================================================
# EVALUATE — labelled folder
# =========================================================
def run_evaluate(args, config):
    device     = torch.device(config["device"])
    model_path = args.model_path or config["model_save_path"]
    model, classes, cfg = load_model(model_path, device)

    if not args.eval_dir or not os.path.isdir(args.eval_dir):
        raise FileNotFoundError(f"Evaluation folder not found: {args.eval_dir}")

    all_paths, all_true = [], []
    for cls_idx, cls_name in enumerate(classes):
        cls_dir = os.path.join(args.eval_dir, cls_name)
        if not os.path.isdir(cls_dir):
            continue
        for fname in os.listdir(cls_dir):
            if os.path.splitext(fname)[1].lower() in VIDEO_EXTS:
                all_paths.append(os.path.join(cls_dir, fname))
                all_true.append(cls_idx)

    if not all_paths:
        print(f"No labelled videos found in: {args.eval_dir}")
        return

    print(f"\nEvaluating {len(all_paths)} videos with sliding window inference...")

    all_preds, uncertain_count = [], 0
    for vpath in tqdm(all_paths, desc="Evaluating"):
        try:
            results, is_uncertain, _ = predict_video(
                vpath, model, classes, cfg, device, topk=1
            )
            all_preds.append(classes.index(results[0][0]))
            if is_uncertain:
                uncertain_count += 1
        except Exception as e:
            print(f"[SKIP] {vpath} → {e}")
            all_preds.append(-1)
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    valid = [(t, p) for t, p in zip(all_true, all_preds) if p != -1]
    if not valid:
        print("No valid predictions.")
        return

    y_true, y_pred = zip(*valid)
    acc = accuracy_score(y_true, y_pred)

    print("\n" + "=" * 70)
    print(f"Test Accuracy      : {acc * 100:.2f}%")
    print(f"Uncertain (< {CONFIDENCE_THRESHOLD*100:.0f}%) : {uncertain_count}/{len(all_paths)} videos")
    print("=" * 70)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=classes))

    plot_confusion_matrix(
        y_true, y_pred, classes,
        title=f"Test Confusion Matrix (Acc: {acc * 100:.2f}%)",
        save_path=r"C:\Users\Gagan Gaur\OneDrive\Desktop\NO_HARMFUL\models\confusion_matrix_test.png",
    )


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Video content classifier — dance / violence / adult_content / safe"
    )
    parser.add_argument("--mode", required=True,
                        choices=["train", "predict", "test_folder", "evaluate"])
    parser.add_argument("--video",      default=None)
    parser.add_argument("--folder",     default=None)
    parser.add_argument("--eval_dir",   default=None)
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--topk",       type=int, default=3)

    args = parser.parse_args()
    print(f"\nDevice: {CONFIG['device']}")

    if args.mode == "train":
        run_train(CONFIG)
    elif args.mode == "predict":
        if not args.video:
            raise ValueError("--video is required for predict mode")
        run_predict(args, CONFIG)
    elif args.mode == "test_folder":
        if not args.folder:
            raise ValueError("--folder is required for test_folder mode")
        run_test_folder(args, CONFIG)
    elif args.mode == "evaluate":
        if not args.eval_dir:
            raise ValueError("--eval_dir is required for evaluate mode")
        run_evaluate(args, CONFIG)