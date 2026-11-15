import os
import sys
import time
import math
import base64
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
from torchvision import transforms, models
from pathlib import Path

# Add backend to path
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import config


def binary_entropy(p: float) -> float:
    """Normalized binary entropy H(p)/ln(2) in [0, 1]."""
    p = float(np.clip(p, 1e-7, 1.0 - 1e-7))
    h = -(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))
    return float(np.clip(h, 0.0, 1.0))


def compute_frame_blur(frame: np.ndarray) -> float:
    """Computes Laplacian variance as an indicator of frame sharpness."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_frame_brightness(frame: np.ndarray) -> float:
    """Computes mean grayscale luminance [0, 255]."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))


def load_nsfw_model(weights_path: str = None, device: str = None) -> nn.Module:
    """Loads ResNet18 binary NSFW checkpoint."""
    weights_path = weights_path or str(config.NSFW_MODEL_PATH)
    device = device or config.DEVICE

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def analyze_static_frames(video_path: str, model: nn.Module = None, device: str = None,
                          sample_interval: float = 0.5, corruption: str = None) -> dict:
    """
    Extracts periodic frames from video, scores with ResNet18 NSFW model,
    and returns risk p, entropy uncertainty u, visual quality q, and top 3 frames.
    """
    t0 = time.time()
    device = device or config.DEVICE
    if model is None:
        model = load_nsfw_model(device=device)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {
            "p": 0.0, "u": 0.0, "q": 0.0, "present": 0,
            "verdict": "SAFE", "time_seconds": round(time.time() - t0, 3),
            "frame_scores": [], "top_frames": [],
            "raw_signals": {"error": "could not open video"}
        }

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = max(1, int(fps * sample_interval))

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    frame_scores = []
    blur_scores = []
    bright_scores = []
    top_frames_raw = []
    frame_num = 0

    model.eval()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_num % frame_interval == 0:
            if corruption == "blur":
                frame = cv2.GaussianBlur(frame, (9, 9), 3.0)
            elif corruption == "dark":
                frame = cv2.convertScaleAbs(frame, alpha=0.35)

            timestamp = round(frame_num / fps, 2)
            blur_val = compute_frame_blur(frame)
            bright_val = compute_frame_brightness(frame)
            blur_scores.append(blur_val)
            bright_scores.append(bright_val)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb).convert("RGB")
            img = ImageOps.exif_transpose(img)
            x = preprocess(img).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(x)
                probs = F.softmax(logits, dim=1).cpu().numpy()[0]
                nsfw_prob = float(probs[0])
                sfw_prob = float(probs[1])

            frame_scores.append({
                "timestamp": timestamp,
                "nsfw": round(nsfw_prob, 4),
                "sfw": round(sfw_prob, 4)
            })

            # Keep top 3 scoring frames
            if len(top_frames_raw) < 3 or nsfw_prob > top_frames_raw[-1][0]:
                thumb = cv2.resize(frame, (256, 256))
                top_frames_raw.append((nsfw_prob, timestamp, thumb))
                top_frames_raw.sort(key=lambda item: item[0], reverse=True)
                top_frames_raw = top_frames_raw[:3]

        frame_num += 1

    cap.release()

    top_frames = []
    for s_prob, ts, th_img in top_frames_raw:
        success, buf = cv2.imencode(".jpg", th_img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if success:
            b64_str = base64.b64encode(buf).decode("utf-8")
            top_frames.append({
                "timestamp": ts,
                "score": round(float(s_prob), 4),
                "image_b64": f"data:image/jpeg;base64,{b64_str}"
            })

    if not frame_scores:
        return {
            "p": 0.0, "u": 0.0, "q": 0.0, "present": 0,
            "verdict": "SAFE", "time_seconds": round(time.time() - t0, 3),
            "frame_scores": [], "top_frames": [],
            "raw_signals": {"error": "no frames extracted"}
        }

    nsfw_vals = [f["nsfw"] for f in frame_scores]
    total_frames = len(nsfw_vals)
    max_nsfw = float(np.max(nsfw_vals))
    mean_nsfw = float(np.mean(nsfw_vals))

    # Aggregated frame risk score: 60% peak + 40% mean
    p = float(np.clip(0.6 * max_nsfw + 0.4 * mean_nsfw, 0.0, 1.0))
    u = binary_entropy(p)

    # Streak tracking (from original predict_video_nsfw.py)
    # Condition A: at least 5 consecutive frames (>0.5) AND >= 30% of total frames
    # Condition B: at least 15 consecutive frames alone (>0.5)
    nsfw_threshold = 0.5
    min_consecutive = 5
    min_flagged_percent = 30.0
    severe_consecutive = 15

    max_consecutive = 0
    current_streak = 0
    flagged_streak_start = None
    sustained_detections = []

    for i, s_score in enumerate(nsfw_vals):
        if s_score > nsfw_threshold:
            current_streak += 1
            if current_streak == 1:
                flagged_streak_start = frame_scores[i]["timestamp"]
            if current_streak > max_consecutive:
                max_consecutive = current_streak
        else:
            if current_streak >= min_consecutive:
                sustained_detections.append({
                    "start": flagged_streak_start,
                    "end": frame_scores[i - 1]["timestamp"],
                    "duration": round(current_streak * sample_interval, 1)
                })
            current_streak = 0

    if current_streak >= min_consecutive:
        sustained_detections.append({
            "start": flagged_streak_start,
            "end": frame_scores[-1]["timestamp"],
            "duration": round(current_streak * sample_interval, 1)
        })

    flagged_count = sum(1 for s in nsfw_vals if s > nsfw_threshold)
    flagged_percent = round(flagged_count / total_frames * 100, 1) if total_frames > 0 else 0.0

    condition_a = (max_consecutive >= min_consecutive) and (flagged_percent >= min_flagged_percent)
    condition_b = max_consecutive >= severe_consecutive
    overall_flagged = condition_a or condition_b

    if condition_b:
        flag_reason = "severe_sustained_nudity"
    elif condition_a:
        flag_reason = "sustained_and_widespread_nudity"
    elif max_consecutive >= min_consecutive:
        flag_reason = "consecutive_only"
    elif flagged_percent >= min_flagged_percent:
        flag_reason = "percent_only"
    else:
        flag_reason = "clean"

    # Quality indicators: sharpness (Laplacian variance), brightness, confidence
    mean_blur = float(np.mean(blur_scores)) if blur_scores else 0.0
    q_blur = float(np.clip(mean_blur / 200.0, 0.0, 1.0))

    mean_bright = float(np.mean(bright_scores)) if bright_scores else 128.0
    q_bright = float(np.clip(1.0 - 2.0 * abs(mean_bright - 128.0) / 255.0, 0.0, 1.0))

    q_conf = float(np.clip(2.0 * abs(p - 0.5), 0.0, 1.0))
    q = float(np.clip(0.4 * q_blur + 0.3 * q_bright + 0.3 * q_conf, 0.0, 1.0))

    # Original rule: flagged only if sustained/widespread condition is met
    verdict = "UNSAFE" if overall_flagged else "SAFE"
    elapsed = round(time.time() - t0, 3)

    return {
        "p": round(p, 4),
        "u": round(u, 4),
        "q": round(q, 4),
        "present": 1,
        "verdict": verdict,
        "overall_flagged": overall_flagged,
        "flag_reason": flag_reason,
        "max_consecutive": max_consecutive,
        "flagged_percent": flagged_percent,
        "time_seconds": elapsed,
        "frame_scores": frame_scores,
        "top_frames": top_frames,
        "sustained_detections": sustained_detections,
        "raw_signals": {
            "max_nsfw": round(max_nsfw, 4),
            "mean_nsfw": round(mean_nsfw, 4),
            "max_consecutive": max_consecutive,
            "flagged_percent": flagged_percent,
            "condition_a": condition_a,
            "condition_b": condition_b,
            "flag_reason": flag_reason,
            "mean_blur": round(mean_blur, 2),
            "mean_bright": round(mean_bright, 2),
            "q_blur": round(q_blur, 4),
            "q_bright": round(q_bright, 4),
            "q_conf": round(q_conf, 4),
            "total_frames_scored": len(frame_scores)
        }
    }
