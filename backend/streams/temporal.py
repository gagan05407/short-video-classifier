import os
import sys
import time
import math
import cv2
import numpy as np
import torch
from pathlib import Path

# Add backend and streams to path
BACKEND_DIR = Path(__file__).resolve().parent.parent
STREAMS_DIR = BACKEND_DIR / "streams"
for p in [str(BACKEND_DIR), str(STREAMS_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import config
from video_classifier import VideoClassifier, load_model, center_crop_square, get_transforms, extract_frames


def categorical_entropy(probs: np.ndarray) -> float:
    """Normalized categorical entropy H(P)/ln(K) in [0, 1]."""
    k = len(probs)
    if k <= 1:
        return 0.0
    probs = np.clip(probs, 1e-7, 1.0)
    probs = probs / np.sum(probs)
    h = -float(np.sum(probs * np.log(probs)))
    return float(np.clip(h / math.log(k), 0.0, 1.0))


def compute_frames_motion(frames: list) -> float:
    """Mean absolute pixel difference between consecutive frames."""
    if len(frames) < 2:
        return 0.0
    diffs = []
    for i in range(1, len(frames)):
        d = cv2.absdiff(frames[i], frames[i - 1])
        diffs.append(float(np.mean(d)))
    return float(np.mean(diffs)) if diffs else 0.0


def analyze_temporal_clip(video_path: str, model=None, classes=None, device=None,
                          window_seconds: int = 10, stride_seconds: int = 5,
                          aggregation: str = "max", corruption: str = None) -> dict:
    """
    Sliding window video classification using ResNeXt50+BiLSTM.
    Extracts 32 frames per 10s window (5s stride), predicts across 4 classes,
    and computes risk p = P(violence) + P(adult_content), uncertainty u, and quality q.
    """
    t0 = time.time()
    device = device or config.DEVICE
    classes = classes or config.CLASSES

    if model is None:
        model = load_model(str(config.TEMPORAL_MODEL_PATH), device=device)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {
            "p": 0.0, "u": 0.0, "q": 0.0, "present": 0,
            "verdict": "SAFE", "time_seconds": round(time.time() - t0, 3),
            "top_class": "safe", "per_window_scores": [],
            "raw_signals": {"error": "cannot open video"}
        }

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    duration = total_frames / fps if fps > 0 else 0.0
    num_frames = config.NUM_FRAMES
    frame_size = config.FRAME_SIZE

    # Determine window start times
    starts = []
    t = 0.0
    while t < duration:
        starts.append(t)
        t += stride_seconds

    if duration > window_seconds:
        last_start = max(0.0, duration - window_seconds)
        if starts and abs(starts[-1] - last_start) > 1.0:
            starts.append(last_start)

    window_indices = []
    all_needed = set()
    for start in starts:
        if start >= duration - 1.0 and len(starts) > 1:
            break
        s_idx = int(start * fps)
        e_idx = min(total_frames, s_idx + int(fps * window_seconds))
        e_idx = max(e_idx, s_idx + 1)
        indices = np.linspace(s_idx, e_idx - 1, num_frames, dtype=int)
        window_indices.append((start, indices))
        all_needed.update(indices)

    frame_dict = {}
    f_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if f_idx in all_needed:
            if corruption == "blur":
                frame = cv2.GaussianBlur(frame, (9, 9), 3.0)
            elif corruption == "dark":
                frame = cv2.convertScaleAbs(frame, alpha=0.35)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cropped = center_crop_square(rgb)
            resized = cv2.resize(cropped, frame_size, interpolation=cv2.INTER_LINEAR)
            frame_dict[f_idx] = resized
        f_idx += 1
    cap.release()

    if not window_indices:
        return {
            "p": 0.0, "u": 0.0, "q": 0.0, "present": 0,
            "verdict": "SAFE", "time_seconds": round(time.time() - t0, 3),
            "top_class": "safe", "per_window_scores": [],
            "raw_signals": {"error": "no windows extracted"}
        }

    _, val_tf = get_transforms()
    window_tensors = []
    window_motion_scores = []
    for start_sec, indices in window_indices:
        w_frames = []
        for idx in indices:
            if idx in frame_dict:
                w_frames.append(frame_dict[idx])
            else:
                w_frames.append(np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8))
        motion_val = compute_frames_motion([cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in w_frames])
        window_motion_scores.append(motion_val)
        tensor = torch.stack([val_tf(f) for f in w_frames])
        window_tensors.append(tensor)

    stacked_windows = torch.stack(window_tensors)
    all_raw_probs = []
    window_details = []
    model.eval()

    with torch.no_grad():
        batch_size = 4
        for b in range(0, len(stacked_windows), batch_size):
            batch = stacked_windows[b:b + batch_size].to(device)
            logits = model(batch)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            for p_row in probs:
                all_raw_probs.append(p_row)

    for w_idx, (start_sec, _) in enumerate(window_indices):
        p_row = all_raw_probs[w_idx]
        window_details.append({
            "window": w_idx + 1,
            "start_second": round(start_sec, 2),
            "probs": {classes[i]: round(float(p_row[i]), 4) for i in range(len(classes))}
        })

    stacked = np.array(all_raw_probs)
    if aggregation == "max":
        agg_raw = np.max(stacked, axis=0)
        agg_probs = agg_raw / np.sum(agg_raw)
    else:
        agg_probs = np.mean(stacked, axis=0)

    cls_to_idx = {cls: i for i, cls in enumerate(classes)}
    v_idx = cls_to_idx.get("violence", -1)
    a_idx = cls_to_idx.get("adult_content", -1)

    p_violence = float(agg_probs[v_idx]) if v_idx >= 0 else 0.0
    p_adult = float(agg_probs[a_idx]) if a_idx >= 0 else 0.0

    # Risk score: combined violence and adult content probabilities
    p = float(np.clip(p_violence + p_adult, 0.0, 1.0))
    u = categorical_entropy(agg_probs)

    # Quality indicators
    mean_motion = float(np.mean(window_motion_scores)) if window_motion_scores else 15.0
    q_motion = float(np.clip(mean_motion / 30.0, 0.0, 1.0))

    if len(all_raw_probs) > 1:
        per_window_p = [float(w[v_idx] + w[a_idx]) for w in all_raw_probs]
        std_p = float(np.std(per_window_p))
        q_consist = float(np.clip(1.0 - std_p, 0.0, 1.0))
    else:
        q_consist = 1.0

    q_model = float(np.clip(1.0 - u, 0.0, 1.0))
    q = float(np.clip(0.35 * q_motion + 0.35 * q_consist + 0.30 * q_model, 0.0, 1.0))

    # ── Temporal Window Persistence / Streak Tracking (Two-Tier) ──
    # Track per-window probabilities, class streaks, and peak severity
    window_violence_flags = []
    window_adult_flags = []
    window_harm_risks = []

    for p_row in all_raw_probs:
        w_top_idx = int(np.argmax(p_row))
        w_top_class = classes[w_top_idx]
        w_v = float(p_row[v_idx])
        w_a = float(p_row[a_idx])
        w_harm = w_v + w_a
        window_harm_risks.append(w_harm)

        # Flag window for specific harm category
        window_violence_flags.append((w_v >= 0.40 and w_v >= w_a) or (w_top_class == "violence"))
        window_adult_flags.append((w_a >= 0.40 and w_a >= w_v) or (w_top_class == "adult_content"))

    total_windows = len(all_raw_probs)
    max_peak_harm = max(window_harm_risks) if window_harm_risks else 0.0

    def calc_streak(flags):
        max_s = 0
        curr = 0
        for f in flags:
            if f:
                curr += 1
                if curr > max_s:
                    max_s = curr
            else:
                curr = 0
        return max_s

    max_streak_violence = calc_streak(window_violence_flags)
    max_streak_adult = calc_streak(window_adult_flags)
    max_consecutive_windows = max(max_streak_violence, max_streak_adult)

    count_violence = sum(1 for f in window_violence_flags if f)
    count_adult = sum(1 for f in window_adult_flags if f)
    flagged_windows_count = sum(1 for i in range(total_windows) if window_violence_flags[i] or window_adult_flags[i])
    flagged_windows_percent = round(flagged_windows_count / total_windows * 100, 1) if total_windows > 0 else 0.0

    # Two-Tier Decision:
    # Tier 1 (Severe Peak Bypass): If any single window has extreme harm >= 0.85, flag immediately!
    severe_peak = max_peak_harm >= 0.85

    # Tier 2 (Moderate Harm Persistence): For moderate risk, require >= 2 consecutive windows of the SAME class OR >= 30% coverage
    if total_windows <= 1:
        sustained_temporal = (flagged_windows_count >= 1) or severe_peak
    else:
        sustained_class_streak = (max_streak_violence >= 2) or (max_streak_adult >= 2)
        sustained_coverage = flagged_windows_percent >= 30.0
        sustained_temporal = severe_peak or sustained_class_streak or sustained_coverage

    top_idx = int(np.argmax(agg_probs))
    top_class = classes[top_idx]

    if sustained_temporal:
        verdict = "UNSAFE"
        if top_class in ["safe", "dance"]:
            top_class = "violence" if count_violence >= count_adult else "adult_content"
    else:
        verdict = "SAFE"
        if top_class in ["violence", "adult_content"]:
            d_idx = cls_to_idx.get("dance", -1)
            s_idx = cls_to_idx.get("safe", -1)
            p_dance = float(agg_probs[d_idx]) if d_idx >= 0 else 0.0
            p_safe = float(agg_probs[s_idx]) if s_idx >= 0 else 0.0
            top_class = "dance" if p_dance >= p_safe else "safe"
        mean_harm = float(np.mean([p_row[v_idx] + p_row[a_idx] for p_row in all_raw_probs]))
        p = min(p, mean_harm)

    elapsed = round(time.time() - t0, 3)

    return {
        "p": round(p, 4),
        "u": round(u, 4),
        "q": round(q, 4),
        "present": 1,
        "verdict": verdict,
        "top_class": top_class,
        "max_consecutive_windows": max_consecutive_windows,
        "flagged_windows_count": flagged_windows_count,
        "total_windows": total_windows,
        "sustained_temporal": sustained_temporal,
        "time_seconds": elapsed,
        "per_window_scores": window_details,
        "raw_signals": {
            "p_violence": round(p_violence, 4),
            "p_adult": round(p_adult, 4),
            "max_consecutive_windows": max_consecutive_windows,
            "flagged_windows_percent": flagged_windows_percent,
            "sustained_temporal": sustained_temporal,
            "mean_motion": round(mean_motion, 2),
            "q_motion": round(q_motion, 4),
            "q_consist": round(q_consist, 4),
            "q_model": round(q_model, 4),
            "num_windows": len(all_raw_probs),
            "agg_probs": {classes[i]: round(float(agg_probs[i]), 4) for i in range(len(classes))}
        }
    }
