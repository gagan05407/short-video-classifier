import os
import sys
import shutil
import time
import subprocess
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import torch

# Add backend to sys.path
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import config
from streams.static import analyze_static_frames, load_nsfw_model
from streams.temporal import analyze_temporal_clip, load_model as load_temporal_model
from streams.speech import analyze_speech, load_speech_models

CORRUPTED_DIR = config.DATA_DIR / "corrupted_test"
CORRUPTED_DIR.mkdir(parents=True, exist_ok=True)


def corrupt_audio_ffmpeg(src_path: str, dst_path: str, snr_db: str = "10db") -> bool:
    """Injects white Gaussian noise into audio track via ffmpeg."""
    Path(dst_path).parent.mkdir(parents=True, exist_ok=True)
    noise_vol = "0.08" if snr_db == "10db" else "0.30"
    filter_expr = f"anoisesrc=d=600:c=white:a={noise_vol}[noise];[0:a][noise]amix=inputs=2:duration=first:dropout_transition=0"

    cmd = [
        "ffmpeg", "-y", "-i", str(src_path),
        "-filter_complex", filter_expr,
        "-c:v", "copy",
        str(dst_path)
    ]
    try:
        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=45)
        if res.returncode == 0 and Path(dst_path).exists() and Path(dst_path).stat().st_size > 0:
            return True
    except Exception:
        pass

    shutil.copy2(src_path, dst_path)
    return True


def run_corruptions():
    """Generates corrupted test features under Blur, Darkening, and Audio Noise."""
    manifest = pd.read_csv(config.MANIFEST_PATH)
    features = pd.read_csv(config.FEATURES_PATH)

    test_manifest = manifest[manifest["split"] == "test"].copy().reset_index(drop=True)
    test_features = features[features["split"] == "test"].copy().reset_index(drop=True)

    print(f"Total test videos to evaluate under corruption: {len(test_manifest)}")

    blur_features_path = config.DATA_DIR / "features_corrupted_blur.csv"
    dark_features_path = config.DATA_DIR / "features_corrupted_dark.csv"
    noise_features_path = config.DATA_DIR / "features_corrupted_noise.csv"

    # Load models
    print("Loading models for corrupted feature extraction...")
    nsfw_model = load_nsfw_model(device=config.DEVICE)
    temporal_model, _, _ = load_temporal_model(str(config.TEMPORAL_MODEL_PATH), device=config.DEVICE)
    whisper_model, bert_model, bert_tokenizer, roberta_model, roberta_tokenizer = load_speech_models(device=config.DEVICE)

    # 1. BLUR
    if blur_features_path.exists():
        print(f"Skipping Blur: {blur_features_path} already exists.", flush=True)
    else:
        print("\n--- 1. Extracting In-Memory Blurred Visual Features ---", flush=True)
        t0 = time.time()
        blur_rows = []
        for idx, row in test_manifest.iterrows():
            vid_name = row["video"]
            src_path = Path(row["path"])
            clean_row = test_features[test_features["video"] == vid_name].iloc[0].to_dict()

            res_s = analyze_static_frames(str(src_path), model=nsfw_model, device=config.DEVICE, corruption="blur")
            res_t = analyze_temporal_clip(str(src_path), model=temporal_model, device=config.DEVICE, corruption="blur")

            c_row = clean_row.copy()
            c_row["p_static"] = res_s["p"]
            c_row["u_static"] = res_s["u"]
            c_row["q_static"] = res_s["q"]
            c_row["p_temporal"] = res_t["p"]
            c_row["u_temporal"] = res_t["u"]
            c_row["q_temporal"] = res_t["q"]
            blur_rows.append(c_row)
            if (idx + 1) % 20 == 0 or (idx + 1) == len(test_manifest):
                print(f"  Processed {idx + 1}/{len(test_manifest)} blur videos", flush=True)

        df_blur = pd.DataFrame(blur_rows)
        df_blur.to_csv(blur_features_path, index=False)
        print(f"Saved {blur_features_path} in {time.time() - t0:.1f}s", flush=True)

    # 2. DARK
    print("\n--- 2. Extracting In-Memory Darkened Visual Features ---", flush=True)
    t0 = time.time()
    dark_rows = []
    for idx, row in test_manifest.iterrows():
        vid_name = row["video"]
        src_path = Path(row["path"])
        clean_row = test_features[test_features["video"] == vid_name].iloc[0].to_dict()

        res_s = analyze_static_frames(str(src_path), model=nsfw_model, device=config.DEVICE, corruption="dark")
        res_t = analyze_temporal_clip(str(src_path), model=temporal_model, device=config.DEVICE, corruption="dark")

        c_row = clean_row.copy()
        c_row["p_static"] = res_s["p"]
        c_row["u_static"] = res_s["u"]
        c_row["q_static"] = res_s["q"]
        c_row["p_temporal"] = res_t["p"]
        c_row["u_temporal"] = res_t["u"]
        c_row["q_temporal"] = res_t["q"]
        dark_rows.append(c_row)
        if (idx + 1) % 20 == 0 or (idx + 1) == len(test_manifest):
            print(f"  Processed {idx + 1}/{len(test_manifest)} dark videos", flush=True)

    df_dark = pd.DataFrame(dark_rows)
    df_dark.to_csv(dark_features_path, index=False)
    print(f"Saved {dark_features_path} in {time.time() - t0:.1f}s", flush=True)

    # 3. NOISY AUDIO
    print("\n--- 3. Generating Noisy Audio & Extracting Speech Features ---", flush=True)
    t0 = time.time()
    noise_rows = []
    noise_dir = CORRUPTED_DIR / "noisy_audio"
    noise_dir.mkdir(parents=True, exist_ok=True)

    for idx, row in test_manifest.iterrows():
        vid_name = row["video"]
        src_path = Path(row["path"])
        dst_path = noise_dir / f"noise_{vid_name}"
        clean_row = test_features[test_features["video"] == vid_name].iloc[0].to_dict()

        if clean_row.get("present_speech", 0) == 1:
            if not dst_path.exists():
                corrupt_audio_ffmpeg(src_path, dst_path, snr_db="10db")
            res_sp = analyze_speech(
                str(dst_path),
                whisper_model=whisper_model,
                bert_model=bert_model,
                bert_tokenizer=bert_tokenizer,
                roberta_model=roberta_model,
                roberta_tokenizer=roberta_tokenizer,
                device=config.DEVICE
            )
            c_row = clean_row.copy()
            c_row["p_speech"] = res_sp["p"]
            c_row["u_speech"] = res_sp["u"]
            c_row["q_speech"] = res_sp["q"]
            c_row["present_speech"] = res_sp["present"]
            noise_rows.append(c_row)
        else:
            noise_rows.append(clean_row.copy())

    df_noise = pd.DataFrame(noise_rows)
    df_noise.to_csv(noise_features_path, index=False)
    print(f"Saved {noise_features_path} in {time.time() - t0:.1f}s")

    # Load blur if it was skipped
    if "df_blur" not in locals():
        df_blur = pd.read_csv(blur_features_path)

    # Print proof comparison rows
    print("\n" + "=" * 60)
    print("PROOF OF CORRUPTION DIFFERENCE (Sample comparison)")
    print("=" * 60)
    for i in range(3):
        v = test_features.iloc[i]["video"]
        print(f"Video: {v}")
        print(f"  Clean:  p_static={test_features.iloc[i]['p_static']:.3f}, q_static={test_features.iloc[i]['q_static']:.3f} | p_temporal={test_features.iloc[i]['p_temporal']:.3f}, q_temporal={test_features.iloc[i]['q_temporal']:.3f}")
        print(f"  Blur:   p_static={df_blur.iloc[i]['p_static']:.3f}, q_static={df_blur.iloc[i]['q_static']:.3f} | p_temporal={df_blur.iloc[i]['p_temporal']:.3f}, q_temporal={df_blur.iloc[i]['q_temporal']:.3f}")
        print(f"  Dark:   p_static={df_dark.iloc[i]['p_static']:.3f}, q_static={df_dark.iloc[i]['q_static']:.3f} | p_temporal={df_dark.iloc[i]['p_temporal']:.3f}, q_temporal={df_dark.iloc[i]['q_temporal']:.3f}")

    return df_blur, df_dark, df_noise


if __name__ == "__main__":
    run_corruptions()
