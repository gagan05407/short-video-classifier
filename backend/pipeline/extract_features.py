import sys
import os
import time
from pathlib import Path
import pandas as pd
import numpy as np
import torch

# Add backend to sys.path
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import config
from streams.static import analyze_static_frames, load_nsfw_model
from streams.temporal import analyze_temporal_clip, load_model as load_temporal_model
from streams.speech import analyze_speech, load_speech_models


def load_all_models():
    print(f"Loading models on {config.DEVICE}...")
    nsfw_model = load_nsfw_model(device=config.DEVICE)
    temporal_model, _, _ = load_temporal_model(str(config.TEMPORAL_MODEL_PATH), device=config.DEVICE)
    whisper_model, bert_model, bert_tokenizer, roberta_model, roberta_tokenizer = load_speech_models(device=config.DEVICE)
    return nsfw_model, temporal_model, whisper_model, bert_model, bert_tokenizer, roberta_model, roberta_tokenizer


def run_feature_extraction(limit: int = None):
    print("=" * 60)
    print("Multi-modal Feature Extraction")
    print("=" * 60)

    if not config.MANIFEST_PATH.exists():
        raise FileNotFoundError(f"manifest.csv not found at {config.MANIFEST_PATH}")

    manifest_df = pd.read_csv(config.MANIFEST_PATH)
    total_videos = len(manifest_df)
    print(f"Total videos in manifest: {total_videos}")

    # Check already extracted
    extracted_names = set()
    if config.FEATURES_PATH.exists():
        try:
            feat_df = pd.read_csv(config.FEATURES_PATH)
            extracted_names = set(feat_df["video"].tolist())
            print(f"Resuming: {len(extracted_names)} videos already extracted.")
        except Exception:
            extracted_names = set()

    models = load_all_models()
    nsfw_m, temp_m, whisp_m, bert_m, bert_tok, rob_m, rob_tok = models

    rows_to_save = []
    processed = 0

    for idx, row in manifest_df.iterrows():
        vid_name = row["video"]
        if vid_name in extracted_names:
            continue

        vid_path = Path(row["path"])
        if not vid_path.exists():
            print(f"Warning: Video not found at {vid_path}. Skipping.")
            continue

        try:
            t0 = time.time()
            res_s = analyze_static_frames(str(vid_path), model=nsfw_m, device=config.DEVICE)
            res_t = analyze_temporal_clip(str(vid_path), model=temp_m, device=config.DEVICE)
            res_sp = analyze_speech(
                str(vid_path),
                whisper_model=whisp_m,
                bert_model=bert_m,
                bert_tokenizer=bert_tok,
                roberta_model=rob_m,
                roberta_tokenizer=rob_tok,
                device=config.DEVICE
            )

            entry = {
                "video": vid_name,
                "split": row.get("split", "train"),
                "class": row.get("class", "safe"),
                "binary": int(row.get("binary", 0)),
                "reason": row.get("reason", "None"),
                "p_static": res_s["p"], "u_static": res_s["u"], "q_static": res_s["q"], "present_static": res_s["present"],
                "p_temporal": res_t["p"], "u_temporal": res_t["u"], "q_temporal": res_t["q"], "present_temporal": res_t["present"],
                "p_speech": res_sp["p"], "u_speech": res_sp["u"], "q_speech": res_sp["q"], "present_speech": res_sp["present"],
                "total_time": round(time.time() - t0, 3)
            }
            rows_to_save.append(entry)
            processed += 1
            print(f"[{processed}/{total_videos - len(extracted_names)}] Extracted {vid_name} ({entry['total_time']}s)")

            # Periodic write to disk
            if len(rows_to_save) >= 10:
                df_chunk = pd.DataFrame(rows_to_save)
                header = not config.FEATURES_PATH.exists()
                df_chunk.to_csv(config.FEATURES_PATH, mode="a", header=header, index=False)
                rows_to_save = []

            if limit and processed >= limit:
                break
        except Exception as e:
            print(f"Error on {vid_name}: {e}")

    if rows_to_save:
        df_chunk = pd.DataFrame(rows_to_save)
        header = not config.FEATURES_PATH.exists()
        df_chunk.to_csv(config.FEATURES_PATH, mode="a", header=header, index=False)

    print(f"Feature extraction complete. Total saved: {len(pd.read_csv(config.FEATURES_PATH))}")


if __name__ == "__main__":
    run_feature_extraction()

