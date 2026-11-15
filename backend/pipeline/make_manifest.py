"""
build_speech_dataset.py - Part 1: Synthetic Speech Dataset Generation.

1. Downloads/extracts 60 toxic and 60 neutral English sentences from HateXplain (MIT License),
   a public benchmark that toxic-bert and twitter-roberta were NOT trained on.
   Saves to backend/ml/data/speech_texts.csv.
2. Synthesizes offline English speech using pyttsx3 into backend/ml/data/tts_wavs/.
3. Uses ffmpeg to mux audio into copies of 120 existing videos (72 train, 24 val, 24 test),
   producing all 4 combinations in every split:
     - Unsafe visual + Toxic speech (binary=1)
     - Unsafe visual + Neutral speech (binary=1)
     - Safe visual + Toxic speech (binary=1)
     - Safe visual + Neutral speech (binary=0)
4. Saves new videos to backend/ml/data/speech_data/.
5. Updates backend/ml/data/manifest.csv with columns:
   [video, path, class, binary, reason, split, source_video, speech_label, synthetic].
"""

import sys
import os
import json
import subprocess
import time
import urllib.request
from pathlib import Path
import pandas as pd
import numpy as np

BASE_ML_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_ML_DIR))

import config

DATA_DIR = config.DATA_DIR
SPEECH_DATA_DIR = DATA_DIR / "speech_data"
TTS_WAV_DIR = DATA_DIR / "tts_wavs"
SPEECH_TEXTS_CSV = DATA_DIR / "speech_texts.csv"


def download_hatexplain_texts(num_toxic=60, num_neutral=60):
    """
    Downloads HateXplain dataset (MIT License) and extracts clean English sentences.
    HateXplain was NOT used to train toxic-bert (Jigsaw) or twitter-roberta (TweetEval).
    """
    if SPEECH_TEXTS_CSV.exists():
        print(f"speech_texts.csv already exists at {SPEECH_TEXTS_CSV}")
        df = pd.read_csv(SPEECH_TEXTS_CSV)
        if len(df[df["label"] == "toxic"]) >= num_toxic and len(df[df["label"] == "neutral"]) >= num_neutral:
            print(f"Loaded existing {len(df)} speech texts from CSV.")
            return df

    print("Downloading HateXplain dataset from GitHub (MIT License)...")
    url = "https://raw.githubusercontent.com/hate-alert/HateXplain/master/Data/dataset.json"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw_data = json.loads(resp.read().decode("utf-8"))

    print(f"Fetched {len(raw_data)} posts from HateXplain. Filtering candidates (8 to 22 words)...")

    toxic_candidates = []
    neutral_candidates = []

    for post_id, post_info in raw_data.items():
        tokens = post_info.get("post_tokens", [])
        text = " ".join(tokens).strip()

        # Word count suitable for 3 to 10 seconds speech (~8 to 22 words)
        words = text.split()
        if not (8 <= len(words) <= 22):
            continue

        # Clean non-standard tokens
        if "<user>" in text or "<url>" in text or "http" in text:
            continue

        # Get majority label from annotators
        labels = [a["label"] for a in post_info.get("annotators", [])]
        if not labels:
            continue
        
        # Count majority
        h_count = labels.count("hatespeech") + labels.count("offensive")
        n_count = labels.count("normal")

        # Capitalize and clean punctuation
        cleaned = text.capitalize()
        if not cleaned.endswith((".", "!", "?")):
            cleaned += "."

        # Check printable ASCII
        if not all(32 <= ord(c) < 127 for c in cleaned):
            continue

        if h_count >= 2 and len(toxic_candidates) < num_toxic:
            if cleaned not in toxic_candidates:
                toxic_candidates.append(cleaned)
        elif n_count >= 2 and len(neutral_candidates) < num_neutral:
            if cleaned not in neutral_candidates:
                neutral_candidates.append(cleaned)

        if len(toxic_candidates) >= num_toxic and len(neutral_candidates) >= num_neutral:
            break

    print(f"Selected {len(toxic_candidates)} toxic and {len(neutral_candidates)} neutral sentences.")

    rows = []
    for t in toxic_candidates[:num_toxic]:
        rows.append({"text": t, "label": "toxic", "source": "HateXplain (MIT License)"})
    for n in neutral_candidates[:num_neutral]:
        rows.append({"text": n, "label": "neutral", "source": "HateXplain (MIT License)"})

    df_texts = pd.DataFrame(rows)
    df_texts.to_csv(SPEECH_TEXTS_CSV, index=False)
    print(f"Saved {len(df_texts)} texts to {SPEECH_TEXTS_CSV}")
    return df_texts


def synthesize_speech(df_texts):
    """
    Synthesizes WAV audio files using Windows native SAPI via win32com.
    Ultra-fast and 100% reliable offline TTS.
    """
    import win32com.client

    TTS_WAV_DIR.mkdir(parents=True, exist_ok=True)
    speaker = win32com.client.Dispatch("SAPI.SpVoice")
    speaker.Rate = 0  # Standard rate

    wav_records = []
    print(f"Synthesizing {len(df_texts)} audio files with native Windows SAPI TTS...")

    for idx, row in df_texts.iterrows():
        label = row["label"]
        txt = row["text"]
        wav_name = f"tts_{label}_{idx+1:03d}.wav"
        wav_path = TTS_WAV_DIR / wav_name

        if not wav_path.exists():
            filestream = win32com.client.Dispatch("SAPI.SpFileStream")
            filestream.Open(str(wav_path.resolve()), 3, False)  # 3 = SSFMCreateForWrite
            speaker.AudioOutputStream = filestream
            speaker.Speak(txt)
            filestream.Close()

        wav_records.append({
            "text": txt,
            "label": label,
            "wav_path": str(wav_path.resolve()),
            "wav_name": wav_name
        })

    print(f"Synthesized {len(wav_records)} WAV files in {TTS_WAV_DIR}")
    return wav_records


import shutil

def get_ffmpeg_binary():
    bin_path = shutil.which("ffmpeg")
    if bin_path:
        return bin_path
    fallback = Path(r"C:\Users\Gagan Gaur\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.2-full_build\bin\ffmpeg.exe")
    if fallback.exists():
        return str(fallback)
    return "ffmpeg"

def mux_video_and_audio(video_in: str, audio_in: str, video_out: str) -> bool:
    """
    Uses ffmpeg to mux audio into video:
    - If audio longer than video: trims to video duration with -shortest.
    - If audio shorter than video: pads audio with silence using apad filter.
    - Copies video stream directly (-c:v copy) to preserve original video quality and speed up processing.
    """
    ffmpeg_bin = get_ffmpeg_binary()
    cmd = [
        ffmpeg_bin, "-y",
        "-i", str(video_in),
        "-i", str(audio_in),
        "-filter_complex", "[1:a]apad[a]",
        "-map", "0:v",
        "-map", "[a]",
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        str(video_out)
    ]
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg mux error for {video_in}: {e.stderr.decode('utf-8', errors='ignore')[:300]}")
        return False


def create_synthetic_speech_dataset():
    """
    Main pipeline for Part 1:
    - Prepares texts & audio
    - Selects 120 stratified videos from manifest.csv
    - Muxes audio into copies in speech_data/
    - Updates manifest.csv with source_video, speech_label, synthetic
    """
    print("=" * 60)
    print("PART 1: Creating Synthetic Multimodal Speech Dataset")
    print("=" * 60)

    # 1. Texts & Audio
    df_texts = download_hatexplain_texts(num_toxic=60, num_neutral=60)
    wav_records = synthesize_speech(df_texts)

    toxic_wavs = [r for r in wav_records if r["label"] == "toxic"]
    neutral_wavs = [r for r in wav_records if r["label"] == "neutral"]

    # 2. Read existing manifest
    if not config.MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest not found at {config.MANIFEST_PATH}")

    orig_manifest = pd.read_csv(config.MANIFEST_PATH)
    # Ensure backward-compatible columns exist on original rows
    if "source_video" not in orig_manifest.columns:
        orig_manifest["source_video"] = orig_manifest["video"]
    if "speech_label" not in orig_manifest.columns:
        orig_manifest["speech_label"] = "none"
    if "synthetic" not in orig_manifest.columns:
        orig_manifest["synthetic"] = 0

    # Filter to original non-synthetic videos
    base_df = orig_manifest[orig_manifest["synthetic"] == 0].copy()

    # 3. Stratified selection of 120 videos across splits and classes
    # Train: 72 (18 per class: 9 toxic, 9 neutral)
    # Val:   24 (6 per class: 3 toxic, 3 neutral)
    # Test:  24 (6 per class: 3 toxic, 3 neutral)
    split_targets = {
        "train": {"per_class": 18, "toxic_per_class": 9},
        "val":   {"per_class": 6,  "toxic_per_class": 3},
        "test":  {"per_class": 6,  "toxic_per_class": 3},
    }

    SPEECH_DATA_DIR.mkdir(parents=True, exist_ok=True)

    toxic_idx = 0
    neutral_idx = 0
    new_records = []

    np.random.seed(42)

    for split_name, cfg in split_targets.items():
        n_per_class = cfg["per_class"]
        n_toxic_per_class = cfg["toxic_per_class"]
        n_neutral_per_class = n_per_class - n_toxic_per_class

        for cls_name in ["adult_content", "violence", "dance", "safe"]:
            sub_df = base_df[(base_df["split"] == split_name) & (base_df["class"] == cls_name)]
            selected = sub_df.sample(n=n_per_class, random_state=42).reset_index(drop=True)

            # First half gets toxic speech, second half gets neutral speech
            for i, row in selected.iterrows():
                is_toxic = (i < n_toxic_per_class)
                speech_lbl = "toxic" if is_toxic else "neutral"

                if is_toxic:
                    wav_item = toxic_wavs[toxic_idx % len(toxic_wavs)]
                    toxic_idx += 1
                else:
                    wav_item = neutral_wavs[neutral_idx % len(neutral_wavs)]
                    neutral_idx += 1

                # Formulate target filename
                orig_stem = Path(row["video"]).stem
                target_video_name = f"{orig_stem}_tts_{speech_lbl}.mp4"
                target_video_path = SPEECH_DATA_DIR / target_video_name

                # Mux video with audio
                success = mux_video_and_audio(row["path"], wav_item["wav_path"], str(target_video_path))
                if not success:
                    print(f"Skipping failed mux: {target_video_name}")
                    continue

                # Determine binary label & reasons
                # binary = 1 if visual is unsafe (adult_content or violence) OR speech is toxic
                is_visual_unsafe = (row["class"] in ["adult_content", "violence"])
                binary_val = 1 if (is_visual_unsafe or is_toxic) else 0

                reasons = []
                if row["class"] == "adult_content":
                    reasons.append("Sexual_NSFW")
                elif row["class"] == "violence":
                    reasons.append("Violence")
                
                if is_toxic:
                    reasons.append("Toxic_language")

                if not reasons:
                    reasons.append("Safe_general")

                reason_str = "; ".join(reasons)

                new_records.append({
                    "video": target_video_name,
                    "path": str(target_video_path.resolve()),
                    "class": row["class"],
                    "binary": binary_val,
                    "reason": reason_str,
                    "split": row["split"],  # EXACT SAME SPLIT AS SOURCE VIDEO
                    "source_video": row["video"],
                    "speech_label": speech_lbl,
                    "synthetic": 1
                })

    print(f"\nSuccessfully generated {len(new_records)} synthetic speech video copies!")

    # 4. Update manifest.csv
    new_df = pd.DataFrame(new_records)
    # Combine original non-synthetic with new synthetic records
    combined_manifest = pd.concat([orig_manifest[orig_manifest["synthetic"] == 0], new_df], ignore_index=True)
    combined_manifest.to_csv(config.MANIFEST_PATH, index=False)
    print(f"Updated {config.MANIFEST_PATH} with {len(combined_manifest)} total records ({len(orig_manifest[orig_manifest['synthetic'] == 0])} original + {len(new_df)} synthetic).")

    # 5. Report counts per split & per combination
    print("\n--- SYNTHETIC DATASET DISTRIBUTION REPORT ---")
    print("Counts by Split:")
    print(new_df["split"].value_counts())

    print("\nCounts by Class & Speech Label:")
    print(pd.crosstab(new_df["class"], new_df["speech_label"]))

    print("\nCounts by 4 Combinations (Visual Safe/Unsafe x Speech Safe/Toxic):")
    combos = []
    for _, r in new_df.iterrows():
        v_stat = "Unsafe Visual" if r["class"] in ["adult_content", "violence"] else "Safe Visual"
        s_stat = "Toxic Speech" if r["speech_label"] == "toxic" else "Neutral Speech"
        combos.append(f"{v_stat} + {s_stat}")
    combo_series = pd.Series(combos, name="Combination")
    print(pd.crosstab(new_df["split"], combo_series))

    print("=" * 60)
    return new_df


if __name__ == "__main__":
    create_synthetic_speech_dataset()
