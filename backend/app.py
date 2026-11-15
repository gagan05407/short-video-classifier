import os
import sys
import time
import uuid
import shutil
from pathlib import Path
from typing import Optional, Dict, Any
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd

# Add backend to sys.path
BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import config
from gate import LiveGateManager, ReliabilityGate
from streams.static import analyze_static_frames, load_nsfw_model
from streams.temporal import analyze_temporal_clip, load_model as load_temporal_model
from streams.speech import analyze_speech, load_speech_models

# Temporary uploads directory
TEMP_DIR = BACKEND_DIR / "temp_uploads"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Reliability-Aware Multimodal Video Moderation API")

# Enable CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global models and gate manager
live_gate = LiveGateManager()
models_cache: Dict[str, Any] = {}
memory_cache: Dict[str, Dict[str, Any]] = {}


def get_models():
    """Lazily load and cache base models on startup."""
    if "loaded" not in models_cache:
        print(f"Loading multimodal models on {config.DEVICE}...")
        models_cache["nsfw"] = load_nsfw_model(device=config.DEVICE)
        temp_model, _, _ = load_temporal_model(str(config.TEMPORAL_MODEL_PATH), device=config.DEVICE)
        models_cache["temporal"] = temp_model
        w_m, b_m, b_tok, r_m, r_tok = load_speech_models(device=config.DEVICE)
        models_cache["whisper"] = w_m
        models_cache["bert"] = b_m
        models_cache["bert_tok"] = b_tok
        models_cache["roberta"] = r_m
        models_cache["roberta_tok"] = r_tok
        models_cache["loaded"] = True
    return models_cache


class FeedbackRequest(BaseModel):
    analysis_id: str
    human_label: str


@app.get("/api/gate/status")
def get_gate_status():
    return live_gate.get_status()


@app.post("/api/gate/reset")
def reset_gate():
    return live_gate.reset_gate()


@app.get("/api/results/{table_id}")
def get_table_results(table_id: int):
    table_files = {
        1: config.TABLE_1_PATH,
        2: config.TABLE_2_PATH,
        3: config.TABLE_3_PATH,
        4: config.TABLE_4_PATH,
        5: config.TABLE_5_PATH,
    }
    path = table_files.get(table_id)
    if not path or not Path(path).exists():
        raise HTTPException(status_code=404, detail=f"Table {table_id} not found.")

    df = pd.read_csv(path)
    df = df.fillna("")
    return df.to_dict(orient="records")


@app.post("/api/analyze")
async def analyze_video(
    file: Optional[UploadFile] = File(None),
    video_path: Optional[str] = Form(None)
):
    t0 = time.time()
    models = get_models()

    # Determine input video path
    temp_file = None
    if file is not None:
        filename = f"{uuid.uuid4().hex}_{file.filename}"
        saved_path = TEMP_DIR / filename
        with open(saved_path, "wb") as f_out:
            shutil.copyfileobj(file.file, f_out)
        target_path = str(saved_path)
        temp_file = saved_path
        video_name = file.filename
    elif video_path:
        target_path = video_path
        video_name = Path(video_path).name
    else:
        raise HTTPException(status_code=400, detail="Must provide either a video file or video_path.")

    if not Path(target_path).exists():
        raise HTTPException(status_code=404, detail="Specified video file not found.")

    # 1. Static Stream
    res_s = analyze_static_frames(target_path, model=models["nsfw"], device=config.DEVICE)

    # 2. Temporal Stream
    res_t = analyze_temporal_clip(target_path, model=models["temporal"], device=config.DEVICE)

    # 3. Speech Stream
    res_sp = analyze_speech(
        target_path,
        whisper_model=models["whisper"],
        bert_model=models["bert"],
        bert_tokenizer=models["bert_tok"],
        roberta_model=models["roberta"],
        roberta_tokenizer=models["roberta_tok"],
        device=config.DEVICE
    )

    # Clean up uploaded temp file
    if temp_file and temp_file.exists():
        try:
            temp_file.unlink()
        except Exception:
            pass

    p_vals = [res_s["p"], res_t["p"], res_sp["p"]]
    u_vals = [res_s["u"], res_t["u"], res_sp["u"]]
    q_vals = [res_s["q"], res_t["q"], res_sp["q"]]
    mask_vals = [res_s["present"], res_t["present"], res_sp["present"]]

    # Run Live Gate Prediction
    pred = live_gate.predict_sample(p_vals, u_vals, q_vals, mask_vals, use_live=True)
    fused_score = pred["fused_score"]
    final_verdict = pred["verdict"]
    weights = pred["weights"]

    # Check stream disagreement with final verdict
    stream_disagreements = {}
    for name, res in [("Frames", res_s), ("Clip", res_t), ("Speech", res_sp)]:
        if final_verdict in ["SAFE", "UNSAFE"]:
            disagrees = res["verdict"] != final_verdict
        else:
            disagrees = False
        reason = ""
        if disagrees:
            if res["verdict"] == "UNSAFE":
                reason = f"{name} detected elevated risk ({res['p']:.1%}), but final verdict is {final_verdict}"
            else:
                reason = f"{name} detected low risk ({res['p']:.1%}), but final verdict is {final_verdict}"
        stream_disagreements[name] = {"disagrees": disagrees, "reason": reason}

    # Synthesize plain 1-sentence "why this verdict"
    if final_verdict == "SAFE":
        why = f"Video classified as SAFE (risk {fused_score:.1%}): no sustained visual or audio violations detected."
    elif final_verdict == "UNSAFE":
        reasons = []
        if res_s["p"] >= config.THRESH_STATIC:
            reasons.append(f"static frames ({res_s['p']:.1%})")
        if res_t["p"] >= config.THRESH_TEMPORAL:
            reasons.append(f"temporal motion ({res_t['p']:.1%}, {res_t['top_class']})")
        if res_sp["p"] >= config.THRESH_SPEECH and res_sp["present"] == 1:
            reasons.append(f"speech toxicity ({res_sp['p']:.1%})")
        joined = ", ".join(reasons) if reasons else "multimodal risk"
        why = f"Video flagged as UNSAFE (fused risk {fused_score:.1%}): driven primarily by {joined}."
    else:
        why = f"Video routed to REVIEW (risk {fused_score:.1%}): signals are ambiguous or conflicting between streams."

    # Cache sample inputs in memory for feedback fine-tuning
    analysis_id = str(uuid.uuid4())
    memory_cache[analysis_id] = {
        "video": video_name,
        "p": p_vals,
        "u": u_vals,
        "q": q_vals,
        "mask": mask_vals,
        "timestamp": time.time()
    }

    elapsed = round(time.time() - t0, 3)

    return {
        "analysis_id": analysis_id,
        "video_name": video_name,
        "verdict": final_verdict,
        "fused_score": fused_score,
        "weights": weights,
        "why_verdict": why,
        "latency": elapsed,
        "gate_status": live_gate.get_status(),
        "streams": {
            "static": {
                "name": "Frames",
                "verdict": res_s["verdict"],
                "score": res_s["p"],
                "weight": weights["static"],
                "quality": res_s["q"],
                "uncertainty": res_s["u"],
                "present": res_s["present"],
                "max_consecutive": res_s.get("max_consecutive", 0),
                "flagged_percent": res_s.get("flagged_percent", 0.0),
                "flag_reason": res_s.get("flag_reason", "clean"),
                "disagrees": stream_disagreements["Frames"]["disagrees"],
                "disagree_reason": stream_disagreements["Frames"]["reason"],
                "top_frames": res_s.get("top_frames", [])
            },
            "temporal": {
                "name": "Clip",
                "verdict": res_t["verdict"],
                "score": res_t["p"],
                "weight": weights["temporal"],
                "quality": res_t["q"],
                "uncertainty": res_t["u"],
                "top_class": res_t.get("top_class", "safe"),
                "max_consecutive_windows": res_t.get("max_consecutive_windows", 0),
                "total_windows": res_t.get("total_windows", 0),
                "sustained_temporal": res_t.get("sustained_temporal", False),
                "present": res_t["present"],
                "disagrees": stream_disagreements["Clip"]["disagrees"],
                "disagree_reason": stream_disagreements["Clip"]["reason"],
                "per_window_scores": res_t.get("per_window_scores", [])
            },
            "speech": {
                "name": "Speech",
                "verdict": "NOT USED" if res_sp["present"] == 0 else res_sp["verdict"],
                "score": res_sp["p"],
                "weight": weights["speech"],
                "quality": res_sp["q"],
                "uncertainty": res_sp["u"],
                "present": res_sp["present"],
                "transcript": res_sp.get("transcript", ""),
                "language": res_sp.get("language", "none"),
                "toxic_bert_categories": res_sp.get("toxic_bert_categories", {}),
                "roberta_offensive": res_sp.get("roberta_offensive", 0.0),
                "disagrees": stream_disagreements["Speech"]["disagrees"],
                "disagree_reason": stream_disagreements["Speech"]["reason"]
            }
        }
    }


@app.post("/api/feedback")
def submit_feedback(body: FeedbackRequest):
    analysis_id = body.analysis_id
    human_label = body.human_label.upper()

    if human_label not in ["SAFE", "UNSAFE"]:
        raise HTTPException(status_code=400, detail="human_label must be SAFE or UNSAFE.")

    sample = memory_cache.get(analysis_id)
    if not sample:
        raise HTTPException(status_code=404, detail="Analysis ID expired or not found in memory.")

    res = live_gate.update_feedback(
        video_name=sample["video"],
        sample_inputs=sample,
        human_label=human_label
    )
    return res

