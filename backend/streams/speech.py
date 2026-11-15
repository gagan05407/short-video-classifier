import os
import sys
import time
import math
import numpy as np
import torch
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


def load_speech_models(device: str = None):
    """Loads Whisper, Toxic-BERT, and Twitter-RoBERTa."""
    device = device or config.DEVICE
    import whisper
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    print("Loading Whisper model (base)...")
    whisper_model = whisper.load_model("base", device=device)

    print("Loading Toxic-BERT model...")
    bert_tokenizer = AutoTokenizer.from_pretrained("unitary/toxic-bert")
    bert_model = AutoModelForSequenceClassification.from_pretrained("unitary/toxic-bert").to(device)
    bert_model.eval()

    print("Loading Twitter-RoBERTa offensive model...")
    roberta_tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-offensive")
    roberta_model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-offensive").to(device)
    roberta_model.eval()

    return whisper_model, bert_model, bert_tokenizer, roberta_model, roberta_tokenizer


def analyze_speech(video_path: str, whisper_model=None, bert_model=None, bert_tokenizer=None,
                   roberta_model=None, roberta_tokenizer=None, device=None) -> dict:
    """
    Transcribes audio track using Whisper and detects toxic/offensive language.
    If no audio or fewer than 2 meaningful words, returns present=0 and weight 0.
    """
    t0 = time.time()
    device = device or config.DEVICE

    transcript = ""
    language = "none"
    avg_logprob = -1.0
    no_speech_prob = 1.0
    speech_duration = 0.0

    try:
        if whisper_model is not None:
            res = whisper_model.transcribe(str(video_path))
            transcript = res.get("text", "").strip()
            language = res.get("language", "unknown")
            segments = res.get("segments", [])
            if segments:
                avg_logprob = float(np.mean([s.get("avg_logprob", -1.0) for s in segments]))
                no_speech_prob = float(np.mean([s.get("no_speech_prob", 0.0) for s in segments]))
                speech_duration = sum(s.get("end", 0.0) - s.get("start", 0.0) for s in segments)
    except Exception:
        pass

    words = transcript.split()
    if len(words) < 2 or no_speech_prob > 0.85 or not transcript:
        return {
            "p": 0.0, "u": 0.0, "q": 0.0, "present": 0,
            "verdict": "SAFE", "time_seconds": round(time.time() - t0, 3),
            "transcript": transcript or "No speech detected",
            "language": language,
            "toxic_bert_categories": {},
            "roberta_offensive": 0.0,
            "raw_signals": {
                "words_count": len(words),
                "avg_logprob": round(avg_logprob, 3),
                "no_speech_prob": round(no_speech_prob, 3),
                "speech_duration": round(speech_duration, 2),
                "note": "No audio or no speech detected (not used in fusion)"
            }
        }

    # Toxic-BERT evaluation
    toxic_bert_score = 0.0
    scores_b = []
    if bert_model is not None and bert_tokenizer is not None:
        inputs_b = bert_tokenizer(transcript, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            out_b = bert_model(**inputs_b)
            scores_b = torch.sigmoid(out_b.logits)[0].cpu().numpy()
            toxic_bert_score = float(np.max(scores_b))

    # Twitter-RoBERTa offensive evaluation
    roberta_score = 0.0
    if roberta_model is not None and roberta_tokenizer is not None:
        inputs_r = roberta_tokenizer(transcript, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            out_r = roberta_model(**inputs_r)
            scores_r = torch.softmax(out_r.logits, dim=1)[0].cpu().numpy()
            roberta_score = float(scores_r[1]) if len(scores_r) > 1 else float(scores_r[0])

    TOXIC_LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    toxic_bert_categories = {
        TOXIC_LABELS[i]: round(float(scores_b[i]), 4)
        for i in range(min(len(TOXIC_LABELS), len(scores_b)))
    }

    # Combined speech risk
    p = float(np.clip(max(toxic_bert_score, roberta_score), 0.0, 1.0))
    u = binary_entropy(p)

    # Acoustic quality indicators
    q_logprob = float(np.clip((avg_logprob + 2.0) / 2.0, 0.0, 1.0))
    q_speech = float(np.clip(1.0 - no_speech_prob, 0.0, 1.0))
    q_len = float(np.clip(len(words) / 15.0, 0.0, 1.0))
    q = float(np.clip(0.4 * q_logprob + 0.4 * q_speech + 0.2 * q_len, 0.0, 1.0))

    verdict = "UNSAFE" if p >= config.THRESH_SPEECH else "SAFE"
    elapsed = round(time.time() - t0, 3)

    return {
        "p": round(p, 4),
        "u": round(u, 4),
        "q": round(q, 4),
        "present": 1,
        "verdict": verdict,
        "transcript": transcript,
        "language": language,
        "toxic_bert_categories": toxic_bert_categories,
        "roberta_offensive": round(float(roberta_score), 4),
        "time_seconds": elapsed,
        "raw_signals": {
            "toxic_bert_score": round(toxic_bert_score, 4),
            "roberta_score": round(roberta_score, 4),
            "avg_logprob": round(avg_logprob, 3),
            "no_speech_prob": round(no_speech_prob, 3),
            "q_logprob": round(q_logprob, 4),
            "q_speech": round(q_speech, 4),
            "q_len": round(q_len, 4),
            "words_count": len(words),
            "speech_duration": round(speech_duration, 2)
        }
    }

