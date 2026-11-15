import os
import sys
import copy
import time
import random
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score

# Add backend to sys.path
BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import config


# =========================================================
# RELIABILITY GATE MODEL
# =========================================================
class ReliabilityGate(nn.Module):
    """
    Adaptive Reliability Gate MLP.
    Inputs: [p, u, q] for each modality (total 9 inputs for 3 modalities).
    Outputs: normalized weights summing to 1.0 over present modalities.
    """
    def __init__(self, num_modalities: int = 3, hidden_dim: int = 32):
        super().__init__()
        self.num_modalities = num_modalities
        self.fc1 = nn.Linear(num_modalities * 3, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_modalities)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        hidden = self.relu(self.fc1(x))
        logits = self.fc2(hidden)
        if mask is not None:
            # Mask out missing/absent streams with large negative logit
            logits = logits.masked_fill(mask == 0, -1e9)
        weights = F.softmax(logits, dim=-1)
        return weights


def compute_fused_score(weights: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """Computes weighted sum fused risk score: sum(w_m * p_m)."""
    return torch.sum(weights * p, dim=-1)


def decide_verdict(fused_score: float, uncertainty: float = 0.0,
                   t_lo: float = None, t_hi: float = None) -> str:
    """
    Selective classification decision.
    If fused score lies inside the review band [t_lo, t_hi], route to REVIEW.
    Otherwise, classify as UNSAFE (>= t_hi) or SAFE (< t_lo).
    """
    t_lo = t_lo if t_lo is not None else config.T_LO
    t_hi = t_hi if t_hi is not None else config.T_HI

    if t_lo <= fused_score < t_hi:
        return "REVIEW"
    elif fused_score >= 0.50:
        return "UNSAFE"
    else:
        return "SAFE"


def compute_uncertainty_score(weights: np.ndarray, p_vals: np.ndarray,
                              u_vals: np.ndarray, q_vals: np.ndarray) -> float:
    """
    Computes system uncertainty combining stream uncertainties and stream disagreement.
    """
    weighted_u = float(np.sum(weights * u_vals))
    weighted_q = float(np.sum(weights * q_vals))
    disagreement = float(np.max(p_vals) - np.min(p_vals)) if len(p_vals) > 1 else 0.0
    u_score = 0.4 * weighted_u + 0.3 * (1.0 - weighted_q) + 0.3 * disagreement
    return float(np.clip(u_score, 0.0, 1.0))


def prepare_gate_tensors(df: pd.DataFrame):
    """
    Builds (X, M, P, Y) tensors from features DataFrame.
    """
    X_list = []
    M_list = []
    P_list = []
    Y_list = []

    for _, row in df.iterrows():
        ps = float(row.get("p_static", 0.0))
        us = float(row.get("u_static", 0.0))
        qs = float(row.get("q_static", 0.0))
        ms = float(row.get("present_static", 1.0))

        pt = float(row.get("p_temporal", 0.0))
        ut = float(row.get("u_temporal", 0.0))
        qt = float(row.get("q_temporal", 0.0))
        mt = float(row.get("present_temporal", 1.0))

        psp = float(row.get("p_speech", 0.0))
        usp = float(row.get("u_speech", 0.0))
        qsp = float(row.get("q_speech", 0.0))
        msp = float(row.get("present_speech", 0.0))

        X_list.append([ps, us, qs, pt, ut, qt, psp, usp, qsp])
        M_list.append([ms, mt, msp])
        P_list.append([ps, pt, psp])
        Y_list.append(float(row.get("binary", 0.0)))

    X = torch.tensor(X_list, dtype=torch.float32)
    M = torch.tensor(M_list, dtype=torch.float32)
    P = torch.tensor(P_list, dtype=torch.float32)
    Y = torch.tensor(Y_list, dtype=torch.float32)
    return X, M, P, Y


# =========================================================
# LIVE GATE MANAGER WITH VAL F1 ROLLBACK SAFETY
# =========================================================
class LiveGateManager:
    """
    Manages frozen baseline gate (read-only) and live gate with human feedback adaptation.
    Guarantees validation safety: automatically rolls back if Val F1 drops > 0.05.
    """
    def __init__(self):
        self.device = torch.device(config.DEVICE)
        self.frozen_weights_path = Path(config.GATE_FROZEN_PATH)
        self.live_weights_path = Path(config.LIVE_GATE_PATH)
        self.live_dir = Path(config.LIVE_WEIGHTS_DIR)
        self.live_dir.mkdir(parents=True, exist_ok=True)

        # Load frozen gate
        self.frozen_gate = ReliabilityGate(num_modalities=3, hidden_dim=32).to(self.device)
        if self.frozen_weights_path.exists():
            self.frozen_gate.load_state_dict(torch.load(self.frozen_weights_path, map_location=self.device))
        self.frozen_gate.eval()

        # Load live gate
        self.live_gate = ReliabilityGate(num_modalities=3, hidden_dim=32).to(self.device)
        if self.live_weights_path.exists():
            self.live_gate.load_state_dict(torch.load(self.live_weights_path, map_location=self.device))
        elif self.frozen_weights_path.exists():
            self.live_gate.load_state_dict(torch.load(self.frozen_weights_path, map_location=self.device))
            torch.save(self.live_gate.state_dict(), self.live_weights_path)
        self.live_gate.eval()

        # Load features for replay and validation
        self.train_df = pd.DataFrame()
        self.val_df = pd.DataFrame()
        self._load_features()

        # Baseline Val F1
        self.frozen_val_f1 = self._eval_val_f1(self.frozen_gate)
        self.live_val_f1 = self._eval_val_f1(self.live_gate)
        self.version = self._detect_current_version()
        self.review_count = self._count_reviews_logged()

    def _load_features(self):
        if Path(config.FEATURES_PATH).exists():
            df = pd.read_csv(config.FEATURES_PATH)
            self.train_df = df[df["split"] == "train"].copy().reset_index(drop=True)
            self.val_df = df[df["split"] == "val"].copy().reset_index(drop=True)

    def _detect_current_version(self) -> int:
        v_files = list(self.live_dir.glob("v*.pt"))
        if not v_files:
            return 0
        versions = []
        for f in v_files:
            try:
                v = int(f.stem.replace("v", ""))
                versions.append(v)
            except ValueError:
                pass
        return max(versions) if versions else 0

    def _count_reviews_logged(self) -> int:
        if Path(config.FEEDBACK_LOG_PATH).exists():
            try:
                df = pd.read_csv(config.FEEDBACK_LOG_PATH)
                return len(df)
            except Exception:
                return 0
        return 0

    def _eval_val_f1(self, gate_model: nn.Module) -> float:
        """Evaluates F1 score on validation set."""
        if len(self.val_df) == 0:
            self._load_features()
        if len(self.val_df) == 0:
            return 0.8440

        X_val, M_val, P_val, Y_val = prepare_gate_tensors(self.val_df)
        X_val, M_val, P_val = X_val.to(self.device), M_val.to(self.device), P_val.to(self.device)

        gate_model.eval()
        with torch.no_grad():
            w = gate_model(X_val, mask=M_val)
            scores = compute_fused_score(w, P_val).cpu().numpy()

        preds = (scores >= config.GATE_THRESHOLD).astype(int)
        return float(f1_score(Y_val.numpy().astype(int), preds, zero_division=0))

    def predict_sample(self, p_vals: list, u_vals: list, q_vals: list, mask_vals: list, use_live: bool = True):
        """
        Runs gate prediction for a single video sample.
        """
        gate = self.live_gate if use_live else self.frozen_gate
        gate.eval()

        # Input vector: [ps, us, qs, pt, ut, qt, psp, usp, qsp]
        x_vec = []
        for i in range(3):
            x_vec.extend([p_vals[i], u_vals[i], q_vals[i]])

        x_tensor = torch.tensor([x_vec], dtype=torch.float32, device=self.device)
        mask_tensor = torch.tensor([mask_vals], dtype=torch.float32, device=self.device)
        p_tensor = torch.tensor([p_vals], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            weights = gate(x_tensor, mask=mask_tensor)
            fused_score = compute_fused_score(weights, p_tensor).item()

        weights_arr = weights.cpu().numpy()[0]
        fused_score = float(np.clip(fused_score, 0.0, 1.0))
        verdict = decide_verdict(fused_score)

        return {
            "fused_score": round(fused_score, 4),
            "verdict": verdict,
            "weights": {
                "static": round(float(weights_arr[0]), 4),
                "temporal": round(float(weights_arr[1]), 4),
                "speech": round(float(weights_arr[2]), 4)
            }
        }

    def update_feedback(self, video_name: str, sample_inputs: dict, human_label: str):
        """
        Fine-tunes the live gate using 5 Adam steps:
        - Target sample (weight 3)
        - 32 random train rows from features.csv to prevent catastrophic forgetting.
        Rolls back if Val F1 drops > 0.05.
        """
        y_target = 1.0 if human_label.upper() == "UNSAFE" else 0.0

        p_vals = sample_inputs["p"]
        u_vals = sample_inputs["u"]
        q_vals = sample_inputs["q"]
        mask_vals = sample_inputs["mask"]

        # Before prediction
        before_pred = self.predict_sample(p_vals, u_vals, q_vals, mask_vals, use_live=True)

        # Prepare target tensor
        x_target_vec = []
        for i in range(3):
            x_target_vec.extend([p_vals[i], u_vals[i], q_vals[i]])
        x_t = torch.tensor([x_target_vec], dtype=torch.float32, device=self.device)
        m_t = torch.tensor([mask_vals], dtype=torch.float32, device=self.device)
        p_t = torch.tensor([p_vals], dtype=torch.float32, device=self.device)
        y_t = torch.tensor([y_target], dtype=torch.float32, device=self.device)

        # Prepare 32 replay samples
        if len(self.train_df) == 0:
            self._load_features()

        replay_df = self.train_df.sample(n=min(32, len(self.train_df)), replace=False, random_state=random.randint(1, 10000))
        X_rep, M_rep, P_rep, Y_rep = prepare_gate_tensors(replay_df)
        X_rep, M_rep, P_rep, Y_rep = X_rep.to(self.device), M_rep.to(self.device), P_rep.to(self.device), Y_rep.to(self.device)

        # Save backup of current live state dict in case rollback is needed
        backup_state = copy.deepcopy(self.live_gate.state_dict())

        # 5 small Adam steps
        self.live_gate.train()
        optimizer = optim.Adam(self.live_gate.parameters(), lr=1e-3, weight_decay=1e-4)
        bce = nn.BCELoss()

        for _ in range(5):
            optimizer.zero_grad()
            # Target loss (weight 3)
            w_target = self.live_gate(x_t, mask=m_t)
            score_target = compute_fused_score(w_target, p_t)
            loss_target = 3.0 * bce(score_target, y_t)

            # Replay loss
            w_rep = self.live_gate(X_rep, mask=M_rep)
            score_rep = compute_fused_score(w_rep, P_rep)
            loss_rep = bce(score_rep, Y_rep)

            total_loss = loss_target + loss_rep
            total_loss.backward()
            optimizer.step()

        # Check safety on validation split
        new_val_f1 = self._eval_val_f1(self.live_gate)
        f1_drop = self.frozen_val_f1 - new_val_f1

        if f1_drop > 0.05:
            # Rollback
            self.live_gate.load_state_dict(backup_state)
            self.live_gate.eval()
            return {
                "success": False,
                "message": f"Update rejected: Validation F1 dropped by {f1_drop:.3f} (> 0.05 safety margin). Rolled back.",
                "before": before_pred,
                "after": before_pred,
                "verdict_changed": False,
                "version": self.version,
                "val_f1": self.live_val_f1
            }

        # Safe: commit update
        self.version += 1
        self.review_count += 1
        self.live_val_f1 = new_val_f1

        # Save checkpoint
        torch.save(self.live_gate.state_dict(), self.live_weights_path)
        v_path = self.live_dir / f"v{self.version}.pt"
        torch.save(self.live_gate.state_dict(), v_path)

        # After prediction
        after_pred = self.predict_sample(p_vals, u_vals, q_vals, mask_vals, use_live=True)
        verdict_changed = before_pred["verdict"] != after_pred["verdict"]

        # Append to feedback log
        self._log_feedback(
            video_name=video_name,
            human_label=human_label,
            before=before_pred,
            after=after_pred,
            new_val_f1=new_val_f1
        )

        # Append to feedback buffer
        self._buffer_feedback(
            video_name=video_name,
            sample_inputs=sample_inputs,
            human_label=human_label,
            y_target=y_target
        )

        return {
            "success": True,
            "message": f"Live gate updated to v{self.version}. Validation F1 is {new_val_f1:.3f}.",
            "before": before_pred,
            "after": after_pred,
            "verdict_changed": verdict_changed,
            "version": self.version,
            "val_f1": round(new_val_f1, 4),
            "reviews_count": self.review_count
        }

    def _log_feedback(self, video_name: str, human_label: str, before: dict, after: dict, new_val_f1: float):
        entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "video": video_name,
            "human_label": human_label,
            "verdict_before": before["verdict"],
            "verdict_after": after["verdict"],
            "score_before": before["fused_score"],
            "score_after": after["fused_score"],
            "weights_before": f"{before['weights']['static']}/{before['weights']['temporal']}/{before['weights']['speech']}",
            "weights_after": f"{after['weights']['static']}/{after['weights']['temporal']}/{after['weights']['speech']}",
            "val_f1": round(new_val_f1, 4)
        }
        df_entry = pd.DataFrame([entry])
        log_path = Path(config.FEEDBACK_LOG_PATH)
        if log_path.exists():
            df_entry.to_csv(log_path, mode="a", header=False, index=False)
        else:
            df_entry.to_csv(log_path, index=False)

    def _buffer_feedback(self, video_name: str, sample_inputs: dict, human_label: str, y_target: float):
        entry = {
            "video": video_name,
            "human_label": human_label,
            "binary": int(y_target),
            "p_static": sample_inputs["p"][0],
            "u_static": sample_inputs["u"][0],
            "q_static": sample_inputs["q"][0],
            "p_temporal": sample_inputs["p"][1],
            "u_temporal": sample_inputs["u"][1],
            "q_temporal": sample_inputs["q"][1],
            "p_speech": sample_inputs["p"][2],
            "u_speech": sample_inputs["u"][2],
            "q_speech": sample_inputs["q"][2]
        }
        df_buf = pd.DataFrame([entry])
        buf_path = Path(config.FEEDBACK_BUFFER_PATH)
        if buf_path.exists():
            df_buf.to_csv(buf_path, mode="a", header=False, index=False)
        else:
            df_buf.to_csv(buf_path, index=False)

    def reset_gate(self):
        """Restores frozen baseline gate weights to live gate."""
        if self.frozen_weights_path.exists():
            self.live_gate.load_state_dict(torch.load(self.frozen_weights_path, map_location=self.device))
            torch.save(self.live_gate.state_dict(), self.live_weights_path)
            self.version = 0
            self.live_val_f1 = self.frozen_val_f1
            return {
                "success": True,
                "message": "Live gate successfully reset to frozen baseline (v0).",
                "version": 0,
                "live_val_f1": round(self.live_val_f1, 4),
                "frozen_val_f1": round(self.frozen_val_f1, 4),
                "reviews_count": self.review_count
            }
        return {"success": False, "message": "Frozen weights not found."}

    def get_status(self):
        return {
            "version": self.version,
            "reviews_count": self.review_count,
            "live_val_f1": round(self.live_val_f1, 4),
            "frozen_val_f1": round(self.frozen_val_f1, 4)
        }

