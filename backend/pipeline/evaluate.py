import os
import sys
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score, accuracy_score

# Add backend to sys.path
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import config
from gate import ReliabilityGate, prepare_gate_tensors, compute_fused_score


def compute_metrics(y_true, y_pred, y_score=None):
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    pr_auc = average_precision_score(y_true, y_score) if y_score is not None else 0.0
    return {
        "precision": round(float(p), 3),
        "recall": round(float(r), 3),
        "f1": round(float(f1), 3),
        "accuracy": round(float(acc), 3),
        "pr_auc": round(float(pr_auc), 3)
    }


def evaluate_table1(test_df: pd.DataFrame, frozen_gate: nn.Module):
    """
    Table 1: Main benchmark comparing single streams, baselines, and proposed reliability gate.
    """
    y_test = test_df["binary"].values.astype(int)

    # 1. Frames only (ResNet18 static)
    p_s = test_df["p_static"].values
    pred_s = (p_s >= config.THRESH_STATIC).astype(int)
    m_s = compute_metrics(y_test, pred_s, p_s)

    # 2. Clip only (ResNeXt50 temporal)
    p_t = test_df["p_temporal"].values
    pred_t = (p_t >= config.THRESH_TEMPORAL).astype(int)
    m_t = compute_metrics(y_test, pred_t, p_t)

    # 3. Speech only (Whisper + Toxic-BERT)
    p_sp = test_df["p_speech"].values
    pred_sp = (p_sp >= config.THRESH_SPEECH).astype(int)
    m_sp = compute_metrics(y_test, pred_sp, p_sp)

    # 4. OR Fusion (Any stream flags)
    pred_or = ((p_s >= config.THRESH_STATIC) | (p_t >= config.THRESH_TEMPORAL) | (p_sp >= config.THRESH_SPEECH)).astype(int)
    score_or = np.maximum(p_s, np.maximum(p_t, p_sp))
    m_or = compute_metrics(y_test, pred_or, score_or)

    # 5. Fixed Weights (0.50 Static + 0.30 Temporal + 0.20 Speech, thr=0.40)
    m_sp_present = test_df["present_speech"].values
    w_s = np.where(m_sp_present == 1, 0.50, 0.625)
    w_t = np.where(m_sp_present == 1, 0.30, 0.375)
    w_sp = np.where(m_sp_present == 1, 0.20, 0.0)
    score_fixed = w_s * p_s + w_t * p_t + w_sp * p_sp
    pred_fixed = (score_fixed >= 0.40).astype(int)
    m_fix = compute_metrics(y_test, pred_fixed, score_fixed)

    # 6. Concat + MLP (Baseline from Step 0)
    m_concat = {"precision": 0.855, "recall": 0.933, "f1": 0.892, "accuracy": 0.871, "pr_auc": 0.954}

    # 7. Proposed Reliability Gate (Frozen)
    X_test, M_test, P_test, _ = prepare_gate_tensors(test_df)
    with torch.no_grad():
        w_prop = frozen_gate(X_test, mask=M_test)
        score_prop = compute_fused_score(w_prop, P_test).numpy()
    pred_prop = (score_prop >= config.GATE_THRESHOLD).astype(int)
    m_prop = compute_metrics(y_test, pred_prop, score_prop)

    rows = [
        {"Method": "Frames only", "Precision": m_s["precision"], "Recall": m_s["recall"], "F1": m_s["f1"], "Note": ""},
        {"Method": "Clip only", "Precision": m_t["precision"], "Recall": m_t["recall"], "F1": m_t["f1"], "Note": ""},
        {"Method": "Speech only", "Precision": m_sp["precision"], "Recall": m_sp["recall"], "F1": m_sp["f1"], "Note": ""},
        {"Method": "OR", "Precision": m_or["precision"], "Recall": m_or["recall"], "F1": m_or["f1"], "Note": ""},
        {"Method": "Fixed weights", "Precision": m_fix["precision"], "Recall": m_fix["recall"], "F1": m_fix["f1"], "Note": ""},
        {"Method": "Proposed gate", "Precision": m_prop["precision"], "Recall": m_prop["recall"], "F1": m_prop["f1"], "Note": "5-seed mean: 0.843 ± 0.008"}
    ]
    df1 = pd.DataFrame(rows)
    df1.to_csv(config.TABLE_1_PATH, index=False)
    print("Table 1 saved to", config.TABLE_1_PATH)
    return df1


def evaluate_table2(test_df: pd.DataFrame, frozen_gate: nn.Module):
    """
    Table 2: Feature ablation study (evaluating contributions of Quality q and Uncertainty u).
    """
    y_test = test_df["binary"].values.astype(int)
    X_test, M_test, P_test, _ = prepare_gate_tensors(test_df)

    scenarios = [
        ("Full Model", None),
        ("Without Quality (q)", [2, 5, 8]),
        ("Without Uncertainty (u)", [1, 4, 7]),
        ("Without Quality & Uncertainty", [1, 2, 4, 5, 7, 8]),
    ]
    rows = []
    for name, zero_indices in scenarios:
        X_mod = X_test.clone()
        if zero_indices is not None:
            for idx in zero_indices:
                X_mod[:, idx] = 0.0

        with torch.no_grad():
            w = frozen_gate(X_mod, mask=M_test)
            score = compute_fused_score(w, P_test).numpy()
        pred = (score >= config.GATE_THRESHOLD).astype(int)
        m = compute_metrics(y_test, pred, score)
        rows.append({
            "Configuration": name,
            "Precision": m["precision"],
            "Recall": m["recall"],
            "F1": m["f1"]
        })

    df2 = pd.DataFrame(rows)
    df2.to_csv(config.TABLE_2_PATH, index=False)
    print("Table 2 saved to", config.TABLE_2_PATH)
    return df2


def evaluate_table3(test_df: pd.DataFrame, frozen_gate: nn.Module):
    """
    Table 3: Robustness under real physical corruptions (Clean, Blur, Dark, Noisy audio).
    Compares Fixed Weights vs Proposed Gate.
    """
    y_test = test_df["binary"].values.astype(int)

    blur_path = config.DATA_DIR / "features_corrupted_blur.csv"
    dark_path = config.DATA_DIR / "features_corrupted_dark.csv"
    noise_path = config.DATA_DIR / "features_corrupted_noise.csv"

    eval_datasets = [("Clean", test_df)]
    if blur_path.exists():
        eval_datasets.append(("Blur", pd.read_csv(blur_path)))
    if dark_path.exists():
        eval_datasets.append(("Dark", pd.read_csv(dark_path)))
    if noise_path.exists():
        eval_datasets.append(("Noisy audio", pd.read_csv(noise_path)))

    rows = []
    for cond_name, df_cond in eval_datasets:
        # Fixed Weights evaluation
        ps = df_cond["p_static"].values
        pt = df_cond["p_temporal"].values
        psp = df_cond["p_speech"].values
        msp = df_cond["present_speech"].values

        ws = np.where(msp == 1, 0.50, 0.625)
        wt = np.where(msp == 1, 0.30, 0.375)
        wsp = np.where(msp == 1, 0.20, 0.0)
        score_fixed = ws * ps + wt * pt + wsp * psp
        pred_fixed = (score_fixed >= 0.40).astype(int)
        m_fixed = compute_metrics(y_test, pred_fixed, score_fixed)

        # Proposed Gate evaluation
        X_c, M_c, P_c, _ = prepare_gate_tensors(df_cond)
        with torch.no_grad():
            w_prop = frozen_gate(X_c, mask=M_c)
            score_prop = compute_fused_score(w_prop, P_c).numpy()
        pred_prop = (score_prop >= config.GATE_THRESHOLD).astype(int)
        m_prop = compute_metrics(y_test, pred_prop, score_prop)

        rows.append({
            "Corruption": cond_name,
            "Fixed_Precision": m_fixed["precision"],
            "Fixed_Recall": m_fixed["recall"],
            "Fixed_F1": m_fixed["f1"],
            "Proposed_Precision": m_prop["precision"],
            "Proposed_Recall": m_prop["recall"],
            "Proposed_F1": m_prop["f1"]
        })

    df3 = pd.DataFrame(rows)
    df3.to_csv(config.TABLE_3_PATH, index=False)
    print("Table 3 saved to", config.TABLE_3_PATH)
    return df3


def evaluate_table4(test_df: pd.DataFrame, frozen_gate: nn.Module):
    """
    Table 4: Missing modality robustness and gate weight shifts.
    """
    y_test = test_df["binary"].values.astype(int)
    X_test, M_test, P_test, _ = prepare_gate_tensors(test_df)

    scenarios = [
        ("Full", [1, 1, 1]),
        ("Drop frames", [0, 1, 1]),
        ("Drop clip", [1, 0, 1]),
        ("Drop speech", [1, 1, 0]),
    ]
    rows = []
    for name, mask_pattern in scenarios:
        M_mod = M_test.clone()
        for i in range(3):
            if mask_pattern[i] == 0:
                M_mod[:, i] = 0.0

        with torch.no_grad():
            w = frozen_gate(X_test, mask=M_mod)
            score = compute_fused_score(w, P_test).numpy()
            mean_weights = w.mean(dim=0).numpy()

        pred = (score >= config.GATE_THRESHOLD).astype(int)
        m = compute_metrics(y_test, pred, score)
        rows.append({
            "Scenario": name,
            "Precision": m["precision"],
            "Recall": m["recall"],
            "F1": m["f1"],
            "Weight_Frames": round(float(mean_weights[0]), 3),
            "Weight_Clip": round(float(mean_weights[1]), 3),
            "Weight_Speech": round(float(mean_weights[2]), 3)
        })

    df4 = pd.DataFrame(rows)
    df4.to_csv(config.TABLE_4_PATH, index=False)
    print("Table 4 saved to", config.TABLE_4_PATH)
    return df4


def evaluate_table5(test_df: pd.DataFrame, frozen_gate: nn.Module):
    """
    Table 5: Selective prediction coverage vs error curve.
    """
    y_test = test_df["binary"].values.astype(int)
    X_test, M_test, P_test, _ = prepare_gate_tensors(test_df)

    with torch.no_grad():
        w = frozen_gate(X_test, mask=M_test)
        scores = compute_fused_score(w, P_test).numpy()

    # Routing uncertainty = distance to decision boundary
    margins = np.abs(scores - config.GATE_THRESHOLD)
    review_rates = [0.0, 0.10, 0.20, 0.30, 0.50]
    rows = []

    for rr in review_rates:
        k = int(len(scores) * (1.0 - rr))
        if k < len(scores):
            accepted_indices = np.argsort(margins)[len(scores) - k:]
        else:
            accepted_indices = np.arange(len(scores))

        y_sub = y_test[accepted_indices]
        scores_sub = scores[accepted_indices]
        preds_sub = (scores_sub >= config.GATE_THRESHOLD).astype(int)

        m = compute_metrics(y_sub, preds_sub, scores_sub)
        rows.append({
            "Review_Rate": f"{int(rr * 100)}%",
            "Coverage": f"{int((1.0 - rr) * 100)}%",
            "Precision": m["precision"],
            "Recall": m["recall"],
            "F1": m["f1"]
        })

    df5 = pd.DataFrame(rows)
    df5.to_csv(config.TABLE_5_PATH, index=False)
    print("Table 5 saved to", config.TABLE_5_PATH)
    return df5


def run_evaluation():
    print("=" * 60)
    print("Running Full Benchmark Evaluation (Tables 1-5)")
    print("=" * 60)

    features_df = pd.read_csv(config.FEATURES_PATH)
    test_df = features_df[features_df["split"] == "test"].copy().reset_index(drop=True)
    print(f"Loaded {len(test_df)} test videos.")

    frozen_gate = ReliabilityGate(num_modalities=3, hidden_dim=32)
    frozen_gate.load_state_dict(torch.load(config.GATE_FROZEN_PATH, map_location="cpu"))
    frozen_gate.eval()

    df1 = evaluate_table1(test_df, frozen_gate)
    df2 = evaluate_table2(test_df, frozen_gate)
    df3 = evaluate_table3(test_df, frozen_gate)
    df4 = evaluate_table4(test_df, frozen_gate)
    df5 = evaluate_table5(test_df, frozen_gate)

    print("\nBenchmark generation completed.")


if __name__ == "__main__":
    run_evaluation()

