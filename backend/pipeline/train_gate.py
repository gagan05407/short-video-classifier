import os
import sys
import json
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# Add backend to sys.path
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import config
from gate import ReliabilityGate, prepare_gate_tensors, compute_fused_score


def train_gate_single_seed(X_train, M_train, P_train, y_train,
                           X_val, M_val, P_val, y_val,
                           seed: int = 42, max_epochs: int = 250, patience: int = 25,
                           lr: float = 1e-3, weight_decay: float = 1e-4,
                           modality_dropout: float = 0.15):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = ReliabilityGate(num_modalities=3, hidden_dim=32)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCELoss()

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    N_train = X_train.shape[0]

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()

        X_aug = X_train.clone()
        M_aug = M_train.clone()

        # Modality dropout
        dropout_mask = (torch.rand(M_aug.shape) > modality_dropout).float()
        all_zero = (M_aug * dropout_mask).sum(dim=-1) == 0
        dropout_mask[all_zero, 0] = 1.0  # keep at least one modality
        M_aug = M_aug * dropout_mask

        # Quality perturbation
        q_noise = (torch.rand(N_train, 3) * 0.20 - 0.10)
        X_aug[:, 2] = torch.clamp(X_aug[:, 2] + q_noise[:, 0], 0.0, 1.0)
        X_aug[:, 5] = torch.clamp(X_aug[:, 5] + q_noise[:, 1], 0.0, 1.0)
        X_aug[:, 8] = torch.clamp(X_aug[:, 8] + q_noise[:, 2], 0.0, 1.0)

        weights = model(X_aug, mask=M_aug)
        fused_score = compute_fused_score(weights, P_train)

        loss = criterion(fused_score, y_train)
        loss.backward()
        optimizer.step()

        # Validation step
        model.eval()
        with torch.no_grad():
            val_weights = model(X_val, mask=M_val)
            val_fused = compute_fused_score(val_weights, P_val)
            val_loss = criterion(val_fused, y_val).item()

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    model.load_state_dict(best_state)
    return model, best_val_loss, epoch + 1


def train_fusion():
    print("=" * 60)
    print("Training Reliability-Aware Multimodal Fusion Gate")
    print("=" * 60)

    if not config.FEATURES_PATH.exists():
        raise FileNotFoundError(f"features.csv not found at {config.FEATURES_PATH}.")

    features_df = pd.read_csv(config.FEATURES_PATH)
    train_df = features_df[features_df["split"] == "train"].copy()
    val_df = features_df[features_df["split"] == "val"].copy()
    test_df = features_df[features_df["split"] == "test"].copy()

    print(f"Splits: Train={len(train_df)} | Val={len(val_df)} | Test={len(test_df)}")

    X_train, M_train, P_train, y_train = prepare_gate_tensors(train_df)
    X_val, M_val, P_val, y_val = prepare_gate_tensors(val_df)
    X_test, M_test, P_test, y_test = prepare_gate_tensors(test_df)

    seeds = config.GATE_SEEDS
    best_overall_val_loss = float("inf")
    best_overall_model = None
    best_seed = seeds[0]

    for seed in seeds:
        model, val_loss, epochs_run = train_gate_single_seed(
            X_train, M_train, P_train, y_train,
            X_val, M_val, P_val, y_val,
            seed=seed, modality_dropout=config.MODALITY_DROPOUT_PROB
        )
        print(f"Seed {seed:4d} | Best Val Loss: {val_loss:.4f} | Epochs: {epochs_run}")
        if val_loss < best_overall_val_loss:
            best_overall_val_loss = val_loss
            best_overall_model = model
            best_seed = seed

    print(f"\nOptimal seed: {best_seed} with val loss {best_overall_val_loss:.4f}")

    # Save frozen gate checkpoint
    torch.save(best_overall_model.state_dict(), config.GATE_FROZEN_PATH)
    print(f"Saved frozen gate checkpoint: {config.GATE_FROZEN_PATH}")

    # Save gate configuration
    gate_config = {
        "best_seed": best_seed,
        "best_val_loss": round(best_overall_val_loss, 4),
        "num_modalities": 3,
        "modalities": ["static", "temporal", "speech"],
        "hidden_dim": 32,
        "seeds_evaluated": seeds,
        "decision_threshold": config.GATE_THRESHOLD,
        "review_band": [config.T_LO, config.T_HI]
    }
    with open(config.GATE_CONFIG_PATH, "w") as f:
        json.dump(gate_config, f, indent=2)
    print(f"Saved gate config: {config.GATE_CONFIG_PATH}")


if __name__ == "__main__":
    train_fusion()

