"""
Two-step decoupled hyperparameter search for v2_id_l1_regression.

Step 1 — Stage 1 sweep: find stable l1_lambda and n_select.
    Grid: l1_lambda x {1e-4, 1e-3, 1e-2}
          n_select  x {15, 25, 40}
    Stage 2 is fixed at default settings during this step.

Step 2 — Stage 2 sweep: fix the best Stage 1 config, then tune Stage 2.
    Grid: lr          x {3e-5, 1e-4, 3e-4}
          batch_size  x {4, 8}
          hidden_size x {64, 128, 256}
    Loss is fixed to MSE throughout.

All results are appended to tuning_results.csv after each run so partial
results are preserved if the script is interrupted.
"""

import json
import os
import csv
import itertools
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

class GeometryDashDataset(Dataset):
    def __init__(self, df):
        self.x = df.drop(columns=["y", "stars"]).values.astype(np.float32)
        self.y = df["y"].values.astype(np.float32).reshape(-1, 1)
        self.stars = df["stars"].values.astype(np.int32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.x[idx]),
            torch.from_numpy(self.y[idx]),
            torch.tensor([self.stars[idx]]),
        )


def preprocess():
    """Load and normalize data once. Returns processed DataFrames and feature list."""
    out_dir = "data_v2"
    train_df = pd.read_csv(f"{out_dir}/train.csv").drop(columns=["id"])
    val_df   = pd.read_csv(f"{out_dir}/val.csv").drop(columns=["id"])
    test_df  = pd.read_csv(f"{out_dir}/test.csv").drop(columns=["id"])

    train_proc = pd.DataFrame()
    val_proc   = pd.DataFrame()
    test_proc  = pd.DataFrame()

    # Normalize obj counts by level length, then log(x+1) + z-score
    for col in train_df.columns:
        if not col.startswith("obj_"):
            continue
        train_proc[col] = train_df[col] / train_df["length"]
        val_proc[col]   = val_df[col]   / val_df["length"]
        test_proc[col]  = test_df[col]  / test_df["length"]

    train_proc["obj_count"] = train_df["length"]
    val_proc["obj_count"]   = val_df["length"]
    test_proc["obj_count"]  = test_df["length"]

    for col in train_proc.columns:
        if not col.startswith("obj_"):
            continue
        log_train = np.log(train_proc[col] + 1)
        mu, std = log_train.mean(), log_train.std()
        if std == 0 or not np.isfinite(std):
            std = 1.0
        train_proc[col] = (log_train - mu) / std
        val_proc[col]   = (np.log(val_proc[col]  + 1) - mu) / std
        test_proc[col]  = (np.log(test_proc[col] + 1) - mu) / std

    for split_proc, split_df in [(train_proc, train_df), (val_proc, val_df), (test_proc, test_df)]:
        split_proc["y"]     = (split_df["stars"] - 1.0) / 9.0
        split_proc["stars"] = split_df["stars"]

    feature_cols = [c for c in train_proc.columns if c not in ("y", "stars")]
    return train_proc, val_proc, test_proc, feature_cols


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class MLPRegression(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.fc1     = nn.Linear(input_size, hidden_size)
        self.fc2     = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.fc2(torch.relu(self.fc1(x))))


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def get_device():
    # if torch.backends.mps.is_available():
    #     return torch.device("mps")
    # if torch.cuda.is_available():
    #     return torch.device("cuda")
    return torch.device("cpu")


def eval_val(model, val_dl, val_dataset, device):
    """Return (val_acc, val_off_by_one) without modifying model state."""
    model.eval()
    got_right, off_by_one = 0, 0
    with torch.no_grad():
        for x, y, stars in val_dl:
            x, y, stars = x.to(device), y.to(device), stars.to(device)
            y_pred   = model(x)
            y_logits = torch.round(y_pred * 9) + 1
            got_right   += (y_logits == stars).sum().item()
            off_by_one  += ((y_logits - stars).abs() == 1).sum().item()
    n = len(val_dataset)
    return got_right / n, (got_right + off_by_one) / n


def run_stage1(train_proc, val_proc, feature_cols,
               l1_lambda, n_select, device,
               stage1_lr=1e-4, stage1_batch=4, stage1_epochs=100,
               hidden_size=128, seed=42):
    """
    Train Stage 1 (L1-regularized MLP on all features).
    Returns the list of selected feature column names.
    """
    set_seed(seed)

    n_features = len(feature_cols)
    train_ds = GeometryDashDataset(train_proc)
    train_dl = DataLoader(train_ds, batch_size=stage1_batch, shuffle=True)

    model     = MLPRegression(input_size=n_features, hidden_size=hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=stage1_lr)
    criterion = nn.MSELoss()

    model.train()
    for epoch in tqdm(range(stage1_epochs)):
        for x, y, _ in train_dl:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y) + l1_lambda * model.fc1.weight.abs().sum()
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        importance   = model.fc1.weight.abs().sum(dim=0)
        _, sel_idx   = torch.topk(importance, min(n_select, n_features))
        sel_idx      = sel_idx.cpu().numpy()
    selected_cols = [feature_cols[i] for i in sel_idx]
    return selected_cols


def run_stage2(train_proc, val_proc, selected_cols,
               lr, batch_size, hidden_size, device,
               stage2_epochs=100, seed=42):
    """
    Train Stage 2 (fresh MLP on selected features only, MSE loss).
    Returns (val_acc, val_off_by_one).
    """
    set_seed(seed)

    cols = selected_cols + ["y", "stars"]
    train_ds = GeometryDashDataset(train_proc[cols].copy())
    val_ds   = GeometryDashDataset(val_proc[cols].copy())
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    model     = MLPRegression(input_size=len(selected_cols), hidden_size=hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in tqdm(range(stage2_epochs)):
        model.train()
        for x, y, _ in train_dl:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            criterion(model(x), y).backward()
            optimizer.step()

    return eval_val(model, val_dl, val_ds, device)


# ---------------------------------------------------------------------------
# Result logging
# ---------------------------------------------------------------------------

RESULTS_FILE = "data_v2/tuning_results.csv"
FIELDNAMES = [
    "step", "l1_lambda", "n_select", "lr", "batch_size", "hidden_size",
    "val_acc", "val_off_by_one", "selected_cols",
]

def log_result(row: dict):
    write_header = not os.path.exists(RESULTS_FILE)
    with open(RESULTS_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def print_separator(title=""):
    width = 72
    if title:
        print(f"\n{'─' * 4} {title} {'─' * (width - 6 - len(title))}")
    else:
        print("─" * width)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    set_seed()

    device = get_device()
    print(f"Using device: {device}")

    print("Loading and preprocessing data...")
    train_proc, val_proc, test_proc, feature_cols = preprocess()
    print(f"  {len(feature_cols)} features, "
          f"{len(train_proc)} train / {len(val_proc)} val samples\n")

    os.makedirs("data_v2", exist_ok=True)

    # -----------------------------------------------------------------------
    # Step 1: sweep l1_lambda x n_select (Stage 2 fixed at defaults)
    # -----------------------------------------------------------------------
    # STAGE1_L1_LAMBDAS = [1e-3, 1e-2]
    # STAGE1_N_SELECTS  = [10, 25, 40]
    STAGE1_L1_LAMBDAS = [1e-2]
    STAGE1_N_SELECTS  = [25]

    STAGE2_DEFAULT_LR          = 1e-4
    STAGE2_DEFAULT_BATCH       = 8
    STAGE2_DEFAULT_HIDDEN      = 64

    print_separator("STEP 1: Stage 1 sweep (l1_lambda × n_select)")
    print(f"  l1_lambda : {STAGE1_L1_LAMBDAS}")
    print(f"  n_select  : {STAGE1_N_SELECTS}")
    print(f"  Stage 2 fixed: lr={STAGE2_DEFAULT_LR}, "
          f"batch={STAGE2_DEFAULT_BATCH}, hidden={STAGE2_DEFAULT_HIDDEN}\n")

    step1_results = []
    for l1_lambda, n_select in itertools.product(STAGE1_L1_LAMBDAS, STAGE1_N_SELECTS):
        print(f"  l1_lambda={l1_lambda:.0e}, n_select={n_select} ...", end=" ", flush=True)

        selected_cols = run_stage1(
            train_proc, val_proc, feature_cols,
            l1_lambda=l1_lambda, n_select=n_select, device=device,
        )
        val_acc, val_obo = run_stage2(
            train_proc, val_proc, selected_cols,
            lr=STAGE2_DEFAULT_LR,
            batch_size=STAGE2_DEFAULT_BATCH,
            hidden_size=STAGE2_DEFAULT_HIDDEN,
            device=device,
        )

        print(f"val_acc={val_acc:.4f}, val_off_by_one={val_obo:.4f}")
        row = dict(
            step=1,
            l1_lambda=l1_lambda, n_select=n_select,
            lr=STAGE2_DEFAULT_LR, batch_size=STAGE2_DEFAULT_BATCH,
            hidden_size=STAGE2_DEFAULT_HIDDEN,
            val_acc=val_acc, val_off_by_one=val_obo,
            selected_cols=json.dumps(selected_cols),
        )
        log_result(row)
        step1_results.append(row)

    # Pick best Stage 1 config by val_acc
    best1 = max(step1_results, key=lambda r: r["val_acc"])
    best_l1_lambda = best1["l1_lambda"]
    best_n_select  = best1["n_select"]
    best_selected_cols = json.loads(best1["selected_cols"])

    print_separator()
    print(f"  Best Stage 1 config: l1_lambda={best_l1_lambda:.0e}, "
          f"n_select={best_n_select}")
    print(f"  Val acc: {best1['val_acc']:.4f}, "
          f"off-by-one: {best1['val_off_by_one']:.4f}")
    print(f"  Selected features: {best_selected_cols[:6]} ...")

    # -----------------------------------------------------------------------
    # Step 2: sweep Stage 2 (lr x batch_size x hidden_size)
    # -----------------------------------------------------------------------
    STAGE2_LRS         = [3e-5, 6e-5, 1e-4]
    STAGE2_BATCHES     = [4, 8, 16]
    STAGE2_HIDDENS     = [64, 128, 256]

    print_separator("STEP 2: Stage 2 sweep (lr × batch_size × hidden_size)")
    print(f"  Stage 1 fixed: l1_lambda={best_l1_lambda:.0e}, "
          f"n_select={best_n_select}")
    print(f"  lr          : {STAGE2_LRS}")
    print(f"  batch_size  : {STAGE2_BATCHES}")
    print(f"  hidden_size : {STAGE2_HIDDENS}\n")

    step2_results = []
    for lr, batch_size, hidden_size in itertools.product(
        STAGE2_LRS, STAGE2_BATCHES, STAGE2_HIDDENS
    ):
        print(f"  lr={lr:.0e}, batch={batch_size}, hidden={hidden_size} ...",
              end=" ", flush=True)

        val_acc, val_obo = run_stage2(
            train_proc, val_proc, best_selected_cols,
            lr=lr, batch_size=batch_size, hidden_size=hidden_size, device=device,
        )

        print(f"val_acc={val_acc:.4f}, val_off_by_one={val_obo:.4f}")
        row = dict(
            step=2,
            l1_lambda=best_l1_lambda, n_select=best_n_select,
            lr=lr, batch_size=batch_size, hidden_size=hidden_size,
            val_acc=val_acc, val_off_by_one=val_obo,
            selected_cols=json.dumps(selected_cols),
        )
        log_result(row)
        step2_results.append(row)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    best2 = max(step2_results, key=lambda r: r["val_acc"])

    print_separator("SUMMARY")
    print(f"\n  Best overall config:")
    print(f"    Stage 1 — l1_lambda : {best2['l1_lambda']:.0e}")
    print(f"    Stage 1 — n_select  : {best2['n_select']}")
    print(f"    Stage 2 — lr        : {best2['lr']:.0e}")
    print(f"    Stage 2 — batch     : {best2['batch_size']}")
    print(f"    Stage 2 — hidden    : {best2['hidden_size']}")
    print(f"\n  Validation accuracy  : {best2['val_acc']:.4f}")
    print(f"  Val off-by-one acc   : {best2['val_off_by_one']:.4f}")
    print(f"\n  Full results saved to: {RESULTS_FILE}")

    print_separator("Step 2 results (sorted by val_acc)")
    sorted2 = sorted(step2_results, key=lambda r: r["val_acc"], reverse=True)
    header = f"  {'lr':>8}  {'batch':>5}  {'hidden':>6}  {'val_acc':>8}  {'val_obo':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in sorted2:
        print(f"  {r['lr']:>8.0e}  {r['batch_size']:>5}  {r['hidden_size']:>6}"
              f"  {r['val_acc']:>8.4f}  {r['val_off_by_one']:>8.4f}")


if __name__ == "__main__":
    main()
