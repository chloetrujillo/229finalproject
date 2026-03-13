"""
Error analysis for the trained object-ID + L1 model.

Loads the saved checkpoint and runs inference on the test set to produce:
  1. Confusion matrix  — 10x10 table of counts (rows=true stars, cols=predicted)
  2. Per-class accuracy — exact accuracy for each star tier derived from the matrix

Both tables are printed to stdout and saved as CSV files under data_v2/.
"""

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Dataset / model (mirror of v2_id_l1_regression.py)
# ---------------------------------------------------------------------------

class GeometryDashDataset(Dataset):
    def __init__(self, df):
        self.x     = df.drop(columns=["y", "stars"]).values.astype(np.float32)
        self.y     = df["y"].values.astype(np.float32).reshape(-1, 1)
        self.stars = df["stars"].values.astype(np.int32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.x[idx]),
            torch.from_numpy(self.y[idx]),
            torch.tensor([self.stars[idx]]),
        )


class MLPRegression(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.fc1     = nn.Linear(input_size, hidden_size)
        self.fc2     = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.fc2(torch.relu(self.fc1(x))))


# ---------------------------------------------------------------------------
# Preprocessing (identical pipeline to v2_id_l1_regression.py)
# ---------------------------------------------------------------------------

def preprocess(selected_cols):
    """
    Load and normalize data using train-set statistics.
    Returns test_processed restricted to selected_cols + ['y', 'stars'].
    """
    out_dir  = "data_v2"
    train_df = pd.read_csv(f"{out_dir}/train.csv").drop(columns=["id"])
    test_df  = pd.read_csv(f"{out_dir}/test.csv").drop(columns=["id"])

    train_proc = pd.DataFrame()
    test_proc  = pd.DataFrame()

    for col in train_df.columns:
        if not col.startswith("obj_"):
            continue
        train_proc[col] = train_df[col] / train_df["length"]
        test_proc[col]  = test_df[col]  / test_df["length"]

    train_proc["obj_count"] = train_df["length"]
    test_proc["obj_count"]  = test_df["length"]

    # log(x+1) + z-score using train statistics
    norm_stats = {}
    for col in train_proc.columns:
        if not col.startswith("obj_"):
            continue
        log_train = np.log(train_proc[col] + 1)
        mu, std   = log_train.mean(), log_train.std()
        if std == 0 or not np.isfinite(std):
            std = 1.0
        norm_stats[col]     = (mu, std)
        train_proc[col]     = (log_train - mu) / std
        test_proc[col]      = (np.log(test_proc[col] + 1) - mu) / std

    test_proc["y"]     = (test_df["stars"] - 1.0) / 9.0
    test_proc["stars"] = test_df["stars"]

    return test_proc[selected_cols + ["y", "stars"]].copy()


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(model, dataloader):
    """Return (true_stars, pred_stars) as 1-D numpy arrays."""
    model.eval()
    all_true, all_pred = [], []
    with torch.no_grad():
        for x, y, stars in dataloader:
            y_pred   = model(x)
            y_logits = torch.round(y_pred * 9) + 1
            # clamp to valid range in case sigmoid output is exactly 0 or 1
            y_logits = y_logits.clamp(1, 10)
            all_true.extend(stars.squeeze().tolist())
            all_pred.extend(y_logits.squeeze().tolist())
    return np.array(all_true, dtype=int), np.array(all_pred, dtype=int)


# ---------------------------------------------------------------------------
# Confusion matrix + per-class accuracy
# ---------------------------------------------------------------------------

def build_confusion_matrix(true_stars, pred_stars):
    """
    Returns a 10x10 DataFrame.
    Rows = ground truth stars (1–10), columns = predicted stars (1–10).
    Entry [i, j] = number of levels with true star i predicted as star j.
    """
    stars = list(range(1, 11))
    matrix = pd.DataFrame(0, index=stars, columns=stars)
    matrix.index.name   = "true \\ pred"
    matrix.columns.name = "predicted"
    for t, p in zip(true_stars, pred_stars):
        matrix.loc[t, p] += 1
    return matrix


def per_class_accuracy(confusion):
    """
    Derives per-class accuracy from a confusion matrix DataFrame.
    Returns a DataFrame with columns: star, correct, total, accuracy.
    """
    rows = []
    for star in confusion.index:
        correct = confusion.loc[star, star]
        total   = confusion.loc[star].sum()
        acc     = correct / total if total > 0 else 0.0
        rows.append({"star": star, "correct": correct, "total": total, "accuracy": acc})
    df = pd.DataFrame(rows).set_index("star")
    # append overall row
    overall_correct = int(np.diag(confusion.values).sum())
    overall_total   = int(confusion.values.sum())
    overall_acc     = overall_correct / overall_total if overall_total > 0 else 0.0
    df.loc["overall"] = {
        "correct": overall_correct,
        "total":   overall_total,
        "accuracy": overall_acc,
    }
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    model_path = "models/v2_regression_model.pth"
    cols_path  = "models/v2_regression_selected_cols.json"
    out_dir    = "data_v2"

    # Load selected feature list
    with open(cols_path) as f:
        selected_cols = json.load(f)
    print(f"Loaded {len(selected_cols)} selected features from {cols_path}")

    # Preprocess test set
    test_proc   = preprocess(selected_cols)
    test_ds     = GeometryDashDataset(test_proc)
    test_dl     = DataLoader(test_ds, batch_size=64, shuffle=False)

    # Load model
    model = MLPRegression(input_size=len(selected_cols), hidden_size=128)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    print(f"Loaded model from {model_path}\n")

    # Run inference
    true_stars, pred_stars = run_inference(model, test_dl)

    # Confusion matrix
    confusion = build_confusion_matrix(true_stars, pred_stars)
    print("=== Confusion Matrix (rows=true, cols=predicted) ===")
    print(confusion.to_string())
    confusion.to_csv(f"{out_dir}/confusion_matrix.csv")
    print(f"\nSaved to {out_dir}/confusion_matrix.csv")

    # Per-class accuracy
    per_class = per_class_accuracy(confusion)
    print("\n=== Per-Class Accuracy ===")
    print(per_class.to_string(float_format=lambda x: f"{x:.4f}"))
    per_class.to_csv(f"{out_dir}/per_class_accuracy.csv")
    print(f"Saved to {out_dir}/per_class_accuracy.csv")


if __name__ == "__main__":
    main()
