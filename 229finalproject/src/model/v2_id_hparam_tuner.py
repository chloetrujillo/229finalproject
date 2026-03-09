import argparse
import csv
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


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


class MLPRegression(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def preprocess(df_train, df_val, df_test):
    train_processed = pd.DataFrame()
    val_processed = pd.DataFrame()
    test_processed = pd.DataFrame()

    if "stars" in df_train.columns:
        target_col = "stars"
        train_stars = df_train[target_col].astype(np.float32)
        val_stars = df_val[target_col].astype(np.float32)
        test_stars = df_test[target_col].astype(np.float32)
    elif "label" in df_train.columns:
        # Generic fallback for class labels (e.g. 0,1,2)
        train_stars = (df_train["label"].astype(np.float32) + 1.0)
        val_stars = (df_val["label"].astype(np.float32) + 1.0)
        test_stars = (df_test["label"].astype(np.float32) + 1.0)
    else:
        raise ValueError("Expected target column 'stars' or 'label'.")

    star_min = float(train_stars.min())
    star_max = float(train_stars.max())
    star_range = max(1e-8, star_max - star_min)

    if "length" in df_train.columns and any(c.startswith("obj_") for c in df_train.columns):
        for col in df_train.columns:
            if not col.startswith("obj_"):
                continue
            train_processed[col] = df_train[col] / df_train["length"]
            val_processed[col] = df_val[col] / df_val["length"]
            test_processed[col] = df_test[col] / df_test["length"]

        train_processed["obj_count"] = df_train["length"]
        val_processed["obj_count"] = df_val["length"]
        test_processed["obj_count"] = df_test["length"]

        for col in train_processed.columns:
            if not col.startswith("obj_"):
                continue
            log_val = np.log(train_processed[col] + 1)
            mu = log_val.mean()
            std = log_val.std()
            if std == 0 or not np.isfinite(std):
                std = 1.0
            train_processed[col] = (log_val - mu) / std
            val_processed[col] = (np.log(val_processed[col] + 1) - mu) / std
            test_processed[col] = (np.log(test_processed[col] + 1) - mu) / std
    else:
        # Already-aggregated dataset fallback: z-score all numeric feature columns.
        drop_cols = {"id", "level_id", "stars", "label", "y", "split"}
        feature_cols = [
            c
            for c in df_train.columns
            if c not in drop_cols and np.issubdtype(df_train[c].dtype, np.number)
        ]
        for col in feature_cols:
            mu = float(df_train[col].mean())
            std = float(df_train[col].std())
            if std == 0 or not np.isfinite(std):
                std = 1.0
            train_processed[col] = (df_train[col] - mu) / std
            val_processed[col] = (df_val[col] - mu) / std
            test_processed[col] = (df_test[col] - mu) / std

    train_processed["y"] = (train_stars - star_min) / star_range
    val_processed["y"] = (val_stars - star_min) / star_range
    test_processed["y"] = (test_stars - star_min) / star_range
    train_processed["stars"] = train_stars
    val_processed["stars"] = val_stars
    test_processed["stars"] = test_stars
    return train_processed, val_processed, test_processed, star_min, star_max


def build_loss(name, huber_beta):
    if name == "mse":
        return nn.MSELoss()
    if name == "l1":
        return nn.L1Loss()
    if name == "huber":
        return nn.SmoothL1Loss(beta=huber_beta)
    raise ValueError(f"Unsupported loss: {name}")


def eval_model(model, dl, dataset_len, criterion, star_min, star_max):
    star_range = max(1e-8, star_max - star_min)
    model.eval()
    with torch.no_grad():
        got_right = 0
        off_by_one = 0
        total_loss = 0.0
        for x, y, stars in dl:
            y_pred = model(x)
            y_logits = torch.round(y_pred * star_range + star_min)
            got_right += (y_logits == stars).sum().item()
            off_by_one += ((y_logits - stars).abs() == 1).sum().item()
            total_loss += criterion(y_pred, y).item() * len(y)

    total_loss /= dataset_len
    acc = got_right / dataset_len
    off1 = (got_right + off_by_one) / dataset_len
    return total_loss, acc, off1


def train_two_stage(
    train_processed,
    val_processed,
    test_processed,
    *,
    hidden_size,
    batch_size,
    lr,
    weight_decay,
    epochs_stage1,
    epochs_stage2,
    l1_lambda,
    n_select,
    loss_name,
    huber_beta,
    star_min,
    star_max,
):
    feature_cols = [c for c in train_processed.columns if c not in ("y", "stars")]
    n_features = len(feature_cols)

    train_dataset = GeometryDashDataset(train_processed)
    val_dataset = GeometryDashDataset(val_processed)
    test_dataset = GeometryDashDataset(test_processed)
    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    criterion = build_loss(loss_name, huber_beta)
    model = MLPRegression(input_size=n_features, hidden_size=hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for _ in range(epochs_stage1):
        model.train()
        for x, y, _ in train_dl:
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y) + l1_lambda * model.fc1.weight.abs().sum()
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        importance = model.fc1.weight.abs().sum(dim=0)
        _, selected_idx = torch.topk(importance, min(n_select, n_features))
        selected_idx = selected_idx.cpu().numpy()
    selected_cols = [feature_cols[i] for i in selected_idx]

    train_sel = train_processed[selected_cols + ["y", "stars"]].copy()
    val_sel = val_processed[selected_cols + ["y", "stars"]].copy()
    test_sel = test_processed[selected_cols + ["y", "stars"]].copy()

    train_dataset_sel = GeometryDashDataset(train_sel)
    val_dataset_sel = GeometryDashDataset(val_sel)
    test_dataset_sel = GeometryDashDataset(test_sel)
    train_dl_sel = DataLoader(train_dataset_sel, batch_size=batch_size, shuffle=True)
    val_dl_sel = DataLoader(val_dataset_sel, batch_size=batch_size, shuffle=False)
    test_dl_sel = DataLoader(test_dataset_sel, batch_size=batch_size, shuffle=False)

    model = MLPRegression(input_size=len(selected_cols), hidden_size=hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for _ in range(epochs_stage2):
        model.train()
        for x, y, _ in train_dl_sel:
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

    val_loss, val_acc, val_off1 = eval_model(
        model, val_dl_sel, len(val_dataset_sel), criterion, star_min, star_max
    )
    test_loss, test_acc, test_off1 = eval_model(
        model, test_dl_sel, len(test_dataset_sel), criterion, star_min, star_max
    )

    return {
        "selected_feature_count": len(selected_cols),
        "selected_features": selected_cols,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "val_off_by_one": val_off1,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "test_off_by_one": test_off1,
    }


def append_result_row(csv_path, row):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    write_header = not os.path.exists(csv_path)
    fieldnames = list(row.keys())
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def parse_sweep_values(raw_values, current_value):
    if not raw_values:
        return [current_value]
    vals = []
    for token in raw_values.split(","):
        token = token.strip()
        if token == "":
            continue
        vals.append(token)
    return vals


def cast_sweep_value(param, value):
    if param in {"hidden_size", "batch_size", "epochs_stage1", "epochs_stage2", "n_select", "seed"}:
        return int(value)
    if param in {"lr", "weight_decay", "l1_lambda", "huber_beta"}:
        return float(value)
    if param in {"loss"}:
        return str(value)
    raise ValueError(f"Unsupported sweep parameter: {param}")


def main():
    parser = argparse.ArgumentParser(description="Tune v2 ID MLP-L1 model with tabular logging.")
    parser.add_argument("--data_dir", type=str, default="data_v2")
    parser.add_argument("--results_csv", type=str, default="experiments/v2_id_results.csv")
    parser.add_argument("--results_json_dir", type=str, default="experiments/json_runs")

    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--epochs_stage1", type=int, default=100)
    parser.add_argument("--epochs_stage2", type=int, default=100)
    parser.add_argument("--l1_lambda", type=float, default=1e-3)
    parser.add_argument("--n_select", type=int, default=25)
    parser.add_argument("--loss", type=str, default="mse", choices=["mse", "l1", "huber"])
    parser.add_argument("--huber_beta", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", type=str, default="manual")

    parser.add_argument("--sweep_param", type=str, default="")
    parser.add_argument("--sweep_values", type=str, default="")

    args = parser.parse_args()
    set_seed(args.seed)

    train_df = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(args.data_dir, "val.csv"))
    test_df = pd.read_csv(os.path.join(args.data_dir, "test.csv"))
    train_df = train_df.drop(columns=["id"], errors="ignore")
    val_df = val_df.drop(columns=["id"], errors="ignore")
    test_df = test_df.drop(columns=["id"], errors="ignore")
    train_processed, val_processed, test_processed, star_min, star_max = preprocess(
        train_df, val_df, test_df
    )

    runs = []
    if args.sweep_param:
        sweep_vals = parse_sweep_values(
            args.sweep_values, getattr(args, args.sweep_param)
        )
    else:
        sweep_vals = [None]

    for sweep_val in sweep_vals:
        run_cfg = {
            "hidden_size": args.hidden_size,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "epochs_stage1": args.epochs_stage1,
            "epochs_stage2": args.epochs_stage2,
            "l1_lambda": args.l1_lambda,
            "n_select": args.n_select,
            "loss": args.loss,
            "huber_beta": args.huber_beta,
            "seed": args.seed,
        }
        if args.sweep_param:
            run_cfg[args.sweep_param] = cast_sweep_value(args.sweep_param, sweep_val)

        result = train_two_stage(
            train_processed,
            val_processed,
            test_processed,
            hidden_size=run_cfg["hidden_size"],
            batch_size=run_cfg["batch_size"],
            lr=run_cfg["lr"],
            weight_decay=run_cfg["weight_decay"],
            epochs_stage1=run_cfg["epochs_stage1"],
            epochs_stage2=run_cfg["epochs_stage2"],
            l1_lambda=run_cfg["l1_lambda"],
            n_select=run_cfg["n_select"],
            loss_name=run_cfg["loss"],
            huber_beta=run_cfg["huber_beta"],
            star_min=star_min,
            star_max=star_max,
        )

        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        row = {
            "timestamp_utc": ts,
            "tag": args.tag,
            "sweep_param": args.sweep_param,
            "sweep_value": "" if sweep_val is None else str(sweep_val),
            **run_cfg,
            "selected_feature_count": result["selected_feature_count"],
            "val_loss": result["val_loss"],
            "val_acc": result["val_acc"],
            "val_off_by_one": result["val_off_by_one"],
            "test_loss": result["test_loss"],
            "test_acc": result["test_acc"],
            "test_off_by_one": result["test_off_by_one"],
        }
        append_result_row(args.results_csv, row)
        runs.append({"row": row, "selected_features": result["selected_features"]})

        os.makedirs(args.results_json_dir, exist_ok=True)
        out_json = os.path.join(args.results_json_dir, f"{ts}_{args.tag}.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(runs[-1], f, indent=2)

        print(
            f"[{args.tag}] val_acc={row['val_acc']:.4f}, "
            f"val_off1={row['val_off_by_one']:.4f}, "
            f"test_acc={row['test_acc']:.4f}, "
            f"test_off1={row['test_off_by_one']:.4f}"
        )

    print(f"Logged {len(runs)} run(s) to {args.results_csv}")


if __name__ == "__main__":
    main()
