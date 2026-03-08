# Authors: Jaduk Suh

import torch
import torch.nn as nn
import os
import json
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from argparse import ArgumentParser


class GeometryDashDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.x = df.drop(columns=['y', 'stars']).values.astype(np.float32)
        self.y = df['y'].values.astype(np.float32).reshape(-1, 1)
        self.stars = df['stars'].values.astype(np.int32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return torch.from_numpy(self.x[idx]), torch.from_numpy(self.y[idx]), torch.tensor([self.stars[idx]])


class MLPRegression(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MLPRegression, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


def evaluate(model, val_dataset, test_dataset, val_dl, test_dl, criterion):
    model.eval()
    with torch.no_grad():
        got_right = 0
        test_loss = 0.0
        test_off_by_ones = 0
        for x, y, stars in test_dl:
            y_pred = model(x)
            y_logits = torch.round(y_pred * 9) + 1
            got_right += (y_logits == stars).sum().item()
            test_off_by_ones += ((y_logits - stars).abs() == 1).sum().item()
            loss = criterion(y_pred, y)
            test_loss += loss.item() * len(y)
        test_loss /= len(test_dataset)
        test_acc = got_right / len(test_dataset)
        test_off_by_one_acc = (got_right + test_off_by_ones) / len(test_dataset)
        print(f"Test Loss: {test_loss}, Test Accuracy: {test_acc}, Test Off by One: {test_off_by_one_acc}")

        got_right = 0
        val_loss = 0.0
        val_off_by_ones = 0
        for x, y, stars in val_dl:
            y_pred = model(x)
            y_logits = torch.round(y_pred * 9) + 1
            got_right += (y_logits == stars).sum().item()
            val_off_by_ones += ((y_logits - stars).abs() == 1).sum().item()
            loss = criterion(y_pred, y)
            val_loss += loss.item() * len(y)
        val_loss /= len(val_dataset)
        val_acc = got_right / len(val_dataset)
        val_off_by_one_acc = (got_right + val_off_by_ones) / len(val_dataset)
        print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_acc}, Validation Off by One: {val_off_by_one_acc}")


def analyze(model, val_dl, test_dl, out_dir):
    for split, dl, fname in [("Test", test_dl, "error_analysis.png"), ("Validation", val_dl, "val_error_analysis.png")]:
        stars_true, stars_pred = [], []
        for x, y, stars in dl:
            y_pred = model(x)
            y_logits = y_pred * 9 + 1
            for i in range(len(stars)):
                stars_true.append(stars[i].item())
                stars_pred.append(y_logits[i].item())
        plt.figure(figsize=(6, 4))
        sns.stripplot(x=stars_true, y=stars_pred, color="steelblue", alpha=0.3, jitter=0.05, size=4)
        sns.pointplot(x=stars_true, y=stars_pred, color="darkred",
                      estimator=np.mean, errorbar="sd", capsize=.05,
                      markers="D", linestyles="--", label="Mean ± StdDev")
        plt.plot([0, 9], [1, 10], color='black', linestyle=':', alpha=0.6, label="Ideal Prediction")
        plt.xlabel("Ground Truth Stars")
        plt.ylabel("Predicted Stars")
        plt.title(f"{split} Set Error Analysis")
        plt.xticks(ticks=range(10), labels=range(1, 11))
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, fname), dpi=150)
        plt.close()
        print(f"Saved {split.lower()} error analysis to {out_dir}/{fname}")


def normalize(train_df, val_df, test_df):
    train_proc = pd.DataFrame()
    val_proc = pd.DataFrame()
    test_proc = pd.DataFrame()

    # density_s{n}: normalize by total length, then log(x+1), then z-score
    for col in train_df.columns:
        if col.startswith("density_s"):
            train_proc[col] = train_df[col] / train_df["length"]
            val_proc[col] = val_df[col] / val_df["length"]
            test_proc[col] = test_df[col] / test_df["length"]

    for col in [c for c in train_proc.columns if c.startswith("density_s")]:
        log_val = np.log(train_proc[col] + 1)
        mu, std = log_val.mean(), log_val.std()
        if std == 0 or not np.isfinite(std):
            std = 1.0
        train_proc[col] = (log_val - mu) / std
        val_proc[col] = (np.log(val_proc[col] + 1) - mu) / std
        test_proc[col] = (np.log(test_proc[col] + 1) - mu) / std

    # yspread_s{n}: z-score only (pixel distances, not counts)
    for col in train_df.columns:
        if col.startswith("yspread_s"):
            mu, std = train_df[col].mean(), train_df[col].std()
            if std == 0 or not np.isfinite(std):
                std = 1.0
            train_proc[col] = (train_df[col] - mu) / std
            val_proc[col] = (val_df[col] - mu) / std
            test_proc[col] = (test_df[col] - mu) / std

    # length: log(x+1), then z-score
    log_len = np.log(train_df["length"] + 1)
    mu, std = log_len.mean(), log_len.std()
    train_proc["length"] = (log_len - mu) / std
    val_proc["length"] = (np.log(val_df["length"] + 1) - mu) / std
    test_proc["length"] = (np.log(test_df["length"] + 1) - mu) / std

    return train_proc, val_proc, test_proc


def main(args):
    out_dir = "data_v2_spatial"
    print("Reading in dataframes...")
    train_df = pd.read_csv(f"{out_dir}/train.csv").drop(columns=["id"])
    val_df = pd.read_csv(f"{out_dir}/val.csv").drop(columns=["id"])
    test_df = pd.read_csv(f"{out_dir}/test.csv").drop(columns=["id"])

    train_proc, val_proc, test_proc = normalize(train_df, val_df, test_df)

    train_proc["y"] = (train_df["stars"] - 1.0) / 9.0
    val_proc["y"] = (val_df["stars"] - 1.0) / 9.0
    test_proc["y"] = (test_df["stars"] - 1.0) / 9.0
    train_proc["stars"] = train_df["stars"]
    val_proc["stars"] = val_df["stars"]
    test_proc["stars"] = test_df["stars"]

    feature_cols = [c for c in train_proc.columns if c not in ("y", "stars")]
    n_features = len(feature_cols)
    print(f"Total features: {n_features}")

    train_dataset = GeometryDashDataset(train_proc)
    val_dataset = GeometryDashDataset(val_proc)
    test_dataset = GeometryDashDataset(test_proc)
    train_dl = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_dl = DataLoader(val_dataset, batch_size=8, shuffle=False)
    test_dl = DataLoader(test_dataset, batch_size=8, shuffle=False)

    criterion = nn.MSELoss()
    L1_LAMBDA = 1e-3
    N_SELECT = 25

    if args.load_model:
        with open("models/v2_regression_selected_cols_spatial.json", "r") as f:
            selected_cols = json.load(f)
        model = MLPRegression(input_size=len(selected_cols), hidden_size=128)
        model.load_state_dict(torch.load("models/v2_regression_model_spatial.pth"))
        print("Loaded model from models/v2_regression_model_spatial.pth")

        test_sel = test_proc[selected_cols + ["y", "stars"]].copy()
        val_sel = val_proc[selected_cols + ["y", "stars"]].copy()
        test_dl_sel = DataLoader(GeometryDashDataset(test_sel), batch_size=8, shuffle=False)
        val_dl_sel = DataLoader(GeometryDashDataset(val_sel), batch_size=8, shuffle=False)

        evaluate(model, GeometryDashDataset(val_sel), GeometryDashDataset(test_sel), val_dl_sel, test_dl_sel, criterion)
        analyze(model, val_dl_sel, test_dl_sel, out_dir)
        return

    # Phase 1: L1 training on all features for feature selection
    print("Phase 1: L1 training for feature selection")
    model = MLPRegression(input_size=n_features, hidden_size=128)
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=0, lr=3e-5)

    train_losses, val_losses, val_accs, val_off_by_ones = [], [], [], []

    for epoch in range(100):
        model.eval()
        with torch.no_grad():
            got_right, val_loss, off_by_one = 0, 0.0, 0
            for x, y, stars in val_dl:
                y_pred = model(x)
                y_logits = torch.round(y_pred * 9) + 1
                got_right += (y_logits == stars).sum().item()
                off_by_one += ((y_logits - stars).abs() == 1).sum().item()
                val_loss += criterion(y_pred, y).item() * len(y)
            val_loss /= len(val_dataset)
            val_acc = got_right / len(val_dataset)
            val_off_by_one_acc = (got_right + off_by_one) / len(val_dataset)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            val_off_by_ones.append(val_off_by_one_acc)
            print(f"Epoch {epoch}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Off-by-1: {val_off_by_one_acc:.4f}")
        model.train()
        train_loss = 0.0
        for x, y, _ in train_dl:
            optimizer.zero_grad()
            loss = criterion(model(x), y) + L1_LAMBDA * model.fc1.weight.abs().sum()
            train_loss += loss.item() * len(y)
            loss.backward()
            optimizer.step()
        train_loss /= len(train_dataset)
        train_losses.append(train_loss)

    # L1 feature selection
    with torch.no_grad():
        importance = model.fc1.weight.abs().sum(dim=0)
        _, selected_idx = torch.topk(importance, min(N_SELECT, n_features))
        selected_idx = selected_idx.cpu().numpy()
    selected_cols = [feature_cols[i] for i in selected_idx]
    print(f"Selected {len(selected_cols)} features: {selected_cols[:10]}...")

    # Phase 2: retrain on selected features
    print("Phase 2: Retraining on selected features")
    train_sel = train_proc[selected_cols + ["y", "stars"]].copy()
    val_sel = val_proc[selected_cols + ["y", "stars"]].copy()
    test_sel = test_proc[selected_cols + ["y", "stars"]].copy()
    train_dl_sel = DataLoader(GeometryDashDataset(train_sel), batch_size=8, shuffle=True)
    val_dl_sel = DataLoader(GeometryDashDataset(val_sel), batch_size=8, shuffle=False)
    test_dl_sel = DataLoader(GeometryDashDataset(test_sel), batch_size=8, shuffle=False)

    model = MLPRegression(input_size=len(selected_cols), hidden_size=128)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)
    for epoch in range(100):
        model.eval()
        with torch.no_grad():
            got_right, val_loss, off_by_one = 0, 0.0, 0
            for x, y, stars in val_dl_sel:
                y_pred = model(x)
                y_logits = torch.round(y_pred * 9) + 1
                got_right += (y_logits == stars).sum().item()
                off_by_one += ((y_logits - stars).abs() == 1).sum().item()
                val_loss += criterion(y_pred, y).item() * len(y)
            val_loss /= len(GeometryDashDataset(val_sel))
            val_acc = got_right / len(GeometryDashDataset(val_sel))
            print(f"[Selected] Epoch {epoch}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        model.train()
        for x, y, _ in train_dl_sel:
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

    # Plot phase 1 training curves
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss", color="C0")
    plt.plot(val_losses, label="Validation Loss", color="C1")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.title("Train vs Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "train_val_loss.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(val_accs, label="Validation Accuracy", color="C0")
    plt.plot(val_off_by_ones, label="Validation Off-by-1 Accuracy", color="C1")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy and Off-by-1 Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "val_acc_off_by_one.png"), dpi=150)
    plt.close()

    evaluate(model, GeometryDashDataset(val_sel), GeometryDashDataset(test_sel), val_dl_sel, test_dl_sel, criterion)
    analyze(model, val_dl_sel, test_dl_sel, out_dir)

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/v2_regression_model_spatial.pth")
    with open("models/v2_regression_selected_cols_spatial.json", "w") as f:
        json.dump(selected_cols, f)
    print("Saved model to models/v2_regression_model_spatial.pth")


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    parser = ArgumentParser()
    parser.add_argument("--load_model", action="store_true", default=False)
    args = parser.parse_args()
    main(args)
