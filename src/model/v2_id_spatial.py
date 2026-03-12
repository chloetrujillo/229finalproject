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
from tqdm import tqdm

N_X, N_Y = 100, 20
SELECTED_COLS_PATH = "models/v2_regression_selected_cols.json"


def load_selected_ids():
    with open(SELECTED_COLS_PATH, "r") as f:
        cols = json.load(f)
    return [int(col[4:]) for col in cols if col.startswith("obj_") and col != "obj_count"]


class GeometryDashCNNDataset(Dataset):
    def __init__(self, df, selected_ids):
        # Build grid columns per object ID: shape (N, n_ids * N_X * N_Y)
        # Then reshape to (N, n_ids, N_Y, N_X)
        n_ids = len(selected_ids)
        all_grids = []
        for obj_id in tqdm(selected_ids):
            cols = [f"obj{obj_id}_grid_x{xi}_y{yi}" for xi in range(N_X) for yi in range(N_Y)]
            grid = df[cols].values.astype(np.float32)  # (N, N_X * N_Y)
            grid = grid.reshape(-1, N_X, N_Y).transpose(0, 2, 1)  # (N, N_Y, N_X)
            all_grids.append(grid)
        # Stack along channel dim: (N, n_ids, N_Y, N_X)
        self.x = np.stack(all_grids, axis=1)
        self.y = df['y'].values.astype(np.float32).reshape(-1, 1)
        self.stars = df['stars'].values.astype(np.int32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.from_numpy(self.x[idx]), torch.from_numpy(self.y[idx]), torch.tensor([self.stars[idx]])


class CNNRegression(nn.Module):
    # Input: (batch, 24, H=20, W=100)
    # After conv1 + pool: (batch, 32, 10, 50)
    # After conv2 + pool: (batch, 64, 5, 25)
    # Flattened: 64 * 5 * 25 = 8000
    def __init__(self, n_ids, hidden_size):
        super(CNNRegression, self).__init__()
        self.conv1 = nn.Conv2d(n_ids, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 5 * 25, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x


def normalize(train_df, val_df, test_df, selected_ids):
    train_proc = pd.DataFrame(index=train_df.index)
    val_proc = pd.DataFrame(index=val_df.index)
    test_proc = pd.DataFrame(index=test_df.index)

    # Per-ID grid columns: divide by length, log(x+1), z-score using train stats
    for obj_id in tqdm(selected_ids):
        cols = [f"obj{obj_id}_grid_x{xi}_y{yi}" for xi in range(N_X) for yi in range(N_Y)]
        for col in cols:
            train_ratio = train_df[col] / train_df["length"]
            val_ratio = val_df[col] / val_df["length"]
            test_ratio = test_df[col] / test_df["length"]

            log_train = np.log(train_ratio + 1)
            mu, std = log_train.mean(), log_train.std()
            if std == 0 or not np.isfinite(std):
                std = 1.0
            train_proc[col] = (log_train - mu) / std
            val_proc[col] = (np.log(val_ratio + 1) - mu) / std
            test_proc[col] = (np.log(test_ratio + 1) - mu) / std

    # length: log(x+1), z-score
    log_len = np.log(train_df["length"] + 1)
    mu, std = log_len.mean(), log_len.std()
    train_proc["length"] = (log_len - mu) / std
    val_proc["length"] = (np.log(val_df["length"] + 1) - mu) / std
    test_proc["length"] = (np.log(test_df["length"] + 1) - mu) / std

    return train_proc, val_proc, test_proc


def evaluate(model, val_dataset, test_dataset, val_dl, test_dl, criterion, save_path):
    model.eval()
    results = {}
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
        results["test_loss"] = test_loss
        results["test_acc"] = test_acc
        results["test_off_by_one_acc"] = test_off_by_one_acc

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
        results["val_loss"] = val_loss
        results["val_acc"] = val_acc
        results["val_off_by_one_acc"] = val_off_by_one_acc
        with open(save_path, "w") as f:
            json.dump(results, f)

    return results


def analyze(model, val_dl, test_dl, out_dir):
    for split, dl, fname in [("Test", test_dl, "cnn_error_analysis.png"), ("Validation", val_dl, "cnn_val_error_analysis.png")]:
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


def main(args):
    selected_ids = load_selected_ids()
    n_ids = len(selected_ids)
    print(f"Using {n_ids} object IDs: {selected_ids}")

    out_dir = "data_v2_id_spatial"
    print("Reading in dataframes...")
    train_df = pd.read_csv(f"{out_dir}/train.csv").drop(columns=["id"])
    val_df = pd.read_csv(f"{out_dir}/val.csv").drop(columns=["id"])
    test_df = pd.read_csv(f"{out_dir}/test.csv").drop(columns=["id"])
    print("Normalizing data...")
    train_proc, val_proc, test_proc = normalize(train_df, val_df, test_df, selected_ids)
    
    train_proc["y"] = (train_df["stars"] - 1.0) / 9.0
    val_proc["y"] = (val_df["stars"] - 1.0) / 9.0
    test_proc["y"] = (test_df["stars"] - 1.0) / 9.0
    train_proc["stars"] = train_df["stars"]
    val_proc["stars"] = val_df["stars"]
    test_proc["stars"] = test_df["stars"]

    print("Building datasets...")
    train_dataset = GeometryDashCNNDataset(train_proc, selected_ids)
    val_dataset = GeometryDashCNNDataset(val_proc, selected_ids)
    test_dataset = GeometryDashCNNDataset(test_proc, selected_ids)
    train_dl = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_dl = DataLoader(val_dataset, batch_size=4, shuffle=False)
    test_dl = DataLoader(test_dataset, batch_size=4, shuffle=False)

    criterion = nn.MSELoss()
    L2_LAMBDA = 1e-4
    model_path = "models/v2_id_spatial_cnn_model.pth"

    if args.load_model:
        model = CNNRegression(n_ids=n_ids, hidden_size=128)
        model.load_state_dict(torch.load(model_path))
        print(f"Loaded model from {model_path}")
        evaluate(model, val_dataset, test_dataset, val_dl, test_dl, criterion, f"{out_dir}/cnn_results.json")
        analyze(model, val_dl, test_dl, out_dir)
        return

    model = CNNRegression(n_ids=n_ids, hidden_size=128)
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=L2_LAMBDA, lr=1e-4)

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
            loss = criterion(model(x), y)
            train_loss += loss.item() * len(y)
            loss.backward()
            optimizer.step()
        train_loss /= len(train_dataset)
        train_losses.append(train_loss)
        print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}")

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

    evaluate(model, val_dataset, test_dataset, val_dl, test_dl, criterion, f"{out_dir}/cnn_results.json")
    analyze(model, val_dl, test_dl, out_dir)

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    parser = ArgumentParser()
    parser.add_argument("--load_model", action="store_true", default=False)
    args = parser.parse_args()
    main(args)
