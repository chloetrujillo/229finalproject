import torch
import torch.nn as nn
import os
import json
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import seaborn as sns

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
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

def evaluate(model, val_dataset_sel, test_dataset_sel, val_dl_sel, test_dl_sel, criterion, out_dir):
    model.eval()
    with torch.no_grad():
        got_right = 0
        test_loss = 0.0
        test_off_by_ones = 0
        for x, y, stars in test_dl_sel:
            y_pred = model(x)
            y_logits = torch.round(y_pred * 9) + 1
            got_right += (y_logits == stars).sum().item()
            test_off_by_ones += ((y_logits - stars).abs() == 1).sum().item()
            loss = criterion(y_pred, y)
            test_loss += loss.item() * len(y)
        test_loss /= len(test_dataset_sel)
        test_acc = got_right / len(test_dataset_sel)
        test_off_by_one_acc = (got_right + test_off_by_ones) / len(test_dataset_sel)
        print(f"Test Loss: {test_loss}, Test Accuracy: {test_acc}, Test Off by One: {test_off_by_one_acc}")

        got_right = 0
        val_loss = 0.0
        val_off_by_ones = 0
        for x, y, stars in val_dl_sel:
            y_pred = model(x)
            y_logits = torch.round(y_pred * 9) + 1
            got_right += (y_logits == stars).sum().item()
            val_off_by_ones += ((y_logits - stars).abs() == 1).sum().item()
            loss = criterion(y_pred, y)
            val_loss += loss.item() * len(y)
        val_loss /= len(val_dataset_sel)
        val_acc = got_right / len(val_dataset_sel)
        val_off_by_one_acc = (got_right + val_off_by_ones) / len(val_dataset_sel)
        print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_acc}, Validation Off by One: {val_off_by_one_acc}")
        with open(f"{out_dir}/results.json", "w") as f:
            json.dump({
                "test_loss": test_loss,
                "test_acc": test_acc,
                "test_off_by_one_acc": test_off_by_one_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_off_by_one_acc": val_off_by_one_acc
            }, f)

def analyze(model, val_dataloader, test_dataloader, out_dir):
    # Do error analysis on validation set and test set
    # Plot x axis as ground truth stars and y axis as predicted stars
    test_stars = []
    test_stars_pred = []
    for x, y, stars in test_dataloader:
        y_pred = model(x)
        y_logits = y_pred * 9 + 1
        for i in range(len(stars)):
            test_stars.append(stars[i].item())
            test_stars_pred.append(y_logits[i].item())
    plt.figure(figsize=(6, 4))
    sns.stripplot(x=test_stars, y=test_stars_pred, color="steelblue", 
            alpha=0.3, jitter=0.05, size=4)
    sns.pointplot(x=test_stars, y=test_stars_pred, color="darkred", 
            estimator=np.mean, errorbar="sd", capsize=.05, 
            markers="D", linestyles="--", label="Mean ± StdDev")
    plt.plot([0, 9], [1, 10], color='black', linestyle=':', alpha=0.6, label="Ideal Prediction")
    plt.xlabel("Ground Truth Stars")
    plt.ylabel("Predicted Stars")
    plt.title("Test Set Error Analysis")
    plt.xticks(ticks=range(10), labels=range(1, 11))
    plt.legend()
    plt.savefig(f"{out_dir}/error_analysis.png", dpi=150)
    plt.close()
    print("Saved error analysis plot to data_v2/error_analysis.png")

    val_stars = []
    val_stars_pred = []
    for x, y, stars in val_dataloader:
        y_pred = model(x)
        y_logits = y_pred * 9 + 1
        for i in range(len(stars)):
            val_stars.append(stars[i].item())
            val_stars_pred.append(y_logits[i].item())
    plt.figure(figsize=(6, 4))
    sns.stripplot(x=val_stars, y=val_stars_pred, color="steelblue", 
            alpha=0.3, jitter=0.05, size=4)
    sns.pointplot(x=val_stars, y=val_stars_pred, color="darkred", 
            estimator=np.mean, errorbar="sd", capsize=.05, 
            markers="D", linestyles="--", label="Mean ± StdDev")
    plt.plot([0, 9], [1, 10], color='black', linestyle=':', alpha=0.6, label="Ideal Prediction")
    plt.xlabel("Ground Truth Stars")
    plt.ylabel("Predicted Stars")
    plt.title("Validation Set Error Analysis")
    plt.xticks(ticks=range(10), labels=range(1, 11))
    plt.legend()
    plt.savefig(f"{out_dir}/val_error_analysis.png", dpi=150)
    plt.close()
    print("Saved validation error analysis plot to data_v2/val_error_analysis.png")


def main(args):
    out_dir = "data_v2"
    train_df = pd.read_csv(f"{out_dir}/train.csv").drop(columns=["id"])
    val_df   = pd.read_csv(f"{out_dir}/val.csv").drop(columns=["id"])
    test_df  = pd.read_csv(f"{out_dir}/test.csv").drop(columns=["id"])

    # Load selected feature columns from tuning output
    with open("models/v2_regression_selected_cols.json", "r") as f:
        selected_cols = json.load(f)
    print(f"Loaded {len(selected_cols)} selected features from models/v2_regression_selected_cols.json")

    # Preprocess: normalize obj counts by level length, log(x+1), z-score (train stats)
    train_processed = pd.DataFrame()
    val_processed   = pd.DataFrame()
    test_processed  = pd.DataFrame()

    for col in train_df.columns:
        if not col.startswith("obj_"):
            continue
        train_processed[col] = train_df[col] / train_df["length"]
        val_processed[col]   = val_df[col]   / val_df["length"]
        test_processed[col]  = test_df[col]  / test_df["length"]
    train_processed["obj_count"] = train_df["length"]
    val_processed["obj_count"]   = val_df["length"]
    test_processed["obj_count"]  = test_df["length"]

    for col in train_processed.columns:
        if not col.startswith("obj_"):
            continue
        log_val = np.log(train_processed[col] + 1)
        mu, std = log_val.mean(), log_val.std()
        if std == 0 or not np.isfinite(std):
            std = 1.0
        train_processed[col] = (log_val - mu) / std
        val_processed[col]   = (np.log(val_processed[col]  + 1) - mu) / std
        test_processed[col]  = (np.log(test_processed[col] + 1) - mu) / std

    train_processed["y"]     = (train_df["stars"] - 1.0) / 9.0
    val_processed["y"]       = (val_df["stars"]   - 1.0) / 9.0
    test_processed["y"]      = (test_df["stars"]  - 1.0) / 9.0
    train_processed["stars"] = train_df["stars"]
    val_processed["stars"]   = val_df["stars"]
    test_processed["stars"]  = test_df["stars"]

    cols = selected_cols + ["y", "stars"]
    train_dataset = GeometryDashDataset(train_processed[cols].copy())
    val_dataset   = GeometryDashDataset(val_processed[cols].copy())
    test_dataset  = GeometryDashDataset(test_processed[cols].copy())
    train_dl = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_dl   = DataLoader(val_dataset,   batch_size=16, shuffle=False)
    test_dl  = DataLoader(test_dataset,  batch_size=16, shuffle=False)

    criterion = nn.MSELoss()
    model = MLPRegression(input_size=len(selected_cols), hidden_size=128)

    if args.load_model:
        model.load_state_dict(torch.load("models/v2_regression_model.pth"))
        print("Loaded model checkpoint from models/v2_regression_model.pth")
        evaluate(model, val_dataset, test_dataset, val_dl, test_dl, criterion, out_dir)
        analyze(model, val_dl, test_dl, out_dir)
        return

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    train_losses, val_losses, val_accs, val_off_by_ones = [], [], [], []

    for epoch in range(100):
        model.eval()
        with torch.no_grad():
            got_right, val_loss, off_by_one = 0, 0.0, 0
            for x, y, stars in val_dl:
                y_pred   = model(x)
                y_logits = torch.round(y_pred * 9) + 1
                got_right   += (y_logits == stars).sum().item()
                off_by_one  += ((y_logits - stars).abs() == 1).sum().item()
                val_loss    += criterion(y_pred, y).item() * len(y)
            val_loss /= len(val_dataset)
            val_acc = got_right / len(val_dataset)
            val_off_by_one_acc = (got_right + off_by_one) / len(val_dataset)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            val_off_by_ones.append(val_off_by_one_acc)
            print(f"Epoch {epoch}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val Off-by-1: {val_off_by_one_acc:.4f}")

        model.train()
        train_loss = 0.0
        for x, y, _ in train_dl:
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(y)
        train_loss /= len(train_dataset)
        train_losses.append(train_loss)

    # Plot train/val loss
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss", color="C0")
    plt.plot(val_losses,   label="Validation Loss", color="C1")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.title("Train vs Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/train_val_loss.png", dpi=150)
    plt.close()
    print(f"Saved loss plot to {out_dir}/train_val_loss.png")

    # Plot val accuracy curves
    plt.figure(figsize=(8, 5))
    plt.plot(val_accs,       label="Validation Accuracy", color="C0")
    plt.plot(val_off_by_ones, label="Validation Off-by-One Accuracy", color="C1")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy and Off-by-One Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/val_acc_off_by_one.png", dpi=150)
    plt.close()
    print(f"Saved accuracy plot to {out_dir}/val_acc_off_by_one.png")

    evaluate(model, val_dataset, test_dataset, val_dl, test_dl, criterion, out_dir)
    analyze(model, val_dl, test_dl, out_dir)

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/v2_regression_model.pth")
    print("Saved model checkpoint to models/v2_regression_model.pth")

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    parser = ArgumentParser()
    parser.add_argument("--load_model", action="store_true", default=False)
    args = parser.parse_args()
    main(args)
