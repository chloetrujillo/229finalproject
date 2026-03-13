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

def evaluate(model, val_dataset_sel, test_dataset_sel, val_dl_sel, test_dl_sel, criterion):
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

def analyze(model, val_dataloader, test_dataloader):
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
    train_df = pd.read_csv(f"{out_dir}/train.csv")
    val_df = pd.read_csv(f"{out_dir}/val.csv")
    test_df = pd.read_csv(f"{out_dir}/test.csv")

    train_df = train_df.drop(columns=["id"])
    val_df = val_df.drop(columns=["id"])
    test_df = test_df.drop(columns=["id"])

    # Preprocess data for normalization
    train_processed = pd.DataFrame()
    val_processed = pd.DataFrame()
    test_processed = pd.DataFrame()
    print(train_df.columns)
    for col in train_df.columns:
        if not col.startswith("obj_"):
            continue
        train_processed[col] = train_df[col] / train_df["length"]
        val_processed[col] = val_df[col] / val_df["length"]
        test_processed[col] = test_df[col] / test_df["length"]
    train_processed["obj_count"] = train_df["length"]
    val_processed["obj_count"] = val_df["length"]
    test_processed["obj_count"] = test_df["length"]
    # normalize -> log(x+1) then z-score
    for col in train_processed.columns:
        if not col.startswith("obj_"):
            continue
        log_val = np.log(train_processed[col] + 1)
        mu = log_val.mean()
        std = log_val.std()
        if std == 0 or not np.isfinite(std):
            std = 1.0  # avoid div by zero / Inf in z-score
        train_processed[col] = (log_val - mu) / std
        val_processed[col] = (np.log(val_processed[col] + 1) - mu) / std
        test_processed[col] = (np.log(test_processed[col] + 1) - mu) / std

    train_processed["y"] = (train_df["stars"] - 1.0) / 9.0
    val_processed["y"] = (val_df["stars"] - 1.0) / 9.0
    test_processed["y"] = (test_df["stars"] - 1.0) / 9.0
    train_processed["stars"] = train_df["stars"]
    val_processed["stars"] = val_df["stars"]
    test_processed["stars"] = test_df["stars"]

    feature_cols = [c for c in train_processed.columns if c not in ("y", "stars")]
    n_features = len(feature_cols)

    train_dataset = GeometryDashDataset(train_processed)
    val_dataset = GeometryDashDataset(val_processed)
    test_dataset = GeometryDashDataset(test_processed)
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    # L1 regression: use L1 on first-layer weights to select ~20-30 strongest features
    L1_LAMBDA = 1e-3
    N_SELECT = 25

    model = MLPRegression(input_size=n_features, hidden_size=128)
    if args.load_model:
        with open("models/v2_regression_selected_cols.json", "r") as f:
            selected_cols = json.load(f)
        num_features = len(selected_cols)
        model = MLPRegression(input_size=num_features, hidden_size=128)
        model.load_state_dict(torch.load("models/v2_regression_model.pth"))
        print("Loaded model checkpoint from models/v2_regression_model.pth")
        
        test_sel = test_processed[selected_cols + ["y", "stars"]].copy()
        val_sel = val_processed[selected_cols + ["y", "stars"]].copy()
        test_dataset_sel = GeometryDashDataset(test_sel)
        val_dataset_sel = GeometryDashDataset(val_sel)
        test_dl_sel = DataLoader(test_dataset_sel, batch_size=4, shuffle=False)
        val_dl_sel = DataLoader(val_dataset_sel, batch_size=4, shuffle=False)
        
        evaluate(model, val_dataset_sel, test_dataset_sel, val_dl_sel, test_dl_sel, nn.MSELoss())
        analyze(model, val_dl_sel, test_dl_sel)
        return

    else:
        print("No model checkpoint found, training from scratch")
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=0, lr=1e-4)
    
    train_losses =[]
    val_losses = []
    val_accs = []
    val_off_by_ones = []

    for epoch in range(100):
        # Calculate validation loss & accuracy
        model.eval()
        with torch.no_grad():
            got_right = 0
            val_loss = 0.0
            off_by_one = 0
            for x, y, stars in val_dataloader:
                y_pred = model(x)
                y_logits = torch.round(y_pred * 9) + 1
                got_right += (y_logits == stars).sum().item()
                off_by_one += ((y_logits - stars).abs() == 1).sum().item()
                loss = criterion(y_pred, y)
                val_loss += loss.item() * len(y)
            val_loss /= len(val_dataset)
            val_acc = got_right / len(val_dataset)
            val_off_by_one_acc = (got_right + off_by_one) / len(val_dataset)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            val_off_by_ones.append(val_off_by_one_acc)
            print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_acc}, Validation Off by One: {val_off_by_one_acc}")
        model.train()
        
        train_loss = 0.0
        for i, (x, y, _) in enumerate(train_dataloader):
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y) + L1_LAMBDA * model.fc1.weight.abs().sum()
            train_loss += loss.item() * len(y)
            loss.backward()
            optimizer.step()
        train_loss /= len(train_dataset)
        train_losses.append(train_loss)
        print(f"Epoch {epoch}, Train Loss: {train_loss}")

    # L1 feature selection: keep only strongest N_SELECT features by |fc1 weight|
    with torch.no_grad():
        importance = model.fc1.weight.abs().sum(dim=0)
        _, selected_idx = torch.topk(importance, min(N_SELECT, n_features))
        selected_idx = selected_idx.cpu().numpy()
    selected_cols = [feature_cols[i] for i in selected_idx]
    print(f"Selected {len(selected_cols)} features (L1): {selected_cols[:10]}...")

    # Retrain on selected features only
    train_sel = train_processed[selected_cols + ["y", "stars"]].copy()
    val_sel = val_processed[selected_cols + ["y", "stars"]].copy()
    test_sel = test_processed[selected_cols + ["y", "stars"]].copy()
    train_dataset_sel = GeometryDashDataset(train_sel)
    val_dataset_sel = GeometryDashDataset(val_sel)
    test_dataset_sel = GeometryDashDataset(test_sel)
    train_dl_sel = DataLoader(train_dataset_sel, batch_size=4, shuffle=True)
    val_dl_sel = DataLoader(val_dataset_sel, batch_size=4, shuffle=False)
    test_dl_sel = DataLoader(test_dataset_sel, batch_size=4, shuffle=False)

    model = MLPRegression(input_size=len(selected_cols), hidden_size=128)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    for epoch in range(100):
        model.eval()
        with torch.no_grad():
            got_right = 0
            val_loss = 0.0
            off_by_one = 0
            for x, y, stars in val_dl_sel:
                y_pred = model(x)
                y_logits = torch.round(y_pred * 9) + 1
                got_right += (y_logits == stars).sum().item()
                off_by_one += ((y_logits - stars).abs() == 1).sum().item()
                loss = criterion(y_pred, y)
                val_loss += loss.item() * len(y)
            val_loss /= len(val_dataset_sel)
            val_acc = got_right / len(val_dataset_sel)
            print(f"[Selected features] Epoch {epoch}, Val Loss: {val_loss}, Val Acc: {val_acc}")
        model.train()
        for x, y, _ in train_dl_sel:
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

    # Plot train and validation loss across epochs
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss", color="C0")
    plt.plot(val_losses, label="Validation Loss", color="C1")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.title("Train vs Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/train_val_loss.png", dpi=150)
    plt.close()
    print("Saved loss plot to data_v2/train_val_loss.png")

    # Plot validation accuracy and off by one accuracy across epochs
    plt.figure(figsize=(8, 5))
    plt.plot(val_accs, label="Validation Accuracy", color="C0")
    plt.plot(val_off_by_ones, label="Validation Off by One Accuracy", color="C1")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy and Off by One Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/val_acc_off_by_one.png", dpi=150)
    plt.close()
    print("Saved accuracy plot to data_v2/val_acc_off_by_one.png")
    
    evaluate(model, val_dataset_sel, test_dataset_sel, val_dl_sel, test_dl_sel, criterion)

    # Save model checkpoint (trained on selected features only)
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/v2_regression_model.pth")
    with open("models/v2_regression_selected_cols.json", "w") as f:
        json.dump(selected_cols, f)
    print("Saved model checkpoint to models/v2_regression_model.pth")

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    parser = ArgumentParser()
    parser.add_argument("--load_model", action="store_true", default=False)
    args = parser.parse_args()
    main(args)
