import torch
import torch.nn as nn
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import seaborn as sns
import json

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
    # Input size -> obj_count, trigger_count, portal_count, x_range, y_range -> 5
    def __init__(self, hidden_size):
        super(MLPRegression, self).__init__()
        self.fc1 = nn.Linear(5, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

def main(args):
    train_df = pd.read_csv("data/train.csv")
    val_df = pd.read_csv("data/val.csv")
    test_df = pd.read_csv("data/test.csv")

    train_df = train_df.drop(columns=["id"])
    val_df = val_df.drop(columns=["id"])
    test_df = test_df.drop(columns=["id"])

    # Preprocess data for normalization
    train_processed = pd.DataFrame()
    val_processed = pd.DataFrame()
    test_processed = pd.DataFrame()


    # obj_count, trigger_count, portal_count -> first log(x+1) then z-score
    for col in ["obj_count", "trigger_count", "portal_count"]:
        log_val = np.log(train_df[col] + 1)
        mu = log_val.mean()
        std = log_val.std()
        train_processed[col] = (log_val - mu) / std
        val_processed[col] = (np.log(val_df[col] + 1) - mu) / std
        test_processed[col] = (np.log(test_df[col] + 1) - mu) / std


    # x_min, x_max -> calculate range then log(x+1) then z-score
    range_val = np.log(train_df["x_max"] - train_df["x_min"] + 1)
    mu = range_val.mean()
    std = range_val.std()
    train_processed["x_range"] = (range_val - mu) / std
    val_processed["x_range"] = (np.log(val_df["x_max"] - val_df["x_min"] + 1) - mu) / std
    test_processed["x_range"] = (np.log(test_df["x_max"] - test_df["x_min"] + 1) - mu) / std

    # y_min, y_max -> linearly normalize to [0, 1]
    range_val = train_df["y_max"] - train_df["y_min"]
    range_min = range_val.min()
    range_max = range_val.max()
    train_processed["y_range"] = (range_val - range_min) / (range_max - range_min)
    val_processed["y_range"] = (val_df["y_max"] - val_df["y_min"] - range_min) / (range_max - range_min)
    test_processed["y_range"] = (test_df["y_max"] - test_df["y_min"] - range_min) / (range_max - range_min)

    train_processed["y"] = (train_df["stars"] - 1.0) / 9.0
    val_processed["y"] = (val_df["stars"] - 1.0) / 9.0
    test_processed["y"] = (test_df["stars"] - 1.0) / 9.0
    train_processed["stars"] = train_df["stars"]
    val_processed["stars"] = val_df["stars"]
    test_processed["stars"] = test_df["stars"]

    train_dataset = GeometryDashDataset(train_processed)
    val_dataset = GeometryDashDataset(val_processed)
    test_dataset = GeometryDashDataset(test_processed)
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # Simple 2-layer MLP regression model
    model = MLPRegression(hidden_size=16)
    if args.load_model:
        model.load_state_dict(torch.load("models/preliminary_regression_model.pth"))
        print("Loaded model checkpoint from models/preliminary_regression_model.pth")
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
        plt.savefig("data/error_analysis.png", dpi=150)
        plt.close()
        print("Saved error analysis plot to data/error_analysis.png")

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
        plt.savefig("data/val_error_analysis.png", dpi=150)
        plt.close()
        print("Saved validation error analysis plot to data/val_error_analysis.png")
        return

    else:
        print("No model checkpoint found, training from scratch")
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-4, lr=3e-5)
    
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
            loss = criterion(y_pred, y)
            train_loss += loss.item() * len(y)
            loss.backward()
            optimizer.step()
        train_loss /= len(train_dataset)
        train_losses.append(train_loss)
        print(f"Epoch {epoch}, Train Loss: {train_loss}")

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
    plt.savefig("data/train_val_loss.png", dpi=150)
    plt.close()
    print("Saved loss plot to data/train_val_loss.png")

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
    plt.savefig("data/val_acc_off_by_one.png", dpi=150)
    plt.close()
    print("Saved accuracy plot to data/val_acc_off_by_one.png")
    
    model.eval()
    with torch.no_grad():
        got_right = 0
        test_loss = 0.0
        test_off_by_ones = 0
        for x, y, stars in test_dataloader:
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
        for x, y, stars in val_dataloader:
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
        with open("data/results.json", "w") as f:
            json.dump({
                "test_loss": test_loss,
                "test_acc": test_acc,
                "test_off_by_one_acc": test_off_by_one_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_off_by_one_acc": val_off_by_one_acc
            }, f)

    # Save model checkpoint
    torch.save(model.state_dict(), "models/preliminary_regression_model.pth")
    print("Saved model checkpoint to models/preliminary_regression_model.pth")

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    parser = ArgumentParser()
    parser.add_argument("--load_model", action="store_true", default=False)
    args = parser.parse_args()
    main(args)
