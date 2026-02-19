import torch
import torch.nn as nn
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np

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

def main():
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

    # y_min, y_max -> calculate range then z-score
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
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-5, lr=2e-5)

    for epoch in range(100):
        # Calculate validation loss & accuracy
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                got_right = 0
                val_loss = 0.0
                for x, y, stars in val_dataloader:
                    y_pred = model(x)
                    y_logits = torch.round(y_pred * 9) + 1
                    got_right += (y_logits == stars).sum().item()
                    loss = criterion(y_pred, y)
                    val_loss += loss.item() * len(y)
                val_loss /= len(val_dataset)
                val_acc = got_right / len(val_dataset)
                print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_acc}")
            model.train()
        
        for i, (x, y, _) in enumerate(train_dataloader):
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch}, Train Loss: {loss.item()}")

    model.eval()
    with torch.no_grad():
        got_right = 0
        test_loss = 0.0
        for x, y, stars in test_dataloader:
            y_pred = model(x)
            y_logits = torch.round(y_pred * 9) + 1
            got_right += (y_logits == stars).sum().item()
            loss = criterion(y_pred, y)
            test_loss += loss.item() * len(y)
        test_loss /= len(test_dataset)
        test_acc = got_right / len(test_dataset)
        print(f"Test Loss: {test_loss}, Test Accuracy: {test_acc}")

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    main()
