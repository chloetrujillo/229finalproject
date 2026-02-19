import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
    train_df = pd.read_csv("data/train.csv")
    val_df = pd.read_csv("data/val.csv")
    test_df = pd.read_csv("data/test.csv")

    # plot distribution of obj_count, trigger_count, portal_count in train dataas separate images
    # save as data/obj_count.png, data/trigger_count.png, data/portal_count.png
    plt.figure(figsize=(10, 5))
    plt.hist(train_df["obj_count"], bins=100, alpha=0.5, label="Train")
    plt.savefig("data/obj_count.png")
    plt.close()
    plt.figure(figsize=(10, 5))
    plt.hist(np.log(train_df["obj_count"] + 1), bins=100, alpha=0.5, label="Train")
    plt.savefig("data/obj_count_log.png")
    plt.close()
    plt.hist(val_df["trigger_count"], bins=100, alpha=0.5, label="Val")
    plt.savefig("data/trigger_count.png")
    plt.close()
    plt.figure(figsize=(10, 5))
    plt.hist(np.log(val_df["trigger_count"] + 1), bins=100, alpha=0.5, label="Val")
    plt.savefig("data/trigger_count_log.png")
    plt.close()
    plt.figure(figsize=(10, 5))
    plt.hist(val_df["trigger_count"] / val_df["obj_count"], bins=100, alpha=0.5, label="Val")
    plt.savefig("data/trigger_count_ratio.png")
    plt.close()

    plt.hist(test_df["portal_count"], bins=100, alpha=0.5, label="Test")
    plt.savefig("data/portal_count.png")
    plt.close()
    plt.figure(figsize=(10, 5))
    plt.hist(np.log(test_df["portal_count"] + 1), bins=100, alpha=0.5, label="Test")
    plt.savefig("data/portal_count_log.png")
    plt.close()
    plt.figure(figsize=(10, 5))
    plt.hist(test_df["portal_count"] / test_df["obj_count"], bins=100, alpha=0.5, label="Test")
    plt.savefig("data/portal_count_ratio.png")
    plt.close()



    # plot distribution of x_max - x_min, y_max - y_min in train data as separate images
    # save as data/x_range.png, data/y_range.png
    plt.figure(figsize=(10, 5))
    plt.hist(train_df["x_max"] - train_df["x_min"], bins=100, alpha=0.5, label="Train")
    plt.savefig("data/x_range.png")
    plt.close()
    plt.figure(figsize=(10, 5))
    plt.hist(np.log(train_df["x_max"] - train_df["x_min"] + 1), bins=100, alpha=0.5, label="Train")
    plt.savefig("data/x_range_log.png")
    plt.close()
    plt.figure(figsize=(10, 5))
    plt.hist(train_df["y_max"] - train_df["y_min"], bins=100, alpha=0.5, label="Train")
    plt.savefig("data/y_range.png")
    plt.close()
    plt.figure(figsize=(10, 5))
    plt.hist(np.log(train_df["y_max"] - train_df["y_min"] + 1), bins=100, alpha=0.5, label="Train")
    plt.savefig("data/y_range_log.png")
    plt.close()

    # Find out outlier in x_max - x_min in train data that's like larger than 1e10
    print(train_df[train_df["x_max"] - train_df["x_min"] > 100000])


if __name__ == "__main__":
    main()