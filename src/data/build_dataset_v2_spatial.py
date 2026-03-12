# Authors: Jaduk Suh

import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

N_X = 100
N_Y = 20


def process_level_data(data_file, metadata_file):

    with open(data_file, "r") as f:
        data = json.load(f)
    with open(metadata_file, "r") as f:
        metadata = json.load(f)

    level_id = metadata.get("id")
    stars = metadata.get("difficulty").get("stars")

    objs = data.get("data")
    if not objs:
        return False

    xs = [obj["x"] for obj in objs if obj.get("x") is not None]
    ys = [obj["y"] for obj in objs if obj.get("y") is not None]
    if not xs or not ys:
        return False

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_range = x_max - x_min
    y_range = y_max - y_min

    grid = [[0] * N_Y for _ in range(N_X)]

    for obj in objs:
        x = obj.get("x")
        y = obj.get("y")
        if x is None or y is None:
            continue

        xi = min(int((x - x_min) / x_range * N_X), N_X - 1) if x_range > 0 else 0
        yi = min(int((y - y_min) / y_range * N_Y), N_Y - 1) if y_range > 0 else 0
        grid[xi][yi] += 1

    result = {"id": level_id, "stars": stars, "length": len(objs)}
    for xi in range(N_X):
        for yi in range(N_Y):
            result[f"grid_x{xi}_y{yi}"] = grid[xi][yi]

    return result


def main():
    os.makedirs("data_v2_spatial", exist_ok=True)

    dataset = []
    for difficulty in range(1, 11):
        print(f"Processing {difficulty} stars levels")
        data_dir = os.path.join("levels", f"{difficulty}stars", "data")
        metadata_dir = os.path.join("levels", f"{difficulty}stars", "metadata")
        for file in tqdm(os.listdir(data_dir)):
            if file.endswith(".json"):
                level_id = file.split(".")[0]
                data_file = os.path.join(data_dir, f"{level_id}.json")
                metadata_file = os.path.join(metadata_dir, f"{level_id}.json")
                level_data = process_level_data(data_file, metadata_file)
                if level_data is not False:
                    dataset.append(level_data)
                else:
                    print(f"Failed to process level {level_id}")

    train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
    val_dataset, test_dataset = train_test_split(test_dataset, test_size=0.5, random_state=42)
    train_df = pd.DataFrame(train_dataset)
    val_df = pd.DataFrame(val_dataset)
    test_df = pd.DataFrame(test_dataset)
    train_df.to_csv("data_v2_spatial/train.csv", index=False)
    val_df.to_csv("data_v2_spatial/val.csv", index=False)
    test_df.to_csv("data_v2_spatial/test.csv", index=False)
    print(f"Saved dataset to data_v2_spatial/. Total samples: {len(dataset)}, features: {len(train_df.columns)}")


if __name__ == "__main__":
    main()
