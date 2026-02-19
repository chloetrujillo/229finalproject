# Authors: Muran Wu, Jaduk Suh

import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def process_level_data(data_file, metadata_file):

    with open(data_file, "r") as f:
        data = json.load(f)
    with open(metadata_file, "r") as f:
        metadata = json.load(f)

    # Fetch level id and number of stars from metadata
    level_id = metadata.get("id")
    stars = metadata.get("difficulty").get("stars")

    # Features to extract
    # Obj count, trigger count, portal count
    # x min, x max, y min, y max
    
    # Fetch objects from data
    objs = data.get("data")
    obj_count = len(objs)
    if obj_count == 0:
        return False

    # Fetch x, y range of objects
    x_min = float('inf')
    x_max = float('-inf')
    y_min = float('inf')
    y_max = float('-inf')
    trigger_count = 0
    portal_count = 0
    for obj in objs:
        if obj.get("x") is not None:
            if obj.get("x") < x_min:
                x_min = obj.get("x")
            if obj.get("x") > x_max:
                x_max = obj.get("x")
        if obj.get("y") is not None:
            if obj.get("y") < y_min:
                y_min = obj.get("y")
            if obj.get("y") > y_max:
                y_max = obj.get("y")
        if obj.get("type") == "trigger":
            trigger_count += 1
        if obj.get("type") == "portal":
            portal_count += 1

    return {
        "id": level_id,
        "obj_count": obj_count,
        "trigger_count": trigger_count,
        "portal_count": portal_count,
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
        "stars": stars
    }

def main():
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

    # Train val test split csv
    train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
    val_dataset, test_dataset = train_test_split(test_dataset, test_size=0.5, random_state=42)
    train_df = pd.DataFrame(train_dataset)
    val_df = pd.DataFrame(val_dataset)
    test_df = pd.DataFrame(test_dataset)
    train_df.to_csv(f"data/train.csv", index=False)
    val_df.to_csv(f"data/val.csv", index=False)
    test_df.to_csv(f"data/test.csv", index=False)

if __name__ == "__main__":
    main()