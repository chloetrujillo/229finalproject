# Authors: Jaduk Suh

import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def process_level_data(obj_ids, data_file, metadata_file):

    with open(data_file, "r") as f:
        data = json.load(f)
    with open(metadata_file, "r") as f:
        metadata = json.load(f)

    # Fetch level id and number of stars from metadata
    level_id = metadata.get("id")
    stars = metadata.get("difficulty").get("stars")

    result = {
        "id": level_id,
        "stars": stars
    }
    
    # Fetch x, y range of objects
    x_min = float('inf')
    x_max = float('-inf')
    y_min = float('inf')
    y_max = float('-inf')
    
    for obj_id in obj_ids:
        result["obj_" + str(obj_id)] = 0
    objs = data.get("data")
    result["length"] = len(objs)
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
        if obj.get("id") is None:
            continue
        result["obj_" + str(obj["id"])] += 1
    result["x_min"] = x_min
    result["x_max"] = x_max
    result["y_min"] = y_min
    result["y_max"] = y_max
    return result

def main():
    with open("src/data/id_counts_by_star.json", "r") as f:
        data = json.load(f)
    total = data["total"]
    ids = list(total.keys())
    ids.sort()

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
                level_data = process_level_data(ids, data_file, metadata_file)
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
    train_df.to_csv(f"data_v2_v1_merge/train.csv", index=False)
    val_df.to_csv(f"data_v2_v1_merge/val.csv", index=False)
    test_df.to_csv(f"data_v2_v1_merge/test.csv", index=False)

if __name__ == "__main__":
    main()