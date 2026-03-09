# Authors: Muran Wu
# v3: Semantic category features based on GDBrowser object mappings

import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def process_level_data(id_to_category, categories, data_file, metadata_file):
    with open(data_file, "r") as f:
        data = json.load(f)
    with open(metadata_file, "r") as f:
        metadata = json.load(f)

    level_id = metadata.get("id")
    stars = metadata.get("difficulty").get("stars")

    objs = data.get("data")
    if not objs or len(objs) == 0:
        return False

    result = {"id": level_id, "stars": stars, "length": len(objs)}

    # Initialize category counts
    for cat in categories:
        result[cat] = 0

    # Count objects by semantic category
    for obj in objs:
        obj_id = obj.get("id")
        if obj_id is None:
            continue
        cat = id_to_category.get(str(obj_id), "unknown")
        result[cat] += 1

    # Also extract spatial features
    x_min = float('inf')
    x_max = float('-inf')
    y_min = float('inf')
    y_max = float('-inf')
    for obj in objs:
        if obj.get("x") is not None:
            x_min = min(x_min, obj["x"])
            x_max = max(x_max, obj["x"])
        if obj.get("y") is not None:
            y_min = min(y_min, obj["y"])
            y_max = max(y_max, obj["y"])

    result["x_range"] = x_max - x_min if x_max > x_min else 0
    result["y_range"] = y_max - y_min if y_max > y_min else 0

    return result


def main():
    # Load ID -> category mapping
    with open("src/data/id_to_category.json", "r") as f:
        id_to_category = json.load(f)

    # Get all unique categories
    categories = sorted(set(id_to_category.values()))
    print(f"Categories ({len(categories)}): {categories}")

    dataset = []
    for difficulty in range(1, 11):
        print(f"Processing {difficulty} stars levels")
        data_dir = os.path.join("levels", f"{difficulty}stars", "data")
        metadata_dir = os.path.join("levels", f"{difficulty}stars", "metadata")
        if not os.path.isdir(data_dir):
            continue
        for file in tqdm(os.listdir(data_dir)):
            if file.endswith(".json"):
                level_id = file.split(".")[0]
                data_file = os.path.join(data_dir, f"{level_id}.json")
                metadata_file = os.path.join(metadata_dir, f"{level_id}.json")
                level_data = process_level_data(
                    id_to_category, categories, data_file, metadata_file
                )
                if level_data is not False:
                    dataset.append(level_data)
                else:
                    print(f"Failed to process level {level_id}")

    # Train/val/test split
    os.makedirs("data_v3", exist_ok=True)
    train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
    val_dataset, test_dataset = train_test_split(test_dataset, test_size=0.5, random_state=42)
    pd.DataFrame(train_dataset).to_csv("data_v3/train.csv", index=False)
    pd.DataFrame(val_dataset).to_csv("data_v3/val.csv", index=False)
    pd.DataFrame(test_dataset).to_csv("data_v3/test.csv", index=False)
    print(f"Saved: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")


if __name__ == "__main__":
    main()
