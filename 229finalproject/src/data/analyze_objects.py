"""
Find every unique object "type" in the Geometry Dash level dataset,
and report counts per type.

Data lives in levels/{n}stars/data/{m}.json where n in 1..10 and m is level ID.
Each JSON has data["data"] = list of game objects; each object has a "type" key.
"""

import json
from pathlib import Path
from collections import Counter
from tqdm import tqdm


def main():
    levels_root = Path(__file__).resolve().parent.parent.parent / "levels"
    type_counts: Counter[str] = Counter()

    # Traverse levels/1stars, levels/2stars, ... levels/10stars
    for n in tqdm(range(1, 11)):
        data_dir = levels_root / f"{n}stars" / "data"
        if not data_dir.is_dir():
            continue

        for json_path in data_dir.glob("*.json"):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                print(f"Warning: skip {json_path}: {e}")
                continue

            objects = data.get("data")
            if not isinstance(objects, list):
                continue

            for obj in objects:
                if not isinstance(obj, dict):
                    continue
                t = obj.get("type")
                if t is not None:
                    type_counts[str(t)] += 1

    # Report: all unique types and their counts
    print("Object types in dataset (type -> count):")
    print("-" * 40)
    for t, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {t!r}: {count:,}")
    print("-" * 40)
    print(f"Total unique types: {len(type_counts)}")
    print(f"Total objects with a 'type' key: {sum(type_counts.values()):,}")

    return type_counts


if __name__ == "__main__":
    main()
