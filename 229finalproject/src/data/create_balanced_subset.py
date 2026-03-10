import argparse
import os
from typing import Tuple

import pandas as pd


def infer_label_col(df: pd.DataFrame) -> str:
    if "stars" in df.columns:
        return "stars"
    if "label" in df.columns:
        return "label"
    raise ValueError("Could not find label column. Expected 'stars' or 'label'.")


def sample_balanced(df: pd.DataFrame, label_col: str, total_n: int, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    classes = sorted(df[label_col].unique().tolist())
    n_classes = len(classes)
    if n_classes == 0:
        raise ValueError("No classes found in dataset.")

    base = total_n // n_classes
    remainder = total_n % n_classes

    sampled_parts = []
    stats_rows = []
    for i, cls in enumerate(classes):
        cls_df = df[df[label_col] == cls]
        target_n = base + (1 if i < remainder else 0)
        take_n = min(target_n, len(cls_df))
        sampled = cls_df.sample(n=take_n, random_state=seed, replace=False)
        sampled_parts.append(sampled)
        stats_rows.append(
            {
                "class": cls,
                "available": int(len(cls_df)),
                "requested": int(target_n),
                "taken": int(take_n),
            }
        )

    out_df = pd.concat(sampled_parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    stats_df = pd.DataFrame(stats_rows)
    return out_df, stats_df


def main():
    parser = argparse.ArgumentParser(description="Create a balanced subset from train/val/test CSVs.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing train.csv, val.csv, test.csv")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to write subset train/val/test")
    parser.add_argument("--total_train", type=int, default=250, help="Total subset size for train.csv")
    parser.add_argument("--total_val", type=int, default=90, help="Total subset size for val.csv")
    parser.add_argument("--total_test", type=int, default=90, help="Total subset size for test.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    totals = {"train": args.total_train, "val": args.total_val, "test": args.total_test}

    for split, total_n in totals.items():
        in_path = os.path.join(args.input_dir, f"{split}.csv")
        out_path = os.path.join(args.output_dir, f"{split}.csv")
        stats_path = os.path.join(args.output_dir, f"{split}_class_counts.csv")

        df = pd.read_csv(in_path)
        label_col = infer_label_col(df)
        subset_df, stats_df = sample_balanced(df, label_col, total_n, args.seed)
        subset_df.to_csv(out_path, index=False)
        stats_df.to_csv(stats_path, index=False)

        print(f"{split}: requested={total_n}, written={len(subset_df)} -> {out_path}")
        print(stats_df.to_string(index=False))
        print("-" * 60)


if __name__ == "__main__":
    main()
