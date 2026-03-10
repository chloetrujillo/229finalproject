import argparse
import os

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Summarize hyperparameter tuning CSV results.")
    parser.add_argument("--results_csv", type=str, default="experiments/v2_id_results.csv")
    parser.add_argument("--output_csv", type=str, default="experiments/v2_id_results_top20.csv")
    parser.add_argument("--top_k", type=int, default=20)
    args = parser.parse_args()

    if not os.path.exists(args.results_csv):
        raise FileNotFoundError(f"Results CSV not found: {args.results_csv}")

    df = pd.read_csv(args.results_csv)
    sort_cols = ["val_off_by_one", "val_acc", "test_off_by_one", "test_acc"]
    df_sorted = df.sort_values(by=sort_cols, ascending=False).reset_index(drop=True)
    top = df_sorted.head(args.top_k)

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    top.to_csv(args.output_csv, index=False)

    print(f"Total runs: {len(df)}")
    print(f"Saved top {len(top)} to: {args.output_csv}")
    if len(top) > 0:
        best = top.iloc[0]
        print("Best row:")
        print(best.to_string())


if __name__ == "__main__":
    main()
