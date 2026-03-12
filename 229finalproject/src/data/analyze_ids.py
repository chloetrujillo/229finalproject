import json
from pathlib import Path
from collections import Counter, defaultdict
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving files
import matplotlib.pyplot as plt
import numpy as np


def main():
    levels_root = Path(__file__).resolve().parent.parent.parent / "levels"
    type_counts: Counter[str] = Counter()
    # Per star level: star -> Counter(id -> count)
    by_star: dict[int, Counter[str]] = defaultdict(Counter)

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
                t = obj.get("id")
                if t is not None:
                    sid = str(t)
                    type_counts[sid] += 1
                    by_star[n][sid] += 1

    # Report: all unique types and their counts
    print("Object ids in dataset (id -> count):")
    print("-" * 40)
    for t, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {t!r}: {count:,}")
    print("-" * 40)
    print(f"Total unique ids: {len(type_counts)}")

    # Per-star summary
    out_dir = Path(__file__).resolve().parent
    summary_path = out_dir / "id_counts_by_star.txt"
    json_path = out_dir / "id_counts_by_star.json"

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Object id counts by star level difficulty\n")
        f.write("=" * 60 + "\n\n")
        for star in sorted(by_star.keys()):
            c = by_star[star]
            total = sum(c.values())
            f.write(f"--- {star} star(s) (total objects: {total:,}) ---\n")
            for tid, count in sorted(c.items(), key=lambda x: -x[1])[:50]:
                f.write(f"  {tid}: {count:,}\n")
            if len(c) > 50:
                f.write(f"  ... and {len(c) - 50} more ids\n")
            f.write("\n")
    print(f"Wrote per-star summary to {summary_path}")

    # JSON: full counts per star (for reuse)
    export = {
        "total": dict(type_counts),
        "by_star": {str(k): dict(v) for k, v in sorted(by_star.items())},
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(export, f, indent=0)
    print(f"Wrote full counts to {json_path}")

    # Visualizations
    plot_results(type_counts, by_star, out_dir)

    return type_counts, by_star


def plot_results(
    type_counts: Counter[str],
    by_star: dict[int, Counter[str]],
    out_dir: Path,
    top_n: int = 25,
) -> None:
    """Generate heatmap and line plot of object id usage by star level."""
    stars = sorted(by_star.keys())
    if not stars:
        return

    # Top N IDs overall
    top_ids = [tid for tid, _ in type_counts.most_common(top_n)]
    # Matrix: rows = object id, cols = star level
    matrix = np.zeros((len(top_ids), len(stars)))
    for i, tid in enumerate(top_ids):
        for j, star in enumerate(stars):
            matrix[i, j] = by_star[star].get(tid, 0)

    # 1) Heatmap: raw counts
    fig, ax = plt.subplots(figsize=(10, max(8, len(top_ids) * 0.35)))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(stars)))
    ax.set_xticklabels([f"{s}★" for s in stars])
    ax.set_yticks(range(len(top_ids)))
    ax.set_yticklabels(top_ids, fontsize=9)
    ax.set_xlabel("Star level (difficulty)")
    ax.set_ylabel("Object ID")
    ax.set_title("Object ID usage by difficulty (top {} IDs by total count)".format(top_n))
    plt.colorbar(im, ax=ax, label="Count")
    plt.tight_layout()
    heatmap_path = out_dir / "id_usage_by_star_heatmap.png"
    plt.savefig(heatmap_path, dpi=120)
    plt.close()
    print(f"Saved heatmap to {heatmap_path}")

    # 2) Normalized by star (proportion within each difficulty) — highlights which IDs "define" each star
    row_sums = matrix.sum(axis=0, keepdims=True)
    row_sums[row_sums == 0] = 1
    matrix_norm = matrix / row_sums

    fig, ax = plt.subplots(figsize=(10, max(8, len(top_ids) * 0.35)))
    im = ax.imshow(matrix_norm, aspect="auto", cmap="plasma", vmin=0)
    ax.set_xticks(range(len(stars)))
    ax.set_xticklabels([f"{s}★" for s in stars])
    ax.set_yticks(range(len(top_ids)))
    ax.set_yticklabels(top_ids, fontsize=9)
    ax.set_xlabel("Star level (difficulty)")
    ax.set_ylabel("Object ID")
    ax.set_title("Object ID proportion within each difficulty (top {} IDs)".format(top_n))
    plt.colorbar(im, ax=ax, label="Proportion")
    plt.tight_layout()
    norm_path = out_dir / "id_usage_by_star_proportion.png"
    plt.savefig(norm_path, dpi=120)
    plt.close()
    print(f"Saved proportion heatmap to {norm_path}")

    # 3) Line plot: top 10 IDs across star levels (easy to see trends)
    top_10_ids = [tid for tid, _ in type_counts.most_common(10)]
    fig, ax = plt.subplots(figsize=(9, 5))
    for tid in top_10_ids:
        counts = [by_star[s].get(tid, 0) for s in stars]
        ax.plot(stars, counts, marker="o", label=f"ID {tid}", linewidth=2, markersize=6)
    ax.set_xlabel("Star level (difficulty)")
    ax.set_ylabel("Count")
    ax.set_title("Top 10 object IDs across difficulties")
    ax.legend(loc="best", fontsize=8)
    ax.set_xticks(stars)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    line_path = out_dir / "id_usage_by_star_lines.png"
    plt.savefig(line_path, dpi=120)
    plt.close()
    print(f"Saved line plot to {line_path}")


if __name__ == "__main__":
    main()
