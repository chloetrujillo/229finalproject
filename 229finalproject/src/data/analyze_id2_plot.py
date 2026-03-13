import json
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

def main():
    with open("src/data/id_counts_by_star.json", "r") as f:
        data = json.load(f)
    total = data["total"]
    ids = []
    for id, count in total.items():
        if count > 10000:
            ids.append(id)

    total_counts = {}
    for star in range(1, 11):
        total_counts[star] = 0
        for id, count in data["by_star"][str(star)].items():
            total_counts[star] += count
        
    for id in tqdm(ids):
        # plot relative frequency of id in all difficulties
        frequencies = []
        for star in range(1, 11):
            if str(id) not in data["by_star"][str(star)]:
                frequency = 0.0
            else:
                frequency = data["by_star"][str(star)][str(id)]
            frequencies.append(frequency / total_counts[star])
        plt.title(f"ID: {id}")
        plt.plot(range(1, 11), frequencies)
        plt.savefig(f"plots/id_{id}.png")
        plt.clf()
    

if __name__ == "__main__":
    main()