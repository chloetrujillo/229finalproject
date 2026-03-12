import json
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

def main():
    with open("src/data/id_counts_by_star.json", "r") as f:
        data = json.load(f)
    total = data["total"]
    ids = list(total.keys())
    ids.sort()

    # For all levels in the training dataset,
    # calculate the ratio of each object id and collect statistics
    # 
    

if __name__ == "__main__":
    main()