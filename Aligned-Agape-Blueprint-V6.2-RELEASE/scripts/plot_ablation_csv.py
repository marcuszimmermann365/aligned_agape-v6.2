
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="out/ablation_results.csv")
    ap.add_argument("--png", type=str, default="out/ablation_results.png")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    g = df.groupby("ablation").agg({
        "J4_mean":"mean",
        "J5_mean":"mean",
        "SCM_mean":"mean",
        "fallback_rate":"mean"
    }).reset_index()

    import matplotlib.pyplot as plt
    fig = plt.figure()  # single plot
    x = np.arange(len(g)); w = 0.2
    plt.bar(x - 1.5*w, g["J4_mean"], width=w, label="J4")
    plt.bar(x - 0.5*w, g["J5_mean"], width=w, label="J5")
    plt.bar(x + 0.5*w, g["SCM_mean"], width=w, label="SCM")
    plt.bar(x + 1.5*w, g["fallback_rate"], width=w, label="Fallback rate")
    plt.xticks(x, g["ablation"])
    plt.title("Ablation Benchmark (means over seeds)")
    plt.legend()
    fig.savefig(args.png, dpi=160, bbox_inches="tight")
    print(f"Wrote {args.png}")

if __name__ == "__main__":
    main()
