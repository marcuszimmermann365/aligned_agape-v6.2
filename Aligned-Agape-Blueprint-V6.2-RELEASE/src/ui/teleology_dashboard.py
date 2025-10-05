
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def plot_series(series_J4, series_SCM, out_path: Path):
    fig = plt.figure(figsize=(5,3))
    ax = plt.subplot(111)
    ax.plot(series_J4, label="J4")
    ax.plot(series_SCM, label="SCM")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120); plt.close(fig)
