
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def plot_radar(J, SCM, out_path: Path):
    labels = ["J1","J2","J3","J4","J5","SCM"]
    values = [J.get("J1",0), J.get("J2",0), J.get("J3",0), J.get("J4",0), J.get("J5",0), SCM]
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    values += values[:1]; angles += angles[:1]
    fig = plt.figure(figsize=(4,4))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, values, linewidth=2)
    ax.fill(angles, values, alpha=0.1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120); plt.close(fig)
