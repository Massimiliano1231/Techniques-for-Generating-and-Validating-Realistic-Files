import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from config.constants import METRICS




def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def load_csv(path):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    if "entropy_random" not in df.columns and "entorpy_random" in df.columns:
        df = df.rename(columns={"entorpy_random": "entropy_random"})
    for col in METRICS + ["entropy_random", "entropy_real"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[m for m in METRICS if m in df.columns], how="all")
    return df

def bar_value_labels(ax, xs, ys, fmt="%.3f"):
    for x, y in zip(xs, ys):
        ax.text(x, y, fmt % y, ha="center", va="bottom", fontsize=9)

def plot_metric_bars(df, metric, out_dir):
    """Un bar-chart con la media per formato per una metrica."""
    if metric not in df.columns:
        return
    grp = df.groupby("format")[metric].agg(["mean", "std", "count"]).dropna()
    if grp.empty:
        return

    formats = grp.index.tolist()
    means = grp["mean"].values

    plt.figure()
    xs = np.arange(len(formats))
    plt.bar(xs, means)
    plt.xticks(xs, formats)
    plt.ylabel(metric)
    plt.title(f"{metric} — media per formato")
    bar_value_labels(plt.gca(), xs, means, fmt="%.3f")
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{metric}_media_per_formato.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[OK] {out_path}")

    summary_path = os.path.join(out_dir, "summary_metriche_per_formato.csv")
    tmp = grp.reset_index()
    tmp.insert(0, "metric", metric)
    if os.path.exists(summary_path):
        tmp.to_csv(summary_path, mode="a", header=False, index=False)
    else:
        tmp.to_csv(summary_path, index=False)

def plot_entropy_by_format(df, out_dir):
    """Boxplot per formato: entropy_random vs entropy_real (stessa scala)."""
    if "entropy_random" not in df.columns or "entropy_real" not in df.columns:
        print("[WARN] Entropie mancanti: non creo i grafici di entropia.")
        return

    formats = sorted(df["format"].dropna().unique().tolist())
    for f in formats:
        sub = df[df["format"] == f]
        e_rnd = sub["entropy_random"].dropna().values
        e_real = sub["entropy_real"].dropna().values
        if len(e_rnd) == 0 and len(e_real) == 0:
            continue

        plt.figure()
        data = [e_rnd, e_real]
        labels = ["random", "real"]
        plt.boxplot(data, labels=labels, showmeans=True)
        plt.ylabel("entropy (bits)")
        plt.title(f"Entropia — {f}: random vs real")
        plt.tight_layout()
        out_path = os.path.join(out_dir, f"entropia_{f}.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"[OK] {out_path}")
