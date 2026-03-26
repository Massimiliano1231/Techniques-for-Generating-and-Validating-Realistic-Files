import matplotlib.pyplot as plt
import numpy as np

# Dati
formats = ["DOCX", "JPG", "PDF", "TXT"]

data = {
    "JSD": {
        "mean": [0.9831, 0.9812, 0.9739, 0.9631],
    },
    "TVD": {
        "mean": [0.9900, 0.9945, 0.9923, 0.9894],
    },
    "L1": {
        "mean": [1.9800, 1.9891, 1.9846, 1.9788],
    },
    "Cosine": {
        "mean": [0.0124, 0.0222, 0.0240, 0.0330],
    }
}

# Stile sobrio da tesi
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.dpi": 120,
    "axes.spines.top": False,
    "axes.spines.right": False
})

# Generazione grafici
for metric, values in data.items():
    means = values["mean"]
    x = np.arange(len(formats))

    plt.figure(figsize=(6, 4))
    plt.bar(
        x,
        means,
        color="lightgray",
        edgecolor="black",
        linewidth=0.8
    )

    plt.xticks(x, formats)
    plt.ylabel("Distanza media")
    plt.grid(axis="y", linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(f"{metric.lower()}_distance_generated_vs_random.pdf")
    plt.close()
