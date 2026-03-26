import matplotlib.pyplot as plt
import numpy as np

# Dati
formats = ["DOCX", "JPG", "PDF", "TXT"]

entropy_random = [11.9379, 14.4502, 15.2401, 15.5480]
entropy_real   = [8.2319,  7.5339,  7.7467,  8.5213]

x = np.arange(len(formats))
width = 0.35

# Stile sobrio da tesi (coerente con gli altri grafici)
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.dpi": 120,
    "axes.spines.top": False,
    "axes.spines.right": False
})

plt.figure(figsize=(6, 4))

plt.bar(
    x - width/2,
    entropy_real,
    width,
    label="File generati",
    color="lightgray",
    edgecolor="black",
    linewidth=0.8
)

plt.bar(
    x + width/2,
    entropy_random,
    width,
    label="File random",
    color="darkgray",
    edgecolor="black",
    linewidth=0.8
)

plt.xticks(x, formats)
plt.ylabel("Entropia media")
plt.legend(frameon=False)
plt.grid(axis="y", linestyle=":", alpha=0.6)

plt.tight_layout()
plt.savefig("entropy_real_vs_random.pdf")
plt.close()
