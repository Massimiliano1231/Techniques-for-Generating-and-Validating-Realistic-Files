import matplotlib.pyplot as plt
import numpy as np

FORMATS = ["DOCX", "JPG", "PDF", "TXT"]
ENTROPY_RANDOM = [11.9379, 14.4502, 15.2401, 15.5480]
ENTROPY_REAL = [8.2319, 7.5339, 7.7467, 8.5213]


def main():
    x = np.arange(len(FORMATS))
    width = 0.35

    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "figure.dpi": 120,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    plt.figure(figsize=(6, 4))
    plt.bar(
        x - width / 2,
        ENTROPY_REAL,
        width,
        label="File generati",
        color="lightgray",
        edgecolor="black",
        linewidth=0.8,
    )
    plt.bar(
        x + width / 2,
        ENTROPY_RANDOM,
        width,
        label="File random",
        color="darkgray",
        edgecolor="black",
        linewidth=0.8,
    )

    plt.xticks(x, FORMATS)
    plt.ylabel("Entropia media")
    plt.legend(frameon=False)
    plt.grid(axis="y", linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig("entropy_real_vs_random.pdf")
    plt.close()


if __name__ == "__main__":
    main()
