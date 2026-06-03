import matplotlib.pyplot as plt

LABELS = [
    "TXT / CSV",
    "DOCX",
    "UNKNOWN",
    "JPG",
    "PDF",
    "Altri formati",
]

VALUES = [32.0, 25.0, 20.1, 8.5, 8.3, 6.2]


def main():
    plt.rcParams.update({
        "font.size": 14,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, _, autotexts = ax.pie(
        VALUES,
        startangle=90,
        autopct="%1.1f%%",
        pctdistance=1.15,
        labeldistance=1.25,
        wedgeprops={"edgecolor": "white"},
    )

    for text in autotexts:
        text.set_fontsize(13)

    ax.legend(
        wedges,
        LABELS,
        title="Classificazione Magika",
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        fontsize=12,
        title_fontsize=13,
    )
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig("grafico_torta_generati_magika.pdf", format="pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
