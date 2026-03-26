import matplotlib.pyplot as plt

# Dati ESATTI dal tuo grafico Magika
labels = [
    "TXT / CSV",
    "DOCX",
    "UNKNOW",
    "JPG",
    "PDF",
    "Altri formati"
]

values = [32.0, 25.0, 20.1, 8.5, 8.3, 6.2]

# Impostazioni tipografiche da tesi
plt.rcParams.update({
    "font.size": 14,
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})

fig, ax = plt.subplots(figsize=(8, 8))

wedges, texts, autotexts = ax.pie(
    values,
    startangle=90,
    autopct='%1.1f%%',
    pctdistance=1.15,     # percentuali fuori
    labeldistance=1.25,   # label fuori
    wedgeprops=dict(edgecolor="white")
)

# Migliora leggibilità percentuali
for t in autotexts:
    t.set_fontsize(13)

# Legenda laterale pulita
ax.legend(
    wedges,
    labels,
    title="Classificazione Magika",
    loc="center left",
    bbox_to_anchor=(1.0, 0.5),
    fontsize=12,
    title_fontsize=13
)

ax.set_aspect("equal")  # torta perfettamente circolare

plt.tight_layout()
plt.savefig("grafico_torta_generati_magika.pdf", format="pdf")
plt.close()
