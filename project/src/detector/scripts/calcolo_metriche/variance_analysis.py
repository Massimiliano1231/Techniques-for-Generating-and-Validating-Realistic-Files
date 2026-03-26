import os, csv, random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os, sys, csv, numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT / "src" / "detector"))
from data.datasets import REAL_SUBDIRS
from config.constants import NGRAM, BUCKETS, EXTS
from core.bfd_features import ngram_bfd_from_path
from core.metrics import jsd, tvd, l1_distance, cosine_sim, entropy
from io.io_utils import list_files, build_get_repr


ROOT = str(PROJECT_ROOT / "data" / "detector" / "datasets")
CSV_DIR = str(PROJECT_ROOT / "data" / "detector" / "derived" / "vari_csv" / "csv_varianza")
PLOTS_DIR = str(PROJECT_ROOT / "results" / "detector" / "grafici" / "varianza")
SEED    = 42
LOGY    = False





def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def main():
    formats = ["pdf", "txt", "jpg", "docx"]

    random.seed(SEED)
    ensure_dir(CSV_DIR)
    ensure_dir(PLOTS_DIR)

    metrics_names = ["JSD","TVD","L1","Cosine","Entropy"]
    # nel CSV: mean, std, var, p95 di ogni metrica
    header = ["format"]
    for m in metrics_names:
        header += [f"{m}_mean", f"{m}_std", f"{m}_var", f"{m}_p95"]

    results = {}

    for fmt in formats:
        if fmt not in REAL_SUBDIRS:
            print(f"[{fmt}] non supportato, skip.")
            continue

        files = []
        for sub in REAL_SUBDIRS[fmt]:
          base = os.path.join(ROOT, sub)
          files += list_files(base, EXTS[fmt])

        if len(files) < 2:
            print(f"[{fmt}] servono almeno 2 file reali. Trovati: {len(files)}")
            continue

        print(f"\n=== {fmt.upper()} | real files = {len(files)} | n={NGRAM}, buckets={BUCKETS} ===")


        rep = build_get_repr(NGRAM, BUCKETS)

        # --- 1) Calcolo firma media
        sum_vec = None
        count = 0

        for f in tqdm(files, desc=f"{fmt}: firma media"):
            v = rep(f)
            if sum_vec is None:
                sum_vec = np.array(v, dtype=float)
            else:
                sum_vec += v
            count += 1

        mean_bfd = sum_vec / count

        # --- 2) Calcolo distanze real-media ---
        d_jsd, d_tvd, d_l1, d_cos, ent_vals = [], [], [], [], []

        for f in tqdm(files, desc=f"{fmt}: distanze real-media"):
            p = rep(f)
            d_jsd.append(jsd(p, mean_bfd))
            d_tvd.append(tvd(p, mean_bfd))
            d_l1.append(l1_distance(p, mean_bfd))
            d_cos.append(cosine_sim(p, mean_bfd))
            ent_vals.append(entropy(p))

        # 4) Statistiche
        def stats(x):
            x = np.array(x, dtype=float)
            return float(np.mean(x)), float(np.std(x)), float(np.var(x)), float(np.percentile(x, 95))

        res_fmt = {}
        res_fmt["JSD"]     = stats(d_jsd)
        res_fmt["TVD"]     = stats(d_tvd)
        res_fmt["L1"]      = stats(d_l1)
        res_fmt["Cosine"]  = stats(d_cos)
        res_fmt["Entropy"] = stats(ent_vals)

        for m in metrics_names:
            mu, sd, var, p95 = res_fmt[m]
            print(f"{m:<9}  μ={mu:.6f}  σ={sd:.6f}  Var={var:.6f}  P95={p95:.6f}")

        results[fmt] = res_fmt

    # CSV
    csv_path = os.path.join(CSV_DIR, "variance_from_mean_summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for fmt, res_fmt in results.items():
            row = [fmt]
            for m in metrics_names:
                mu, sd, var, p95 = res_fmt[m]
                row += [mu, sd, var, p95]
            w.writerow(row)
    print(f"\nCSV salvato: {csv_path}")

    #  Grafici
    formats_order = list(results.keys())
    if not formats_order:
        print("Nessun formato processato, stop.")
        return

    for m in metrics_names:
        vals = [results[fmt][m][2] for fmt in formats_order]  
        plt.figure(figsize=(7,4))
        plt.bar(formats_order, vals)
        if LOGY:
            plt.yscale("log")
            plt.ylabel("Varianza (log scale)")
        else:
            plt.ylabel("Varianza")
        plt.title(f"Varianza {m} dalla firma media (n={NGRAM})")
        for i, v in enumerate(vals):
            if np.isfinite(v):
                y = v*1.02 if v > 0 else 0.0
                plt.text(i, y, f"{v:.3g}", ha="center", va="bottom", fontsize=9)
        plt.tight_layout()
        out_png = os.path.join(PLOTS_DIR, f"Var_{m.replace(' ','_')}.png")
        plt.savefig(out_png, dpi=200)
        plt.close()
        print(f"Figura salvata: {out_png}")

if __name__ == "__main__":
    main()
