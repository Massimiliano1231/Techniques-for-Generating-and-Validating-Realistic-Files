#!/usr/bin/env python3
import os
import sys
import csv
import argparse
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from data.datasets import get_format, DATASETS
from config.constants import NGRAM, BUCKETS, RANGES, EXTS, DEFAULT_ROOT, DEFAULT_CSV
from io.io_utils import list_files, load_thresholds, clamp
from core.bfd_features import ngram_bfd_from_path
from core.metrics import jsd, tvd, l1_distance, cosine_sim, entropy



def main():


    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="Cartella con i file da valutare")
    ap.add_argument("--csv", default=DEFAULT_CSV, help="CSV con mean/std per μ±2σ")
    ap.add_argument("--root", default=DEFAULT_ROOT, help="Dataset reali per firma media")
    ap.add_argument("--ngram", type=int, default=NGRAM)
    ap.add_argument("--buckets", type=int, default=BUCKETS)
    ap.add_argument("--out", default=None, help="CSV opzionale con metriche per file")
    args = ap.parse_args()


    # -----------------------------------------------------------
    # 1) Raccogli file del dataset
    # -----------------------------------------------------------
    all_paths = []
    for root, _, files in os.walk(args.dataset):
        for f in files:
            all_paths.append(os.path.join(root, f))

    if not all_paths:
        print("Nessun file nel dataset.")
        sys.exit(1)

    fmt = get_format(all_paths[0])
    if fmt is None:
        print("Formato non riconosciuto.")
        sys.exit(1)

    # -----------------------------------------------------------
    # 2) Carica le soglie μ e σ dal CSV
    # -----------------------------------------------------------
    thr = load_thresholds(args.csv, fmt)
    if thr is None:
        print(f"Soglie non trovate per formato '{fmt}'.")
        sys.exit(1)

    # -----------------------------------------------------------
    # 3) Calcolo della FIRMA MEDIA reale
    # -----------------------------------------------------------
    real_dir = DATASETS[fmt]["real"]
    real_files = list_files(real_dir, EXTS[fmt])

    if len(real_files) < 2:
        print("Non ci sono abbastanza real per calcolare la firma media.")
        sys.exit(1)

    bfds_real = []
    print(f"Calcolo firma media reale ({fmt})...")
    for p in tqdm(real_files, desc="real BFD"):
        bfds_real.append(ngram_bfd_from_path(p, n=args.ngram, buckets=args.buckets))

    mean_bfd = np.mean(bfds_real, axis=0)

    # -----------------------------------------------------------
    # 4) Analisi del dataset con soglie μ±2σ
    # -----------------------------------------------------------
    plausible = 0
    not_plausible = 0

    if args.out:
        fout = open(args.out, "w", newline="")
        w = csv.writer(fout)
        w.writerow(["path", "format", "JSD", "TVD", "L1", "Cosine", "Entropy", "plausible"])
    else:
        fout = None
        w = None

    print(f"\nValuto file in {args.dataset} con regole μ±2σ...\n")

    for pth in tqdm(all_paths, desc="Valutazione"):
        if get_format(pth) != fmt:
            continue

        bfd = ngram_bfd_from_path(pth, n=args.ngram, buckets=args.buckets)
        if bfd is None:
            continue

        # metriche rispetto al mean_bfd
        vals = {
            "JSD":      jsd(bfd, mean_bfd),
            "TVD":      tvd(bfd, mean_bfd),
            "L1":       l1_distance(bfd, mean_bfd),
            "Cosine":   cosine_sim(bfd, mean_bfd),
            "Entropy":  entropy(bfd),
        }

        # --- regole μ ± 2σ ---
        K = 3.0
        ok = True

        # distanze: val ≤ μ + 2σ
        for m in ["JSD", "TVD", "L1"]:
            mu = thr[m]["mean"]
            sd = thr[m]["std"]
            T = clamp(mu + K*sd, *RANGES[m])
            if not (vals[m] <= T):
                ok = False

        # cosine: val ≥ μ - 2σ
        mu_c = thr["Cosine"]["mean"]
        sd_c = thr["Cosine"]["std"]
        T_cos = clamp(mu_c - K*sd_c, *RANGES["Cosine"])
        if not (vals["Cosine"] >= T_cos):
            ok = False

        # entropy: μ-2σ ≤ val ≤ μ+2σ
        mu_e = thr["Entropy"]["mean"]
        sd_e = thr["Entropy"]["std"]
        E_low  = clamp(mu_e - K*sd_e, *RANGES["Entropy"])
        E_high = clamp(mu_e + K*sd_e, *RANGES["Entropy"])
        if not (E_low <= vals["Entropy"] <= E_high):
            ok = False

        if ok:
            plausible += 1
        else:
            not_plausible += 1

        if w:
            w.writerow([pth, fmt,
                        vals["JSD"], vals["TVD"], vals["L1"],
                        vals["Cosine"], vals["Entropy"],
                        int(ok)])

    if fout:
        fout.close()

    # -----------------------------------------------------------
    # RISULTATO
    # -----------------------------------------------------------
    total = plausible + not_plausible
    print("\n=== RISULTATI ===")
    print(f"Formato:           {fmt}")
    print(f"Totale file:       {total}")
    print(f"Plausibili:        {plausible} ({plausible/total*100:.2f}%)")
    print(f"Non plausibili:    {not_plausible} ({not_plausible/total*100:.2f}%)")

    if args.out:
        print(f"\nMetriche salvate in: {args.out}")


if __name__ == "__main__":
    main()
