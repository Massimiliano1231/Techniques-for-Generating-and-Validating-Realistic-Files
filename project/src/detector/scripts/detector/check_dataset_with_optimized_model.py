#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np
from tqdm import tqdm
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT / "src" / "detector"))

from data.datasets import get_format
from config.constants import NGRAM, BUCKETS, DEFAULT_CENTROIDS_JSON, DEFAULT_THRESHOLDS_CSV
from core.bfd_features import ngram_bfd_from_path
from core.metrics import jsd, tvd, l1_distance, cosine_sim, entropy
from io.io_utils import load_centroids, load_final_thresholds, apply_rules_optimized








def main():
    ap = argparse.ArgumentParser(
        description=(
            "Valuta la PLAUSIBILITÀ dei file in un dataset usando "
            "centroidi pre-calcolati e soglie OTTIMIZZATE (final_thresholds)."
        )
    )
    ap.add_argument(
        "--dataset",
        required=True,
        help="Cartella con i file da valutare (pdf/txt/jpg/docx, anche ricorsivo)",
    )
    ap.add_argument(
        "--centroids",
        default=DEFAULT_CENTROIDS_JSON,
        help=f"Path al JSON con i centroidi (default: {DEFAULT_CENTROIDS_JSON})",
    )
    ap.add_argument(
        "--thr_csv",
        default=DEFAULT_THRESHOLDS_CSV,
        help=f"CSV con soglie finali (default: {DEFAULT_THRESHOLDS_CSV})",
    )
    ap.add_argument(
        "--ngram",
        type=int,
        default=NGRAM,
        help=f"Ordine n-gram (default: {NGRAM})",
    )
    ap.add_argument(
        "--buckets",
        type=int,
        default=BUCKETS,
        help=f"Numero di bucket per n-gram hashing (default: {BUCKETS})",
    )
    ap.add_argument(
        "--print_files",
        action="store_true",
        help="Se impostato, stampa l'elenco dei file plausibili/non plausibili",
    )

    args = ap.parse_args()

    if not os.path.isdir(args.dataset):
        print(f"Errore: {args.dataset} non è una cartella.")
        sys.exit(1)

    # carica centroidi e soglie
    centroids = load_centroids(args.centroids)
    thr_all = load_final_thresholds(args.thr_csv)

    plausible = []
    not_plausible = []

    metric_fail_counter = {
    "JSD": 0,
    "TVD": 0,
    "L1": 0,
    "Cosine": 0,
    "Entropy": 0,
}


    # raccogli tutti i file sotto dataset
    all_paths = []
    for root, _, files in os.walk(args.dataset):
        for name in files:
            path = os.path.join(root, name)
            all_paths.append(path)

    if not all_paths:
        print("Nessun file trovato nel dataset.")
        sys.exit(0)

    start_time = time.perf_counter()

    for pth in tqdm(all_paths, desc="Valutazione file dataset"):
        fmt = get_format(pth)
        if fmt is None:
            # estensione non supportata
            continue

        if fmt not in centroids:
            # non abbiamo centroide per questo formato
            continue
        if fmt not in thr_all:
            # non abbiamo soglie per questo formato
            continue

        centroid = centroids[fmt]
        thr_fmt = thr_all[fmt]

        # BFD / n-gram del file
        bfd = ngram_bfd_from_path(pth, n=args.ngram, buckets=args.buckets)

        # metriche rispetto al centroide
        vals = {
            "JSD":      jsd(bfd, centroid),
            "TVD":      tvd(bfd, centroid),
            "L1":       l1_distance(bfd, centroid),
            "Cosine":   cosine_sim(bfd, centroid),
            "Entropy":  entropy(bfd),
        }

        ok, decisions = apply_rules_optimized(vals, thr_fmt)

        for m, passed in decisions.items():
          if not passed:
           metric_fail_counter[m] += 1


        if ok:
            plausible.append(pth)
        else:
            not_plausible.append(pth)

    total = len(plausible) + len(not_plausible)

    print("\n=== RISULTATI MODELLO OTTIMIZZATO SUL DATASET ===")
    print(f"Cartella:          {args.dataset}")
    print(f"Totale file usati: {total}")
    print(f"Plausibili:        {len(plausible)} ({(len(plausible)/total*100 if total else 0):.2f}%)")
    print(f"Non plausibili:    {len(not_plausible)} ({(len(not_plausible)/total*100 if total else 0):.2f}%)")

    if args.print_files:
        print("\n--- FILE PLAUSIBILI ---")
        for p in plausible:
            print(p)
        print("\n--- FILE NON PLAUSIBILI ---")
        for p in not_plausible:
            print(p)


    print("\n--- METRICHE CHE HANNO FALLITO ---")
    for m, count in metric_fail_counter.items():
        print(f"{m}: {count} file non plausibili")
            
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"\nTempo totale di valutazione: {elapsed_time:.2f} secondi")
    print(f"Tempo medio per file: {(elapsed_time / total if total else 0):.6f} s/file")
if __name__ == "__main__":
    main()
