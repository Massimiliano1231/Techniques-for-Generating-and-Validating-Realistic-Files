#!/usr/bin/env python3
import os
import sys
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT / "src"))
sys.path.append(str(PROJECT_ROOT / "src" / "detector"))

import numpy as np
from detector.config.constants import DEFAULT_THRESH_DIR, DEFAULT_SCORES_DIR
from detector.io.io_utils import load_optimized_thresholds, load_test_rows_by_format, process_format, apply_rules



def main():
    import argparse

    ap = argparse.ArgumentParser(
        description=(
            "Valuta FN/FP in K-fold cross-validation, "
            "usando le soglie ottimizzate su ciascun TRAIN-fold."
        )
    )
    ap.add_argument(
        "--thresholds_dir",
        default=DEFAULT_THRESH_DIR,
        help=f"Cartella con thresholds_all_formats_train_fold{{k}}.csv (default: {DEFAULT_THRESH_DIR})",
    )
    ap.add_argument(
        "--scores_dir",
        default=DEFAULT_SCORES_DIR,
        help=f"Cartella con file_scores_centroid_test_fold{{k}}.csv (default: {DEFAULT_SCORES_DIR})",
    )
    ap.add_argument(
        "--k_folds",
        type=int,
        default=3,
        help="Numero di fold K (default: 3).",
    )
    args = ap.parse_args()

    formats = ["jpg", "docx", "pdf", "txt"]

    # Per aggregare su tutti i fold
    agg_by_fmt = {}
    for fmt in formats:
        agg_by_fmt[fmt] = {
            "N_real": 0,
            "N_rand": 0,
            "FN": 0,
            "FP": 0,
            "metric_FN": {m: 0 for m in ["JSD", "TVD", "L1", "Cosine", "Entropy"]},
            "metric_FP": {m: 0 for m in ["JSD", "TVD", "L1", "Cosine", "Entropy"]},
        }

    print("\n=== START K-FOLD EVALUATION (usando soglie per ogni fold) ===\n")

    # Risultati per fold
    results_per_fold = {}

    for fold_idx in range(args.k_folds):
        thr_csv = os.path.join(
            args.thresholds_dir,
            f"thresholds_all_formats_train_fold{fold_idx}.csv"
        )
        scores_csv = os.path.join(
            args.scores_dir,
            f"file_scores_centroid_test_fold{fold_idx}.csv"
        )

        if not (os.path.isfile(thr_csv) and os.path.isfile(scores_csv)):
            print(f"\n[fold={fold_idx}] File mancanti, skip.")
            print(f"  thresholds: {thr_csv}")
            print(f"  scores    : {scores_csv}")
            continue

        print(f"\n--- FOLD {fold_idx} ---")
        print(f"  thresholds: {thr_csv}")
        print(f"  scores    : {scores_csv}")

        thr_all = load_optimized_thresholds(thr_csv)
        rows_by_fmt = load_test_rows_by_format(scores_csv)

        fold_results = []

        for fmt in formats:
            print(f"\n[{fmt.upper()}] (fold {fold_idx})")
            rows = rows_by_fmt.get(fmt, [])
            r = process_format(fmt, thr_all.get(fmt, {}), rows) #qui applica le soglie e calcola FN/FP
            #torna dizionario con risultati
            if r:
                fold_results.append(r)

                # aggiorna aggregato
                agg = agg_by_fmt[fmt]
                agg["N_real"] += r["N_real"]
                agg["N_rand"] += r["N_rand"]
                agg["FN"] += r["FN"]
                agg["FP"] += r["FP"]
                for m in agg["metric_FN"].keys():
                    agg["metric_FN"][m] += r["metric_FN"][m]
                    agg["metric_FP"][m] += r["metric_FP"][m]

        results_per_fold[fold_idx] = fold_results

        # stampa sommario fold
        print("\nRISULTATI FOLD (per formato):")
        print(f"{'fmt':<6} {'N_real':>7} {'FN':>5} {'FN%':>7}   {'N_rand':>7} {'FP':>5} {'FP%':>7}")
        for r in fold_results:
            print(
                f"{r['fmt']:<6} {r['N_real']:7d} {r['FN']:5d} {100*r['FN_rate']:6.2f}%   "
                f"{r['N_rand']:7d} {r['FP']:5d} {100*r['FP_rate']:6.2f}%" #6.2f = float con 2 decimali e 6 spazi totali
            )

    # --- AGGREGATI SU TUTTI I FOLD ---

    print("\n\n=== RISULTATI GLOBALI SU TUTTI I FOLD ===")
    print(f"{'fmt':<6} {'N_real':>7} {'FN':>5} {'FN%':>7}   {'N_rand':>7} {'FP':>5} {'FP%':>7}") #es: <6 : orizzontale 6 caratteri

    for fmt in formats:
        agg = agg_by_fmt[fmt]
        N_real = agg["N_real"]
        N_rand = agg["N_rand"]
        FN = agg["FN"]
        FP = agg["FP"]

        if N_real == 0 and N_rand == 0:
            continue

        FN_rate = (FN / N_real) if N_real else 0.0
        FP_rate = (FP / N_rand) if N_rand else 0.0

        print(
            f"{fmt:<6} {N_real:7d} {FN:5d} {100*FN_rate:6.2f}%   "
            f"{N_rand:7d} {FP:5d} {100*FP_rate:6.2f}%"
        )

    print("\n=== BREAKDOWN PER METRICA (AGGREGATO SU TUTTI I FOLD) ===")
    for fmt in formats:
        agg = agg_by_fmt[fmt]
        N_real = agg["N_real"]
        N_rand = agg["N_rand"]
        if N_real == 0 and N_rand == 0:
            continue

        print(f"\n[{fmt.upper()}]")
        print("Metric      FN_count   FN%        FP_accepts   FP_accept_rate")
        for m in ["JSD", "TVD", "L1", "Cosine", "Entropy"]:
            fnc = agg["metric_FN"][m]
            fpr = agg["metric_FP"][m]
            fnp = (fnc / N_real * 100) if N_real else 0.0
            fpp = (fpr / N_rand * 100) if N_rand else 0.0
            print(f"{m:<10} {fnc:9d}  {fnp:6.2f}%   {fpr:12d}  {fpp:6.2f}%")
    print("\n=== CONFUSION MATRIX (AGGREGATA SU TUTTI I FOLD) ===")
    for fmt in formats:
        agg = agg_by_fmt[fmt]
        N_real = agg["N_real"]
        N_rand = agg["N_rand"]
        FN = agg["FN"]
        FP = agg["FP"]

        if N_real == 0 and N_rand == 0:
            continue

        # Confusion matrix aggregata
        TP = N_real - FN
        TN = N_rand - FP

        print(f"\n[{fmt.upper()}]")
        print("              Predicted")
        print("            REAL    RANDOM")
        print(f"Actual REAL   {TP:5d}    {FN:6d}")
        print(f"Actual RANDOM {FP:5d}    {TN:6d}")

    print("\n=== DONE K-FOLD EVALUATION ===")


    

    print("\n=== DONE K-FOLD EVALUATION ===")


if __name__ == "__main__":
    main()
