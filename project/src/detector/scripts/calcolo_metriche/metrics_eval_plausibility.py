#!/usr/bin/env python3
import os, sys, csv
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config.constants import MIN_REAL, METRIC_COL_TO_NAME, CSV_THRESHOLDS_OPT, CSV_SCORES_TEST
from io.io_utils import load_optimized_thresholds, apply_rules, load_test_rows_by_format, process_format



def main():
    formats = ["jpg", "docx", "pdf", "txt"]
    results = []

    # carica soglie ottimizzate sul TRAIN
    thr_all = load_optimized_thresholds(CSV_THRESHOLDS_OPT)

    # carica metriche del TEST per formato
    rows_by_fmt = load_test_rows_by_format(CSV_SCORES_TEST)

    print("\n=== START EVALUATION ON TEST (usando soglie del TRAIN) ===\n")
    for fmt in formats:
        print(f"\n--- {fmt.upper()} ---")
        rows = rows_by_fmt.get(fmt, [])
        r = process_format(fmt, thr_all.get(fmt, {}), rows)
        if r:
            results.append(r)

    # stampa sommario
    print("\n\n=== RISULTATI GLOBALI (TEST) ===")
    print(f"{'fmt':<6} {'N_real':>7} {'FN':>5} {'FN%':>7}   {'N_rand':>7} {'FP':>5} {'FP%':>7}")
    for r in results:
        print(
            f"{r['fmt']:<6} {r['N_real']:7d} {r['FN']:5d} {100*r['FN_rate']:6.2f}%   "
            f"{r['N_rand']:7d} {r['FP']:5d} {100*r['FP_rate']:6.2f}%"
        )



    # breakdown per metrica
    print("\n=== BREAKDOWN PER METRICA (TEST) ===")
    for r in results:
        print(f"\n[{r['fmt'].upper()}]")
        print("Metric      FN_count   FN%        FP_accepts   FP_accept_rate")
        for m in ["JSD", "TVD", "L1", "Cosine", "Entropy"]:
            fnc = r["metric_FN"][m]
            fpr = r["metric_FP"][m]
            fnp = (fnc / r["N_real"] * 100) if r["N_real"] else 0.0
            fpp = (fpr / r["N_rand"] * 100) if r["N_rand"] else 0.0
            print(f"{m:<10} {fnc:9d}  {fnp:6.2f}%   {fpr:12d}  {fpp:6.2f}%")


        # === CONFUSION MATRIX PER FORMATO ===
    print("\n=== CONFUSION MATRIX (TEST) ===")
    for r in results:
        cm = r["confusion_matrix"]
        fmt = r["fmt"].upper()

        TP = cm["TP"]
        FN = cm["FN"]
        FP = cm["FP"]
        TN = cm["TN"]

        print(f"\n[{fmt}]")
        print("              Predicted")
        print("            REAL    RANDOM")
        print(f"Actual REAL   {TP:5d}    {FN:6d}")
        print(f"Actual RANDOM {FP:5d}    {TN:6d}")


if __name__ == "__main__":
    main()
