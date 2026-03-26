#!/usr/bin/env python3
import os
import sys
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from config.constants import DEFAULT_VAR_CSV, DEFAULT_SCORES_CSV
from io.io_utils import load_sigma_thresholds, load_test_rows_by_format, process_format_sigma







def main():
    ap = argparse.ArgumentParser(
        description=(
            "Valuta FN/FP usando soglie μ±2σ dal CSV delle varianze, "
            "applicate agli score in file_scores_centroid.csv."
        )
    )
    ap.add_argument(
        "--var_csv",
        default=DEFAULT_VAR_CSV,
        help=f"CSV con mean/std per ogni metrica (default: {DEFAULT_VAR_CSV})",
    )
    ap.add_argument(
        "--scores_csv",
        default=DEFAULT_SCORES_CSV,
        help=f"CSV con gli score dei file (default: {DEFAULT_SCORES_CSV})",
    )

    args = ap.parse_args()

    # carica soglie (mean/std → μ±2σ)
    thr_all = load_sigma_thresholds(args.var_csv)

    # carica metriche dallo scores_csv
    rows_by_fmt = load_test_rows_by_format(args.scores_csv)

    formats = sorted(rows_by_fmt.keys())

    results = []

    print("\n=== EVALUATION CON SOGLIE μ±2σ (da varianza) ===\n")
    for fmt in formats:
        print(f"\n--- {fmt.upper()} ---")
        rows = rows_by_fmt.get(fmt, [])
        thr_fmt = thr_all.get(fmt, {})
        r = process_format_sigma(fmt, thr_fmt, rows)
        if r:
            results.append(r)

    # stampa sommario globale
    print("\n\n=== RISULTATI GLOBALI ===")
    print(f"{'fmt':<6} {'N_real':>7} {'FN':>5} {'FN%':>7}   {'N_rand':>7} {'FP':>5} {'FP%':>7}")
    for r in results:
        print(
            f"{r['fmt']:<6} {r['N_real']:7d} {r['FN']:5d} {100*r['FN_rate']:6.2f}%   "
            f"{r['N_rand']:7d} {r['FP']:5d} {100*r['FP_rate']:6.2f}%"
        )

    # breakdown per metrica
    print("\n=== BREAKDOWN PER METRICA ===")
    for r in results:
        print(f"\n[{r['fmt'].upper()}]")
        print("Metric      FN_count   FN%        FP_accepts   FP_accept_rate")
        for m in ["JSD", "TVD", "L1", "Cosine", "Entropy"]:
            fnc = r["metric_FN"][m]
            fpr = r["metric_FP"][m]
            fnp = (fnc / r["N_real"] * 100) if r["N_real"] else 0.0
            fpp = (fpr / r["N_rand"] * 100) if r["N_rand"] else 0.0
            print(f"{m:<10} {fnc:9d}  {fnp:6.2f}%   {fpr:12d}  {fpp:6.2f}%")

    # precisione per formato
    print("\n=== PRECISIONE PER FORMATO ===")
    print(f"{'fmt':<6} {'TP':>7} {'FP':>7} {'Precision%':>12}")
    for r in results:
        TP = r["N_real"] - r["FN"]
        pred_pos = TP + r["FP"]
        precision = (TP / pred_pos) if pred_pos > 0 else 0.0
        print(f"{r['fmt']:<6} {TP:7d} {r['FP']:7d} {100*precision:11.2f}%")

    print("\n=== DONE (soglie μ±3σ) ===")


if __name__ == "__main__":
    main()
