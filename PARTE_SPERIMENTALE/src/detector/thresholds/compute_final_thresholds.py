#!/usr/bin/env python3
import os
import sys
import csv
import json
import argparse
import numpy as np




def main():
    ap = argparse.ArgumentParser(
        description="Media soglie su più CSV (es. 3 fold) per ottenere soglie finali."
    )
    ap.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Lista di CSV di input (es. thresholds_all_formats_train_fold0.csv ... fold2.csv)",
    )
    ap.add_argument(
        "--out_csv",
        required=True,
        help="CSV di output con soglie finali (es. final_thresholds_mean.csv)",
    )
    ap.add_argument(
        "--out_json",
        default="",
        help="(Opzionale) JSON di output con soglie finali (es. final_thresholds_mean.json)",
    )

   



    args = ap.parse_args()

   
    data = {}

    for path in args.inputs:
        if not os.path.isfile(path):
            print(f"[WARN] Input CSV non trovato: {path}")
            continue

        print(f"Leggo soglie da: {path}")
        with open(path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                fmt = row["format"].strip()
                metric = row["metric"].strip()
                key = (fmt, metric)

                if key not in data:
                    data[key] = {
                        "alpha": row.get("alpha", ""),
                        "beta": row.get("beta", ""),
                        "thresholds": [],
                        "thr_lows": [],
                        "thr_highs": [],
                    }

                # threshold (per tutte le metriche tranne entropy)
                thr_str = row.get("threshold", "")
                if thr_str not in ("", None):
                    try:
                        data[key]["thresholds"].append(float(thr_str))
                    except ValueError:
                        pass

                # banda low/high (per entropy)
                low_str = row.get("thr_low", "")
                high_str = row.get("thr_high", "")
                if low_str not in ("", None):
                    try:
                        data[key]["thr_lows"].append(float(low_str))
                    except ValueError:
                        pass
                if high_str not in ("", None):
                    try:
                        data[key]["thr_highs"].append(float(high_str))
                    except ValueError:
                        pass

    if not data:
        print("Nessun dato valido trovato negli input. Esco.")
        sys.exit(1)

    # Calcolo media soglie
    results = []
    json_struct = {}  # per eventuale JSON

    for (fmt, metric), info in sorted(data.items()):
        alpha = info["alpha"]
        beta = info["beta"]

        thr_vals = info["thresholds"]
        low_vals = info["thr_lows"]
        high_vals = info["thr_highs"]

        if thr_vals:
            thr_final = np.percentile(thr_vals, 95)
        else:
            thr_final = ""

        if low_vals:
            low_final = sum(low_vals) / len(low_vals)
        else:
            low_final = ""

        if high_vals:
            high_final = sum(high_vals) / len(high_vals)
        else:
            high_final = ""

        row = {
            "format": fmt,
            "metric": metric,
            "alpha": alpha,
            "beta": beta,
            "threshold": thr_final,
            "thr_low": low_final,
            "thr_high": high_final,
        }
        results.append(row)

        # per JSON organizziamo per formato -> metric -> dict
        if fmt not in json_struct:
            json_struct[fmt] = {}
        json_struct[fmt][metric] = {
            "alpha": alpha,
            "beta": beta,
            "threshold": thr_final,
        }
        # aggiungo low/high solo se esistono
        if low_final != "" or high_final != "":
            json_struct[fmt][metric]["thr_low"] = low_final
            json_struct[fmt][metric]["thr_high"] = high_final

    # Scrivo CSV finale
    out_dir = os.path.dirname(args.out_csv) or "."
    os.makedirs(out_dir, exist_ok=True)

    fieldnames = ["format", "metric", "alpha", "beta", "threshold", "thr_low", "thr_high"]

    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"\nOK, soglie finali salvate in: {args.out_csv}")

    # Scrivo JSON finale se richiesto
    if args.out_json:
        out_dir_json = os.path.dirname(args.out_json) or "."
        os.makedirs(out_dir_json, exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as jf:
            json.dump(json_struct, jf, indent=2)
        print(f"Soglie finali (JSON) salvate in: {args.out_json}")


if __name__ == "__main__":
    main()
