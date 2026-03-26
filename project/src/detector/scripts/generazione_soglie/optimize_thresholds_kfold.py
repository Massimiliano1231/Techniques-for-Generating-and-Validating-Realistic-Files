#!/usr/bin/env python3
import argparse
import os,sys
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT / "src" / "detector"))
from thresholds.optimize_utils import optimize_for_fold








def main():
    ap = argparse.ArgumentParser(
        description="Ottimizza le soglie per TUTTI i fold K, usando i CSV di TRAIN per ciascun fold."
    )
    ap.add_argument(
        "--scores_dir",
        required=True,
        help=(
            "Cartella contenente i file_scores_centroid_train_fold{k}.csv "
            "(prodotti da generate_scores_kfold.py)."
        ),
    )
    ap.add_argument(
        "--k_folds",
        type=int,
        default=3,
        help="Numero di fold K (default: 3).",
    )
    ap.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Peso per FN (default: 1.0).",
    )
    ap.add_argument(
        "--beta",
        type=float,
        default=20.0,
        help="Peso per FP (default: 20.0).",
    )
    ap.add_argument(
        "--out_dir",
        required=True,
        help=(
            "Cartella di output per i CSV di soglie per fold. "
            "Per ogni fold k viene creato thresholds_all_formats_train_fold{k}.csv"
        ),
    )
    ap.add_argument(
        "--json_dir",
        default="",
        help="Cartella opzionale per salvare i JSON dettagliati (uno per combinazione).",
    )

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    for fold_idx in range(args.k_folds):
        input_csv = os.path.join(
            args.scores_dir,
            f"file_scores_centroid_train_fold{fold_idx}.csv"
        )
        if not os.path.isfile(input_csv):
            print(f"[WARN] File TRAIN per fold {fold_idx} non trovato: {input_csv}, skip.")
            continue

        out_csv = os.path.join(
            args.out_dir,
            f"thresholds_all_formats_train_fold{fold_idx}.csv"
        )

        json_dir_fold = ""
        if args.json_dir:
            json_dir_fold = os.path.join(args.json_dir, f"fold{fold_idx}")

        optimize_for_fold(
            fold_idx=fold_idx,
            input_csv=input_csv,
            alpha=args.alpha,
            beta=args.beta,
            out_csv=out_csv,
            json_dir=json_dir_fold,
        )


if __name__ == "__main__":
    main()
