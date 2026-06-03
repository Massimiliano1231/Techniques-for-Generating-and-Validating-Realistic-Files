#!/usr/bin/env python3
import os
import argparse
import csv
import json

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from detector.io.io_utils import build_get_repr, compute_centroid, write_scores_for_group
from detector.config.constants import DEFAULT_KFOLD_JSON






def main():
    ap = argparse.ArgumentParser(
        description=(
            "Calcola le metriche (JSD, TVD, L1, Cosine, Entropy) "
            "rispetto al centroide calcolato SOLO sui REAL_train, "
            "in un setting K-fold (es. 3-fold). "
            "Per ogni fold k produce due CSV: train_fold_k e test_fold_k."
        )
    )
    ap.add_argument(
        "--out",
        required=True,
        help=(
            "Cartella di output per i CSV. Verranno creati, per ogni fold k, "
            "file_scores_centroid_train_fold{k}.csv e "
            "file_scores_centroid_test_fold{k}.csv"
        ),
    )
    ap.add_argument(
        "--kfold_json",
        default=DEFAULT_KFOLD_JSON,
        help=f"Path al JSON con lo split K-fold (default: {DEFAULT_KFOLD_JSON})",
    )
    ap.add_argument(
        "--ngram",
        type=int,
        default=2,
        help="Ordine n-gram (default: 2).",
    )
    ap.add_argument(
        "--buckets",
        type=int,
        default=65536,
        help="Numero di bucket per hashing n-gram (default: 65536).",
    )
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print(f"Carico split K-fold da: {args.kfold_json}")
    with open(args.kfold_json, "r", encoding="utf-8") as f:
        split_data = json.load(f)

    some_fmt = next(iter(split_data.keys()))
    k_folds = len(split_data[some_fmt]["folds"])
    print(f"Trovati {k_folds} fold (K={k_folds}) nello split.")

    get_repr = build_get_repr(args.ngram, args.buckets)

    
    formats = ["pdf", "txt", "jpg", "docx"]

    for fold_idx in range(k_folds):
        print(f"\n===============================")
        print(f"   FOLD {fold_idx} (test = fold {fold_idx})")
        print(f"===============================\n")

        out_csv_train = os.path.join(args.out, f"file_scores_centroid_train_fold{fold_idx}.csv")
        out_csv_test = os.path.join(args.out, f"file_scores_centroid_test_fold{fold_idx}.csv")

        header = [
            "format",
            "file",
            "class",
            "fold",
            "jsd_mean",
            "tvd_mean",
            "l1_mean",
            "cosine_sim_mean",
            "entropy",
        ]

        with open(out_csv_train, "w", newline="") as f_train, \
             open(out_csv_test, "w", newline="") as f_test:

            w_train = csv.writer(f_train)
            w_test = csv.writer(f_test)

            w_train.writerow(header)
            w_test.writerow(header)

            for fmt in formats:
                if fmt not in split_data:
                    print(f"[{fmt}] WARNING: formato non trovato nello split JSON, skip.")
                    continue

                folds = split_data[fmt]["folds"]

                if fold_idx >= len(folds):
                    print(f"[{fmt}] fold_idx {fold_idx} fuori range, skip.")
                    continue

                real_test = folds[fold_idx].get("real", [])
                rand_test = folds[fold_idx].get("random", [])

                real_train = []
                rand_train = []
                for j, fd in enumerate(folds):
                    if j == fold_idx:
                        continue
                    real_train.extend(fd.get("real", []))
                    rand_train.extend(fd.get("random", []))

                print(f"=== Formato {fmt}, fold={fold_idx} ===")
                print(f"  REAL_train={len(real_train)}, REAL_test={len(real_test)}")
                print(f"  RANDOM_train={len(rand_train)}, RANDOM_test={len(rand_test)}")

                if not real_train:
                    print(f"[{fmt}][fold={fold_idx}] ERRORE: nessun REAL_train, non posso calcolare il centroide, skip.")
                    continue

                centroid = compute_centroid(real_train, get_repr)
                if centroid is None:
                    print(f"[{fmt}][fold={fold_idx}] centroide non calcolato, skip.")
                    continue

                write_scores_for_group(
                    fmt=fmt,
                    real_paths=real_train,
                    rand_paths=rand_train,
                    centroid=centroid,
                    writer=w_train,
                    get_repr=get_repr,
                    group_name="TRAIN",
                    fold_idx=fold_idx,
                )

                write_scores_for_group(
                    fmt=fmt,
                    real_paths=real_test,
                    rand_paths=rand_test,
                    centroid=centroid,
                    writer=w_test,
                    get_repr=get_repr,
                    group_name="TEST",
                    fold_idx=fold_idx,
                )

        print(f"\n[fold={fold_idx}] OK ->")
        print(f"  TRAIN scores: {out_csv_train}")
        print(f"  TEST  scores: {out_csv_test}")

    print("\n=== DONE K-FOLD SCORE GENERATION ===")


if __name__ == "__main__":
    main()
