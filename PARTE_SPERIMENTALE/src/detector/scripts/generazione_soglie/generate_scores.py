#!/usr/bin/env python3
import os
import argparse
import csv
import json
import numpy as np

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT / "src" / "detector"))

from config.constants import DEFAULT_SPLIT_JSON
from io.io_utils import build_get_repr, compute_centroid, write_scores_for_group




def parse_args():
    ap = argparse.ArgumentParser(
        description=(
            "Calcola le metriche (JSD, TVD, L1, Cosine, Entropy) "
            "rispetto al centroide calcolato SOLO sui REAL_train, "
            "producendo due CSV: train e test."
        )
    )
    ap.add_argument(
        "--out",
        required=True,
        help="Cartella di output per i CSV (verranno creati file_scores_centroid_train.csv e _test.csv)",
    )
    ap.add_argument(
        "--split_json",
        default=DEFAULT_SPLIT_JSON,
        help=f"Path al JSON con lo split train/test (default: {DEFAULT_SPLIT_JSON})",
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
    return ap.parse_args()


def main():
    args = parse_args()

    # Crea cartella output
    os.makedirs(args.out, exist_ok=True)

    out_csv_train = os.path.join(args.out, "file_scores_centroid_train.csv")
    out_csv_test = os.path.join(args.out, "file_scores_centroid_test.csv")

    # Carica lo split train/test dal JSON
    print(f"Carico split train/test da: {args.split_json}")
    with open(args.split_json, "r", encoding="utf-8") as f:
        split_data = json.load(f)

    # Prep funzione per rappresentazioni BFD/n-gram
    get_repr = build_get_repr(args.ngram, args.buckets)

    # Apri entrambi i CSV di output
    header = [
        "format",
        "file",
        "class",           # "real" o "random"
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

        # Processa ciascun formato
        for fmt in ["pdf", "txt", "jpg", "docx"]:
            if fmt not in split_data:
                print(f"[{fmt}] WARNING: formato non trovato nello split JSON, skip.")
                continue

            entry = split_data[fmt]
            real_train = entry.get("real_train", [])
            real_test = entry.get("real_test", [])
            rand_train = entry.get("random_train", [])
            rand_test = entry.get("random_test", [])

            print(f"\n=== Formato {fmt} ===")
            print(f"  REAL_train={len(real_train)}, REAL_test={len(real_test)}")
            print(f"  RANDOM_train={len(rand_train)}, RANDOM_test={len(rand_test)}")

            if not real_train:
                print(f"[{fmt}] ERRORE: nessun REAL_train, non posso calcolare il centroide, skip.")
                continue

            # STEP B: centroide SOLO sui REAL_train
            centroid = compute_centroid(real_train, get_repr)
            if centroid is None:
                print(f"[{fmt}] centroide non calcolato, skip.")
                continue

            # STEP C1: metriche per TRAIN (real_train + random_train) rispetto al centroide_train
            write_scores_for_group(
                fmt=fmt,
                real_paths=real_train,
                rand_paths=rand_train,
                centroid=centroid,
                writer=w_train,
                get_repr=get_repr,
                group_name="TRAIN",
                fold_idx=0,
            )

            # STEP C2: metriche per TEST (real_test + random_test) rispetto allo stesso centroide_train
            write_scores_for_group(
                fmt=fmt,
                real_paths=real_test,
                rand_paths=rand_test,
                centroid=centroid,
                writer=w_test,
                get_repr=get_repr,
                group_name="TEST",
                fold_idx=0,
            )

    print("\nOK ->")
    print(f"  TRAIN scores: {out_csv_train}")
    print(f"  TEST  scores: {out_csv_test}")


if __name__ == "__main__":
    main()
