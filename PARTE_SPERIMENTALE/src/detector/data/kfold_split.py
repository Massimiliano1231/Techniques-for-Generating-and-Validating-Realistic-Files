#!/usr/bin/env python3
import os
import sys
import json
import argparse
import random
from typing import List, Dict, Tuple

# Aggiungi la cartella superiore al path per importare utils.*
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from data.datasets import DATASETS, get_format
from io.io_utils import list_files
from config.constants import EXTS


def scan_all_files() -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Usa DATASETS (percorsi assoluti) + list_files + EXTS
    per costruire:
        real_by_fmt[fmt] = lista di path REAL
        rand_by_fmt[fmt] = lista di path RANDOM
    Filtra anche usando get_format(p) per sicurezza.
    """
    real_by_fmt = {}
    rand_by_fmt = {}

    for fmt, paths in DATASETS.items():
        real_folder = paths["real"]
        rand_folder = paths["random"]

        wanted_exts = EXTS.get(fmt, [])

        real_files = list_files(real_folder, wanted_exts)
        rand_files = list_files(rand_folder, wanted_exts)

        # Filtro ulteriore di sicurezza
        real_files = [p for p in real_files if get_format(p) == fmt]
        rand_files = [p for p in rand_files if get_format(p) == fmt]

        real_by_fmt[fmt] = sorted(real_files)
        rand_by_fmt[fmt] = sorted(rand_files)

    return real_by_fmt, rand_by_fmt


def make_k_folds(
    paths: List[str],
    k: int,
    seed: int,
) -> List[List[str]]:
    """
    Divide una lista di path in k fold (lista di liste) in modo deterministico.
    Le dimensioni dei fold differiscono al massimo di 1.
    Se len(paths) < k, alcuni fold saranno vuoti.
    """
    if not paths:
        return [[] for _ in range(k)]

    rnd = random.Random(seed)
    paths_shuffled = paths[:]
    rnd.shuffle(paths_shuffled)

    n = len(paths_shuffled)
    base = n // k
    remainder = n % k

    folds = []
    start = 0
    for i in range(k):
        size = base + (1 if i < remainder else 0)
        end = start + size
        folds.append(paths_shuffled[start:end])
        start = end

    return folds


def build_kfold_splits(
    k_folds: int,
    seed: int,
) -> Dict[str, Dict]:
    """
    Costruisce k fold per ogni formato:
      splits[fmt] = {
        "folds": [
          { "real": [...], "random": [...] },  # fold 0
          { "real": [...], "random": [...] },  # fold 1
          ...
        ]
      }
    """
    real_by_fmt, rand_by_fmt = scan_all_files()

    splits = {}
    for fmt in sorted(real_by_fmt.keys()):
        real_files = real_by_fmt.get(fmt, [])
        rand_files = rand_by_fmt.get(fmt, [])

        print(f"[{fmt}] REAL = {len(real_files)}, RANDOM = {len(rand_files)}")

        # Per avere shuffle indipendente per real e random, uso seed diversi ma derivati
        real_folds = make_k_folds(real_files, k_folds, seed + 13)
        rand_folds = make_k_folds(rand_files, k_folds, seed + 37)

        # Piccolo log di riepilogo
        for i in range(k_folds):
            print(
                f"    Fold {i}: REAL={len(real_folds[i])}, "
                f"RANDOM={len(rand_folds[i])}"
            )

        folds_struct = []
        for i in range(k_folds):
            folds_struct.append({
                "real": real_folds[i],
                "random": rand_folds[i],
            })

        splits[fmt] = {
            "folds": folds_struct
        }

    return splits


def parse_args():
    ap = argparse.ArgumentParser(
        description="Crea uno split K-fold (es. 3-fold) dei file REAL/RANDOM per ogni formato."
    )
    ap.add_argument(
        "--out",
        required=True,
        help="Path del file JSON di output (es: kfold_split_3.json)",
    )
    ap.add_argument(
        "--k_folds",
        type=int,
        default=3,
        help="Numero di fold K (default: 3).",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed per la randomizzazione (default: 42).",
    )
    return ap.parse_args()


def main():
    args = parse_args()

    if args.k_folds < 2:
        raise ValueError("k_folds deve essere almeno 2.")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    print(
        f"Costruisco split {args.k_folds}-fold con seed={args.seed}..."
    )
    splits = build_kfold_splits(
        k_folds=args.k_folds,
        seed=args.seed,
    )

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2, ensure_ascii=False)

    print(f"\nOK, split K-fold salvato in: {args.out}")


if __name__ == "__main__":
    main()
