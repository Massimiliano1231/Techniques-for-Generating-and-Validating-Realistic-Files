#!/usr/bin/env python3
import os
import sys
import json
import argparse
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT / "src" / "detector"))

from data.datasets import DATASETS, get_format
from io.io_utils import list_files
from config.constants import EXTS


def scan_all_files():
    """
    Legge i file real e random da DATASETS e restituisce:

      real_by_fmt[fmt] = [paths...]
      rand_by_fmt[fmt] = [paths...]

    Filtrati con get_format() per sicurezza.
    """
    real_by_fmt = {}
    rand_by_fmt = {}

    for fmt, paths in DATASETS.items():
        real_folder = paths["real"]
        rand_folder = paths["random"]

        wanted_exts = EXTS.get(fmt, [])

        real_files = list_files(real_folder, wanted_exts)
        rand_files = list_files(rand_folder, wanted_exts)

        # filtro extra
        real_files = [p for p in real_files if get_format(p) == fmt]
        rand_files = [p for p in rand_files if get_format(p) == fmt]

        real_by_fmt[fmt] = sorted(real_files)
        rand_by_fmt[fmt] = sorted(rand_files)

    return real_by_fmt, rand_by_fmt


def split_train_test(paths, train_ratio=0.8, seed=42):
    """
    Esegue uno split train/test deterministico.
    """
    rnd = random.Random(seed)
    paths = paths[:]  # copia
    rnd.shuffle(paths)

    n = len(paths)
    n_train = int(n * train_ratio)

    return paths[:n_train], paths[n_train:]


def build_train_test_split(train_ratio=0.8, seed=42):
    """
    Restituisce la struttura:
      {
        "pdf": {
           "real_train": [...],
           "real_test": [...],
           "random_train": [...],
           "random_test": [...]
        },
        "txt": {...},
        ...
      }
    """
    real_by_fmt, rand_by_fmt = scan_all_files()
    split_struct = {}

    for fmt in sorted(real_by_fmt.keys()):
        real_files = real_by_fmt[fmt]
        rand_files = rand_by_fmt[fmt]

        print(f"[{fmt}] REAL={len(real_files)}, RANDOM={len(rand_files)}")

        real_train, real_test = split_train_test(real_files, train_ratio, seed + 13)
        rand_train, rand_test = split_train_test(rand_files, train_ratio, seed + 37)

        print(f"    real_train={len(real_train)}, real_test={len(real_test)}")
        print(f"    rand_train={len(rand_train)}, rand_test={len(rand_test)}")

        split_struct[fmt] = {
            "real_train": real_train,
            "real_test": real_test,
            "random_train": rand_train,
            "random_test": rand_test,
        }

    return split_struct


def parse_args():
    ap = argparse.ArgumentParser(description="Genera uno split train/test per ogni formato.")
    ap.add_argument("--out", required=True, help="JSON di output (es: train_test_split.json)")
    ap.add_argument("--train_ratio", type=float, default=0.8, help="Percentuale train (default 0.8).")
    ap.add_argument("--seed", type=int, default=42, help="Seed random (default 42).")
    return ap.parse_args()


def main():
    args = parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    print(f"Costruisco split train/test (train_ratio={args.train_ratio}, seed={args.seed})...")
    split_struct = build_train_test_split(
        train_ratio=args.train_ratio,
        seed=args.seed,
    )

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(split_struct, f, indent=2, ensure_ascii=False)

    print(f"\nOK, split salvato in: {args.out}")


if __name__ == "__main__":
    main()
