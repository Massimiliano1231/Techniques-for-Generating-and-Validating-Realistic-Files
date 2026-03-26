#!/usr/bin/env python3
import os
import sys
import json
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT / "src" / "detector"))

from config.constants import NGRAM, BUCKETS
from io.io_utils import compute_centroid_for_format
from config.constants import OUT_JSON





def main():
    formats = ["pdf", "txt", "jpg", "docx"]

    centroids = {}

    print("Calcolo dei centroidi (BFD medie) per ogni formato...\n")

    for fmt in formats:
        c = compute_centroid_for_format(fmt, NGRAM, BUCKETS)
        if c is not None:
            centroids[fmt] = c

    # salva JSON
    with open(OUT_JSON, "w", encoding="utf8") as f:
        json.dump(centroids, f, indent=2)

    print(f"\nOK! Centroidi salvati in: {OUT_JSON}")


if __name__ == "__main__":
    main()
