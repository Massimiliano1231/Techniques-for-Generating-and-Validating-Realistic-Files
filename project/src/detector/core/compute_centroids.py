#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from detector.config.constants import BUCKETS, NGRAM, OUT_JSON
from detector.io.io_utils import compute_centroid_for_format


def parse_args():
    parser = argparse.ArgumentParser(description="Compute format-specific real-file centroids.")
    parser.add_argument("--out_json", default=OUT_JSON, help=f"Output JSON path (default: {OUT_JSON})")
    return parser.parse_args()


def main():
    args = parse_args()
    centroids = {}

    print("Calcolo dei centroidi (BFD medie) per ogni formato...\n")

    for fmt in ["pdf", "txt", "jpg", "docx"]:
        centroid = compute_centroid_for_format(fmt, NGRAM, BUCKETS)
        if centroid is not None:
            centroids[fmt] = centroid

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    with open(out_json, "w", encoding="utf8") as f:
        json.dump(centroids, f, indent=2)

    print(f"\nOK! Centroidi salvati in: {out_json}")


if __name__ == "__main__":
    main()
