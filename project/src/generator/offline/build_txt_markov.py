import numpy as np
import sys
import argparse
from pathlib import Path
from tqdm import tqdm 

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from generator.data.datasets import DATASETS
from generator.io.io_utils import iter_files, read_bytes
from generator.formats.extractor_bytes import extract_txt_bytes
from generator.markov.bigram_counter import init_counter, update_bigram_counts
from generator.markov.markov_builder import normalize_rows


OUTPUT_DIR = PROJECT_ROOT / "data" / "generator" / "matrices"


def parse_args():
    parser = argparse.ArgumentParser(description="Build the TXT Markov transition matrix.")
    parser.add_argument("--dataset", default=DATASETS["txt"]["real"])
    parser.add_argument("--out_dir", default=str(OUTPUT_DIR))
    return parser.parse_args()


def build_txt_markov(dataset_path, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[+] Building TXT Markov model from: {dataset_path}")

    files = list(iter_files(dataset_path, suffix=".txt"))
    print(f"[+] Found {len(files)} TXT files")

    C = init_counter()

    for path in tqdm(files, desc="Processing TXT files", unit="file"):
        raw = read_bytes(path)
        data = extract_txt_bytes(raw)

        if len(data) < 2:
            continue

        update_bigram_counts(C, data)

    print("[+] Normalizing to obtain transition matrix P_txt")
    P = normalize_rows(C)

    out_path = out_dir / "P_txt.npy"
    np.save(out_path, P)

    print(f"[+] Saved TXT Markov matrix to {out_path}")
    print("[+] Done.")


if __name__ == "__main__":
    args = parse_args()
    build_txt_markov(args.dataset, args.out_dir)
