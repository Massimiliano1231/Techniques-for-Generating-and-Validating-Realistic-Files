import sys
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from generator.data.datasets import DATASETS
from generator.markov.bigram_counter import init_counter, update_bigram_counts
from generator.markov.markov_builder import normalize_rows
from generator.formats.extractor_bytes import read_structural_docx_bytes


OUTPUT_DIR = PROJECT_ROOT / "data" / "generator" / "matrices"


def parse_args():
    parser = argparse.ArgumentParser(description="Build the DOCX Markov transition matrix.")
    parser.add_argument("--dataset", default=DATASETS["docx"]["real"])
    parser.add_argument("--out_dir", default=str(OUTPUT_DIR))
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = args.dataset
    print(f"[+] Building DOCX Markov model from: {dataset_path}")

    docx_files = list(Path(dataset_path).rglob("*.docx"))
    print(f"[+] Found {len(docx_files)} DOCX files")

    C = init_counter()

    for path in tqdm(docx_files, desc="Processing DOCX files", unit="file"):
        try:
            data = read_structural_docx_bytes(path)
        except Exception:
            continue

        if not data or len(data) < 2:
            continue

        update_bigram_counts(C, data)

    print("[+] Normalizing counts to obtain transition matrix P_docx")
    P = normalize_rows(C)

    out_path = out_dir / "P_docx.npy"
    np.save(out_path, P)

    print(f"[+] Saved DOCX Markov matrix to {out_path}")
    print("[+] Done.")


if __name__ == "__main__":
    main()
