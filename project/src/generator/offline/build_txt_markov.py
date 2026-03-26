import numpy as np
import sys
from pathlib import Path
from tqdm import tqdm 

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT / "src" / "generator"))

from data.datasets import DATASETS
from io.io_utils import iter_files, read_bytes
from formats.extractor_bytes import extract_txt_bytes
from markov.bigram_counter import init_counter, update_bigram_counts
from markov.markov_builder import normalize_rows


OUTPUT_DIR = PROJECT_ROOT / "data" / "generator" / "matrices"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def build_txt_markov():
    dataset_path = DATASETS["txt"]["real"]
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

    out_path = OUTPUT_DIR / "P_txt.npy"
    np.save(out_path, P)

    print(f"[+] Saved TXT Markov matrix to {out_path}")
    print("[+] Done.")


if __name__ == "__main__":
    build_txt_markov()
