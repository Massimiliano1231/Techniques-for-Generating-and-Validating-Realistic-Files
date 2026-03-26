import sys

from pathlib import Path
from tqdm import tqdm
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT / "src" / "generator"))

from data.datasets import DATASETS
from markov.bigram_counter import init_counter, update_bigram_counts
from markov.markov_builder import normalize_rows
from formats.extractor_bytes import read_structural_pdf_bytes

OUTPUT_DIR = PROJECT_ROOT / "data" / "generator" / "matrices"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    dataset_path = DATASETS["pdf"]["real"]
    print(f"[+] Building PDF Markov model from: {dataset_path}")

    root = Path(dataset_path)
    pdf_files = [
        p for p in root.rglob("*") 
        if p.is_file() and p.suffix.lower() in [".pdf"]
    ]
    print(f"[+] Found {len(pdf_files)} PDF files.")

    C = init_counter()


    for path in tqdm(pdf_files, desc="Processing PDF files", unit="file"):
        try:
            data = read_structural_pdf_bytes(path)  
        except Exception:
            continue
        if not data or len(data) < 2:
            continue
        update_bigram_counts(C, data)

    P = normalize_rows(C)

    out_path = OUTPUT_DIR / "P_pdf.npy"
    np.save(out_path, P)

    print(f"[+] Saved PDF Markov model to: {out_path}")


if __name__ == "__main__":
    main()
