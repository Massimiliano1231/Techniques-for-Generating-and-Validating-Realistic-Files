import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys
import argparse

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT / "src" / "generator"))

from io.io_utils import ensure_dir
from markov.markov_loader import load_markov_matrix
from markov.markov_generator import generate_bytes_markov
from io.writer_file import write_txt_file



MARKOV_PATH = PROJECT_ROOT / "data" / "generator" / "matrices" / "P_txt.npy"
OUTPUT_DIR = PROJECT_ROOT / "data" / "generator" / "generated_files" / "txt_generated"

FILE_LENGTH = 2048  

def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic DOCX files using a Markov model")
    parser.add_argument("--num_files", type=int, default=1000, help="Number of DOCX files to generate")
    return parser.parse_args()



def main():

    args = parse_args()
    NUM_FILES = args.num_files

    print("[+] Loading Markov transition matrix")
    P = load_markov_matrix(MARKOV_PATH)

    assert P.shape == (256, 256)
    assert np.allclose(P.sum(axis=1), 1.0)

    ensure_dir(OUTPUT_DIR)

    print(f"[+] Generating {NUM_FILES} TXT files")

    for i in tqdm(range(NUM_FILES), desc="Generating TXT", unit="file"):
        data = generate_bytes_markov(P=P,length=FILE_LENGTH)

        out_path = OUTPUT_DIR / f"synthetic_{i:05d}.txt"
        write_txt_file(out_path, data)

    print("[+] Done.")


if __name__ == "__main__":
    main()
