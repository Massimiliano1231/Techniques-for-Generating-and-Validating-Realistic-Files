import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys
import argparse

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT / "src" / "generator"))

from io.io_utils import ensure_dir
from markov.markov_loader import load_markov_matrix, load_docx_template
from markov.markov_generator import generate_bytes_markov
from io.writer_file import write_docx_file


MARKOV_PATH = PROJECT_ROOT / "data" / "generator" / "matrices" / "P_docx.npy"
TEMPLATE_DIR = PROJECT_ROOT / "src" / "generator" / "templates" / "template_dir_docx"
OUTPUT_DIR = PROJECT_ROOT / "data" / "generator" / "generated_files" / "docx_generated"

DOCUMENT_XML_LEN = 4096
STYLES_XML_LEN = 2048

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_files", type=int, default=1000, help="Number of DOCX files to generate")
    return parser.parse_args()



def main():
    
    args = parse_args()
    NUM_FILES = args.num_files

    print("[+] Loading DOCX Markov transition matrix")
    P = load_markov_matrix(MARKOV_PATH)

    assert P.shape == (256, 256)
    assert np.allclose(P.sum(axis=1), 1.0)

    print("[+] Loading DOCX template")
    template_files = load_docx_template(TEMPLATE_DIR)

    ensure_dir(OUTPUT_DIR)

    print(f"[+] Generating {NUM_FILES} DOCX files")

    for i in tqdm(range(NUM_FILES), desc="Generating DOCX", unit="file"):
        document_xml = generate_bytes_markov(P, DOCUMENT_XML_LEN)
        styles_xml = generate_bytes_markov(P, STYLES_XML_LEN)

        out_path = OUTPUT_DIR / f"synthetic_{i:05d}.docx"

        write_docx_file(out_path,template_files,document_xml=document_xml,styles_xml=styles_xml,)

    print("[+] Done.")


if __name__ == "__main__":
    main()
