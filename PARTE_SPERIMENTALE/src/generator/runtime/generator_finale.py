import time
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys
import argparse

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT / "src" / "generator"))

from io.io_utils import ensure_dir
from markov.markov_loader import load_docx_template
from markov.markov_generator import generate_bytes_markov
from io.writer_file import write_docx_file, write_pdf_file, write_txt_file
from runtime.generate_jpg import generate_jpeg


# =========================
# PATHS
# =========================

MARKOV_BASE = PROJECT_ROOT / "data" / "generator" / "matrices"
ALIAS_DIR = MARKOV_BASE / "matrici_alias"

MARKOV_PATH_DOCX_PROB = ALIAS_DIR / "P_docx_alias_prob.npy"
MARKOV_PATH_DOCX_IDX  = ALIAS_DIR / "P_docx_alias_idx.npy"

MARKOV_PATH_PDF_PROB  = ALIAS_DIR / "P_pdf_alias_prob.npy"
MARKOV_PATH_PDF_IDX   = ALIAS_DIR / "P_pdf_alias_idx.npy"

MARKOV_PATH_TXT_PROB  = ALIAS_DIR / "P_txt_alias_prob.npy"
MARKOV_PATH_TXT_IDX   = ALIAS_DIR / "P_txt_alias_idx.npy"

TEMPLATE_DIR = PROJECT_ROOT / "src" / "generator" / "templates" / "template_dir_docx"
OUTPUT_DIR = PROJECT_ROOT / "data" / "generator" / "generated_files" / "all_generated"


# =========================
# LENGTH RANGES
# =========================

DOCX_XML_MIN = 1024

PDF_MIN = 1024

TXT_MIN = 512

JPG_MIN = 1024


# =========================
# ARGS
# =========================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_files", type=int, default=1000)
    parser.add_argument("--max_len", type=int, default=4096)
    return parser.parse_args()


def random_length(min_len, max_len):
    return int(np.exp(
        np.random.uniform(np.log(min_len), np.log(max_len))
    ))


# =========================
# MAIN
# =========================

def main():
    args = parse_args()
    NUM_FILES = args.num_files
    MAX_LEN = args.max_len

    ensure_dir(OUTPUT_DIR)

    # ---- distribuzione deterministica ----
    n_each = NUM_FILES // 4
    remainder = NUM_FILES % 4

    n_docx = n_each + (1 if remainder > 0 else 0)
    n_pdf  = n_each + (1 if remainder > 1 else 0)
    n_jpg  = n_each + (1 if remainder > 2 else 0)
    n_txt  = n_each

    print("[+] Loading alias tables")

    alias_docx_prob = np.load(MARKOV_PATH_DOCX_PROB)
    alias_docx_idx  = np.load(MARKOV_PATH_DOCX_IDX)

    alias_pdf_prob  = np.load(MARKOV_PATH_PDF_PROB)
    alias_pdf_idx   = np.load(MARKOV_PATH_PDF_IDX)

    alias_txt_prob  = np.load(MARKOV_PATH_TXT_PROB)
    alias_txt_idx   = np.load(MARKOV_PATH_TXT_IDX)

    print("[+] Loading DOCX template")
    template_files = load_docx_template(TEMPLATE_DIR)

    idx = 0




    print("[+] Starting generation")
    start_time = time.perf_counter()

    # =========================
    # DOCX
    # =========================
    print(f"[+] Generating {n_docx} DOCX files")
    for _ in tqdm(range(n_docx), desc="DOCX", unit="file"):
        document_len = random_length(DOCX_XML_MIN, MAX_LEN) // 2

        document_xml = generate_bytes_markov(alias_docx_prob, alias_docx_idx, document_len)
        styles_xml   = generate_bytes_markov(alias_docx_prob, alias_docx_idx, document_len)

        out_path = OUTPUT_DIR / f"synthetic_{idx:05d}.docx"
        write_docx_file(
            out_path,
            template_files,
            document_xml=document_xml,
            styles_xml=styles_xml,
        )
        idx += 1
    

    # =========================
    # PDF
    # =========================
    print(f"[+] Generating {n_pdf} PDF files")
    for _ in tqdm(range(n_pdf), desc="PDF", unit="file"):
        pdf_len = random_length(PDF_MIN, MAX_LEN)
        data = generate_bytes_markov(alias_pdf_prob, alias_pdf_idx, pdf_len)

        out_path = OUTPUT_DIR / f"synthetic_{idx:05d}.pdf"
        write_pdf_file(out_path, data)
        idx += 1

    # =========================
    # JPG
    # =========================
    print(f"[+] Generating {n_jpg} JPG files")
    for _ in tqdm(range(n_jpg), desc="JPG", unit="file"):
        jpg_len = random_length(JPG_MIN, MAX_LEN)
        jpg_bytes = generate_jpeg(jpg_len)

        out_path = OUTPUT_DIR / f"synthetic_{idx:05d}.jpg"
        out_path.write_bytes(jpg_bytes)
        idx += 1

    # =========================
    # TXT
    # =========================
    print(f"[+] Generating {n_txt} TXT files")
    for _ in tqdm(range(n_txt), desc="TXT", unit="file"):
        txt_len = random_length(TXT_MIN, MAX_LEN)
        data = generate_bytes_markov(alias_txt_prob, alias_txt_idx, txt_len)

        out_path = OUTPUT_DIR / f"synthetic_{idx:05d}.txt"
        write_txt_file(out_path, data)
        idx += 1

    end_time = time.perf_counter()
    print(f"[+] Done. Generated {idx} files in {OUTPUT_DIR}")
    print(f"[+] Total generation time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()







"""
import time
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys
import argparse

sys.path.append(str(PROJECT_ROOT / "src" / "generator"))

from io.io_utils import ensure_dir
from markov.markov_loader import load_markov_matrix, load_docx_template
from markov.markov_generator import generate_bytes_markov
from io.writer_file import write_docx_file, write_pdf_file, write_txt_file
from runtime.generate_jpg import generate_jpeg


MARKOV_PATH_DOCX = MARKOV_BASE / "P_docx.npy"
MARKOV_PATH_PDF  = MARKOV_BASE / "P_pdf.npy"
MARKOV_PATH_TXT  = MARKOV_BASE / "P_txt.npy"

TEMPLATE_DIR = PROJECT_ROOT / "src" / "generator" / "templates" / "template_dir_docx"

OUTPUT_DIR = PROJECT_ROOT / "data" / "generator" / "generated_files" / "all_generated"

MARKOV_PATH_DOCX_PROB = ALIAS_DIR / "P_docx_alias_prob.npy"
MARKOV_PATH_DOCX_IDX  = ALIAS_DIR / "P_docx_alias_idx.npy"

MARKOV_PATH_PDF_PROB  = ALIAS_DIR / "P_pdf_alias_prob.npy"
MARKOV_PATH_PDF_IDX   = ALIAS_DIR / "P_pdf_alias_idx.npy"

MARKOV_PATH_TXT_PROB  = ALIAS_DIR / "P_txt_alias_prob.npy"
MARKOV_PATH_TXT_IDX   = ALIAS_DIR / "P_txt_alias_idx.npy"


DOCX_XML_MIN = 1024
DOCX_XML_MAX = 4096

PDF_MIN = 1024
PDF_MAX = 4096

TXT_MIN = 512
TXT_MAX = 4096

JPG_MIN = 1024
JPG_MAX = 4096


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_files", type=int, default=1000, help="Number of DOCX files to generate")
    return parser.parse_args()


def random_length(min_len, max_len):
    return int(np.exp(
        np.random.uniform(np.log(min_len), np.log(max_len))
    ))

formats = ["docx", "pdf", "jpg", "txt"]


def main():
    args = parse_args()
    NUM_FILES = args.num_files

    ensure_dir(OUTPUT_DIR)
    
    n_each = NUM_FILES // 4
    remainder = NUM_FILES % 4

    n_docx = n_each + (1 if remainder > 0 else 0)
    n_pdf  = n_each + (1 if remainder > 1 else 0)
    n_jpg  = n_each + (1 if remainder > 2 else 0)
    n_txt  = n_each
    

    print("[+] Loading alias tables")

    alias_docx_prob = np.load(MARKOV_PATH_DOCX_PROB)
    alias_docx_idx  = np.load(MARKOV_PATH_DOCX_IDX)

    alias_pdf_prob  = np.load(MARKOV_PATH_PDF_PROB)
    alias_pdf_idx   = np.load(MARKOV_PATH_PDF_IDX)

    alias_txt_prob  = np.load(MARKOV_PATH_TXT_PROB)
    alias_txt_idx   = np.load(MARKOV_PATH_TXT_IDX)


    print("[+] Loading DOCX template")
    template_files = load_docx_template(TEMPLATE_DIR)

    idx = 0  # contatore globale
    

    print("[+] Starting generation")
    start_time = time.perf_counter()

    for _ in tqdm(range(NUM_FILES), disable=True, desc="Generating files", unit="file"):

        fmt = np.random.choice(formats)

        if fmt == "docx":
          document_len = random_length(DOCX_XML_MIN, DOCX_XML_MAX) // 2
          document_xml = generate_bytes_markov(alias_docx_prob, alias_docx_idx, document_len)
          styles_xml   = generate_bytes_markov(alias_docx_prob, alias_docx_idx, document_len)

          out_path = OUTPUT_DIR / f"synthetic_{idx:05d}.docx"
          write_docx_file(
            out_path,
            template_files,
            document_xml=document_xml,
            styles_xml=styles_xml,
        )

        elif fmt == "pdf":
          pdf_len = random_length(PDF_MIN, PDF_MAX)
          data = generate_bytes_markov(alias_pdf_prob, alias_pdf_idx, pdf_len)
          out_path = OUTPUT_DIR / f"synthetic_{idx:05d}.pdf"
          write_pdf_file(out_path, data)

        elif fmt == "jpg":
          jpg_len = random_length(JPG_MIN, JPG_MAX)
          jpg_bytes = generate_jpeg(jpg_len)
          out_path = OUTPUT_DIR / f"synthetic_{idx:05d}.jpg"
          out_path.write_bytes(jpg_bytes)

        else:  # txt
          txt_len = random_length(TXT_MIN, TXT_MAX)
          data = generate_bytes_markov(alias_txt_prob, alias_txt_idx, txt_len)
          out_path = OUTPUT_DIR / f"synthetic_{idx:05d}.txt"
          write_txt_file(out_path, data)

        idx += 1
    

    # -------- DOCX --------
    print(f"[+] Generating {n_docx} DOCX files")
    for _ in tqdm(range(n_docx), desc="DOCX", unit="file"):
        document_len = random_length(DOCX_XML_MIN, DOCX_XML_MAX) // 2

         document_xml = generate_bytes_markov(alias_docx_prob, alias_docx_idx, document_len)
    styles_xml   = generate_bytes_markov(alias_docx_prob, alias_docx_idx, document_len)

    out_path = OUTPUT_DIR / f"synthetic_{idx:05d}.docx"
    write_docx_file(
        out_path,
        template_files,
        document_xml=document_xml,
        styles_xml=styles_xml,
    )
    idx += 1

    
    print("[+] Starting generation")
    start_time = time.perf_counter()
    
    print(f"[+] Generating {n_docx} DOCX files")
    for _ in tqdm(range(n_docx), desc="DOCX", unit="file"):

        document_len = random_length(DOCX_XML_MIN, DOCX_XML_MAX)
        document_len = document_len // 2

        document_xml = generate_bytes_markov(alias_docx_prob, alias_docx_idx, document_len)
        styles_xml   = generate_bytes_markov(alias_docx_prob, alias_docx_idx, document_len)

        out_path = OUTPUT_DIR / f"synthetic_{idx:05d}.docx"
        write_docx_file(
            out_path,
            template_files,
            document_xml=document_xml,
            styles_xml=styles_xml,
        )
        idx += 1

    # -------- PDF --------
    print(f"[+] Generating {n_pdf} PDF files")
    for _ in tqdm(range(n_pdf), desc="PDF", unit="file"):
        pdf_len = random_length(PDF_MIN, PDF_MAX)
        data = generate_bytes_markov(alias_pdf_prob, alias_pdf_idx, pdf_len)
        out_path = OUTPUT_DIR / f"synthetic_{idx:05d}.pdf"
        write_pdf_file(out_path, data)
        idx += 1

    # -------- JPG --------
    print(f"[+] Generating {n_jpg} JPG files")
    for _ in tqdm(range(n_jpg), desc="JPG", unit="file"):
        jpg_len = random_length(JPG_MIN, JPG_MAX)
        jpg_bytes = generate_jpeg(jpg_len)
        out_path = OUTPUT_DIR / f"synthetic_{idx:05d}.jpg"
        out_path.write_bytes(jpg_bytes)
        idx += 1

    # -------- TXT --------
    print(f"[+] Generating {n_txt} TXT files")
    for _ in tqdm(range(n_txt), desc="TXT", unit="file"):
        txt_len = random_length(TXT_MIN, TXT_MAX)
        data = generate_bytes_markov(alias_txt_prob, alias_txt_idx, txt_len)
        out_path = OUTPUT_DIR / f"synthetic_{idx:05d}.txt"
        write_txt_file(out_path, data)
        idx += 1
        


    print(f"[+] Done. Generated {idx} files in {OUTPUT_DIR}")
    end_time = time.perf_counter()
    print(f"[+] Total generation time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
# """

