import os
import random
import time
import numpy as np
from pathlib import Path


SIGNATURES = {
    ".txt":  bytes.fromhex("EF BB BF"),
    ".pdf":  bytes.fromhex("25 50 44 46 2D 31 2E 36"),
    ".docx": bytes.fromhex("50 4B 03 04"),
    ".jpg":  bytes.fromhex("FF D8 FF E0"),
}

EXTENSIONS = list(SIGNATURES.keys())


def random_length(min_len, max_len):
    return int(np.exp(
        np.random.uniform(np.log(min_len), np.log(max_len))
    ))


NUM_FILES = 10_000

TXT_MIN,  TXT_MAX  = 1024,   4096
PDF_MIN,  PDF_MAX  = 1024,   4096
DOCX_MIN, DOCX_MAX = 1024,   4096
JPG_MIN,  JPG_MAX  = 2048,   4096

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = PROJECT_ROOT / "data" / "generator" / "generated_files" / "flooding_completamente_casuale"
OUT_DIR.mkdir(exist_ok=True)

from numba import njit

@njit
def generate_bytes_random_numba(length):
    out = np.empty(length, dtype=np.uint8)
    for i in range(length):
        out[i] = np.random.randint(256)
    return out

def generate_bytes_random(length) -> bytes:
    return generate_bytes_random_numba(length).tobytes()

# =========================
# RANDOM FLOODING
# =========================
def random_flood():
    start = time.perf_counter()

    for i in range(NUM_FILES):
        ext = random.choice(EXTENSIONS)
        signature = SIGNATURES[ext]

        if ext == ".txt":
            size = random_length(TXT_MIN, TXT_MAX)
        elif ext == ".pdf":
            size = random_length(PDF_MIN, PDF_MAX)
        elif ext == ".docx":
            size = random_length(DOCX_MIN, DOCX_MAX)
        elif ext == ".jpg":
            size = random_length(JPG_MIN, JPG_MAX)

        size = max(size, len(signature) + 1)

        payload = generate_bytes_random(size - len(signature))
        data = signature + payload

        out_path = OUT_DIR / f"random_{i:05d}{ext}"
        with open(out_path, "wb") as f:
            f.write(data)

    end = time.perf_counter()
    return end - start


if __name__ == "__main__":
    t = random_flood()
    print(f"[RandomFlood-4formats] Generated {NUM_FILES} files in {t:.2f}s")
