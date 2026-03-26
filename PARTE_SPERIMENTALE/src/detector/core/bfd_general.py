#!/usr/bin/env python3
# generic_texty_ngram.py
#
# Reader "quasi generico" text-vs-binary-aware:
#   - head + tail
#   - segmentazione in blocchi
#   - filtro su ratio di caratteri stampabili / entropia locale
#   - n-gram BFD sui blocchi accettati
#
# Uso esempio:
#   python3 generic_texty_ngram.py --ngram 2 --buckets 65536 file1.pdf file2.docx ...

import os
import sys
import math
import argparse
import numpy as np


# ----------------- Utility: entropia e printable ratio ----------------- #

def shannon_entropy_bytes(block: bytes) -> float:
    """
    Entropia di Shannon (base 2) su un blocco di byte.
    """
    if not block:
        return 0.0
    arr = np.frombuffer(block, dtype=np.uint8)
    counts = np.bincount(arr, minlength=256).astype(float)
    s = counts.sum()
    if s == 0:
        return 0.0
    p = counts / s
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def printable_ratio(block: bytes) -> float:
    """
    Ritorna la frazione di byte "stampabili":
    consideriamo ASCII 32-126 + whitespace (tab, newline, carriage return).
    """
    if not block:
        return 0.0
    arr = np.frombuffer(block, dtype=np.uint8)
    # range 32-126
    printable_mask = (arr >= 32) & (arr <= 126)
    # whitespace comuni
    whitespace_mask = (arr == 9) | (arr == 10) | (arr == 13)
    mask = printable_mask | whitespace_mask
    return float(mask.sum()) / float(len(arr))


# ----------------- Reader quasi generico text-vs-binary ----------------- #

def read_texty_structural_bytes(
    path: str,
    head_size: int = 64 * 1024,
    tail_size: int = 8 * 1024,
    max_mb: int = 16,
    block_size: int = 4096,
    min_printable_ratio: float = 0.4,
    max_entropy: float = 7.5,
) -> bytes:
    """
    Reader GENERICO "text-vs-binary-aware":

    1) Legge:
       - se file < max_mb: tutto
       - altrimenti: primi head_size byte + ultimi tail_size byte
    2) Spezza i byte letti in blocchi di block_size.
    3) Per ogni blocco calcola:
       - printable_ratio
       - entropia
    4) Tiene solo i blocchi che soddisfano almeno una di queste condizioni:
       - printable_ratio >= min_printable_ratio
       - entropia <= max_entropy
    5) Concatena i blocchi accettati.
       Se nessun blocco viene accettato, fallback: usa i dati originali.

    Non guarda l'estensione del file.
    """
    try:
        size = os.path.getsize(path)
    except OSError:
        return b""

    try:
        with open(path, "rb") as f:
            if size <= max_mb * 1024 * 1024:
                data = f.read()
            else:
                head = f.read(head_size)
                if tail_size > 0:
                    f.seek(max(size - tail_size, 0), os.SEEK_SET)
                    tail = f.read(tail_size)
                else:
                    tail = b""
                data = head + tail
    except Exception:
        return b""

    if not data:
        return b""

    # Segmentazione in blocchi
    blocks = []
    L = len(data)
    for i in range(0, L, block_size):
        blocks.append(data[i:i + block_size])

    kept = []
    for blk in blocks:
        if not blk:
            continue
        pr = printable_ratio(blk)
        H = shannon_entropy_bytes(blk)
        # criterio "text-vs-binary": blocco strutturale se:
        #   - abbastanza testuale, oppure
        #   - non troppo entropico
        if pr >= min_printable_ratio or H <= max_entropy:
            kept.append(blk)

    if kept:
        return b"".join(kept)
    else:
        # fallback: se il filtro elimina tutto, usa i dati originali
        return data


# ----------------- N-gram BFD sui bytes filtrati ----------------- #

def ngram_bfd_from_path_texty(
    path: str,
    n: int = 2,
    buckets: int = 65536,
    **reader_kwargs,
) -> np.ndarray:
    """
    Calcola la distribuzione n-gram (BFD) su un file
    usando il reader quasi generico text-vs-binary-aware.

    - n=1: BFD classica (vettore di 256)
    - n=2 e buckets=65536: conteggio esatto di tutti i bigrammi
    - altrimenti: hashing in 'buckets'
    """
    data = read_texty_structural_bytes(path, **reader_kwargs)
    if not data:
        return np.zeros(256 if n == 1 else buckets, dtype=float)

    arr = np.frombuffer(data, dtype=np.uint8)

    # 1-gram classico
    if n == 1:
        c = np.bincount(arr, minlength=256).astype(float)
        s = c.sum()
        return c / s if s > 0 else c

    # n-gram
    if len(arr) < n:
        return np.zeros(buckets, dtype=float)

    # caso n=2, buckets=65536 → esatto
    if n == 2 and buckets == 65536:
        idx = arr[:-1].astype(np.uint32) * 256 + arr[1:].astype(np.uint32)
        c = np.bincount(idx, minlength=65536).astype(float)
        s = c.sum()
        return c / s if s > 0 else c

    # n>2 o buckets diversi → rolling hash
    B = int(buckets)
    L = len(arr) - n + 1
    idx = np.zeros(L, dtype=np.uint64)
    base = 257
    MOD = 2 ** 64

    # primo valore
    val = 0
    for k in range(n):
        val = (val * base + int(arr[k])) % MOD
    idx[0] = val

    pow_base = pow(base, n - 1, MOD)

    for i in range(1, L):
        outb = int(arr[i - 1])
        inb = int(arr[i + n - 1])
        val = ((val - (outb * pow_base) % MOD) * base + inb) % MOD
        idx[i] = val

    idx = (idx % B).astype(np.int64)
    c = np.bincount(idx, minlength=B).astype(float)
    s = c.sum()
    return c / s if s > 0 else c
