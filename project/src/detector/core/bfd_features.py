import re
import zipfile

import numpy as np

from detector.config.constants import NGRAM, BUCKETS


def read_structural_pdf_bytes(path, head_size=64 * 1024, tail_size=8 * 1024):
    try:
        with open(path, "rb") as f:
            data = f.read()
    except Exception:
        return b""

    text = data.decode("latin1", errors="ignore")
    text_clean = re.sub(r"stream(.*?)endstream", "", text, flags=re.DOTALL | re.IGNORECASE)
    head = text_clean[:head_size]
    tail = text_clean[-tail_size:]
    return (head + tail).encode("latin1", errors="ignore")


def read_structural_docx_bytes(path, clip_mb=None):
    text_all = ""
    try:
        with zipfile.ZipFile(path, "r") as z:
            parts = []
            for name in z.namelist():
                low = name.lower()
                if low.startswith("word/media/") or "/media/" in low or "embeddings/" in low:
                    continue
                if low.endswith(".xml") or low.endswith(".rels") or low == "[content_types].xml":
                    try:
                        parts.append(z.read(name).decode("utf-8", errors="ignore"))
                    except Exception:
                        continue
            text_all = "\n".join(parts)
    except Exception:
        pass

    if not text_all:
        try:
            with open(path, "rb") as f:
                return f.read(4096)
        except Exception:
            return b""

    if clip_mb is not None and len(text_all) > clip_mb * 1024 * 1024:
        text_all = text_all[: clip_mb * 1024 * 1024]
    return text_all.encode("latin1", errors="ignore")


def read_structural_jpeg_bytes(path, clip_kb=None, post_sos_window=1024):
    try:
        with open(path, "rb") as f:
            data = f.read()
    except Exception:
        return b""

    soi = b"\xFF\xD8"
    sos = b"\xFF\xDA"

    if not data.startswith(soi):
        return data[:256]

    sos_idx = data.find(sos)
    header = data if sos_idx == -1 else data[:sos_idx]
    if clip_kb is not None and len(header) > clip_kb * 1024:
        header = header[: clip_kb * 1024]

    markers = bytearray()
    if sos_idx != -1 and post_sos_window and post_sos_window > 0:
        win = data[sos_idx:sos_idx + post_sos_window]
        i = 0
        length = len(win)
        while i + 1 < length:
            if win[i] == 0xFF:
                j = i + 1
                while j < length and win[j] == 0xFF:
                    j += 1
                if j < length and win[j] != 0x00:
                    markers.append(0xFF)
                    markers.append(win[j])
                    i = j + 1
                else:
                    break
            else:
                i += 1
    return bytes(header) + bytes(markers)


def _ensure_min_len(data: bytes, min_len: int = 256) -> bytes:
    if len(data) >= min_len or len(data) == 0:
        return data
    reps = (min_len + len(data) - 1) // len(data)
    return (data * reps)[:min_len]


def ngram_bfd_from_path(path, n=NGRAM, buckets=BUCKETS):
    low = path.lower()
    if low.endswith(".pdf"):
        data = read_structural_pdf_bytes(path)
    elif low.endswith(".docx"):
        data = read_structural_docx_bytes(path)
    elif low.endswith(".jpg") or low.endswith(".jpeg"):
        data = read_structural_jpeg_bytes(path)
    elif low.endswith(".txt"):
        try:
            with open(path, "rb") as f:
                data = f.read(128 * 1024)
        except Exception:
            data = b""
        data = _ensure_min_len(data, 256)
    else:
        try:
            with open(path, "rb") as f:
                data = f.read()
        except Exception:
            data = b""
    if not data:
        return np.zeros(256 if n == 1 else buckets, dtype=float)

    arr = np.frombuffer(data, dtype=np.uint8)
    if n == 1:
        counts = np.bincount(arr, minlength=256).astype(float)
        total = counts.sum()
        return counts / total if total > 0 else counts

    if len(arr) < n:
        return np.zeros(buckets, dtype=float)

    if n == 2 and buckets == 65536:
        idx = arr[:-1].astype(np.uint32) * 256 + arr[1:].astype(np.uint32)
        counts = np.bincount(idx, minlength=65536).astype(float)
        total = counts.sum()
        return counts / total if total > 0 else counts

    bucket_count = int(buckets) if buckets and buckets > 0 else 4096
    base = 257
    modulus = 2**64
    length = len(arr) - n + 1
    idx = np.zeros(length, dtype=np.uint64)
    value = 0
    for k in range(n):
        value = (value * base + int(arr[k])) % modulus
    idx[0] = value
    pow_base = pow(base, n - 1, modulus)

    for i in range(1, length):
        out_b = int(arr[i - 1])
        in_b = int(arr[i + n - 1])
        value = ((value - (out_b * pow_base) % modulus) * base + in_b) % modulus
        idx[i] = value

    idx = (idx % bucket_count).astype(np.int64)
    counts = np.bincount(idx, minlength=bucket_count).astype(float)
    total = counts.sum()
    return counts / total if total > 0 else counts
