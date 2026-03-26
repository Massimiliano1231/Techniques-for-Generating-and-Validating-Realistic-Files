import numpy as np
import random
from pathlib import Path
from numba import njit
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT / "src" / "generator"))

from formats.jpg_helper import SOI, EOI, emit_segment, sample_next


# =====================================================
# ALIAS SAMPLING (NUMBA)
# =====================================================
@njit
def alias_sample(prob_row, alias_row):
    k = np.random.randint(256)
    if np.random.random() < prob_row[k]:
        return k
    else:
        return alias_row[k]


@njit
def gen_bytes_markov_alias(prob_mat, alias_mat, length):
    out = np.empty(length, dtype=np.uint8)
    cur = np.random.randint(256)
    out[0] = cur

    for i in range(1, length):
        cur = alias_sample(prob_mat[cur], alias_mat[cur])
        out[i] = cur

    return out


# =====================================================
# LOAD MODELS
# =====================================================
BASE_DIR = PROJECT_ROOT / "data" / "generator" / "matrices"
ALIAS_DIR = BASE_DIR / "matrici_alias"

# alias tables: shape (num_markers, 256, 256)
alias_prob = np.load(ALIAS_DIR / "P_jpg_segments_alias_prob.npy")
alias_idx  = np.load(ALIAS_DIR / "P_jpg_segments_alias_idx.npy")

# mapping marker → index
marker_map = np.load(ALIAS_DIR / "P_jpg_segments_marker_map.npy")
marker_to_idx = {int(m): i for i, m in enumerate(marker_map)}

# marker transition model
P_marker = np.load(BASE_DIR / "P_jpg_marker.npy", allow_pickle=True).item()

# JPEG constants
SOS_HDR = b"\x00\x0C\x03\x01\x00\x02\x11\x03\x11\x00\x3F\x00"
RNG = random.Random(42)


# =====================================================
# JPEG GENERATOR
# =====================================================
def generate_jpeg(target_len: int) -> bytes:
    out = bytearray(target_len)
    pos = 0

    # SOI
    out[pos:pos+2] = SOI
    pos += 2

    # start marker
    cur = list(P_marker.keys())[0]

    while pos < target_len - 2:

        # ---------------- SOS ----------------
        if cur == 0xDA:
            min_needed = 2 + len(SOS_HDR) + 2
            if pos + min_needed > target_len:
                break

            out[pos:pos+2] = b"\xFF\xDA"
            pos += 2

            out[pos:pos+len(SOS_HDR)] = SOS_HDR
            pos += len(SOS_HDR)

            remaining = target_len - pos - 2
            scan_len = min(256, remaining)

            if scan_len > 0:
                out[pos:pos+scan_len] = np.random.randint(
                0, 256, scan_len, dtype=np.uint8
                ).tobytes()

                pos += scan_len

            break

        # ---------------- NORMAL SEGMENT ----------------
        payload_len = RNG.randint(8, 128)

        mi = marker_to_idx[cur]

        payload = gen_bytes_markov_alias(
            alias_prob[mi],
            alias_idx[mi],
            payload_len
        )

        seg = emit_segment(cur, payload.tobytes())

        if pos + len(seg) > target_len - 2:
            break

        out[pos:pos+len(seg)] = seg
        pos += len(seg)

        cur = sample_next(P_marker, cur) or 0xDA

    # EOI
    out[pos:pos+2] = EOI
    return bytes(out)
