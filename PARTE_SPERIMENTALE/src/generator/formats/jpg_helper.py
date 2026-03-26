from pathlib import Path
from collections import defaultdict, Counter
import random

SOI = b"\xFF\xD8"
EOI = b"\xFF\xD9"
SOS = b"\xFF\xDA"

RNG = random.Random(42)

def read_bytes(p: Path) -> bytes:
    try:
        return p.read_bytes()
    except Exception:
        return b""

def iter_jpegs(root: Path):
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg"):
            yield p

def parse_jpeg_segments(data: bytes):
    if not data.startswith(SOI):
        return [], {}, []

    i = 2
    L = len(data)
    markers_seq = []
    segments = defaultdict(list)
    sos_scans = []

    while i + 1 < L:
        if data[i] != 0xFF:
            i += 1
            continue

        j = i + 1
        while j < L and data[j] == 0xFF:
            j += 1
        if j >= L:
            break

        marker = data[j]
        i = j + 1

        if marker == 0xD9:
            markers_seq.append(marker)
            break

        if marker == 0xDA:
            markers_seq.append(marker)
            seglen = int.from_bytes(data[i:i+2], "big")
            hdr_end = i + seglen
            scan_start = hdr_end
            eoi = data.find(EOI, scan_start)
            scan = data[scan_start:eoi if eoi != -1 else L]
            sos_scans.append(scan)
            break

        seglen = int.from_bytes(data[i:i+2], "big")
        payload_start = i + 2
        payload_end = i + seglen
        if payload_end > L:
            break

        markers_seq.append(marker)
        segments[marker].append(data[payload_start:payload_end])
        i = payload_end

    return markers_seq, segments, sos_scans

def build_bigram_markov(seqs):
    C = defaultdict(Counter)
    for s in seqs:
        for a, b in zip(s[:-1], s[1:]):
            C[a][b] += 1
    return {
        a: [(b, C[a][b] / sum(C[a].values())) for b in C[a]]
        for a in C
    }

def sample_next(P, cur):
    if cur not in P:
        return None
    r = RNG.random()
    acc = 0.0
    for b, p in P[cur]:
        acc += p
        if r <= acc:
            return b
    return P[cur][-1][0]

def build_byte_markov(byte_arrays):
    C = defaultdict(Counter)
    for arr in byte_arrays:
        for a, b in zip(arr[:-1], arr[1:]):
            C[a][b] += 1
    return {
        a: [(b, C[a][b] / sum(C[a].values())) for b in C[a]]
        for a in C
    }

def gen_bytes_markov(P, length):
    cur = RNG.choice(list(P.keys()))
    out = bytearray([cur])
    while len(out) < length:
        nxt = sample_next(P, cur)
        if nxt is None:
            cur = RNG.choice(list(P.keys()))
        else:
            cur = nxt
        out.append(cur)
    return bytes(out)

def emit_segment(marker, payload):
    return b"\xFF" + bytes([marker]) + (len(payload) + 2).to_bytes(2, "big") + payload


