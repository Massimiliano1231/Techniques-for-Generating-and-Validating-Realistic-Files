import numpy as np

def init_counter():
    return np.zeros((256, 256), dtype=np.uint64)

def update_bigram_counts(C: np.ndarray, data: bytes):
    if len(data) < 2:
        return

    b = np.frombuffer(data, dtype=np.uint8)
    y = b[:-1]
    x = b[1:]

    np.add.at(C, (y, x), 1)

