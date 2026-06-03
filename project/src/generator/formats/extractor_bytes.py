import zipfile
import re


def extract_txt_bytes(raw: bytes) -> bytes:
    return raw


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
