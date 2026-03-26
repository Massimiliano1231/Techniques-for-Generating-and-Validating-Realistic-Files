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
                if low.startswith("word/media/") or "/media/" in low or "embeddings/" in low:# escludi media/embeddings
                    continue
                if low.endswith(".xml") or low.endswith(".rels") or low == "[content_types].xml":
                    try:
                        parts.append(z.read(name).decode("utf-8", errors="ignore"))#metti in parts il contenuto testuale che ci serve
                    except Exception:
                        continue
            text_all = "\n".join(parts)
    except Exception:
        pass
    if not text_all:
        try:
            with open(path, "rb") as f: return f.read(4096)#fallback: leggi i primi 4KB del file se non è un docx valido
        except Exception:
            return b""
    if clip_mb is not None and len(text_all) > clip_mb*1024*1024: #limita la dimensione massima(per ora non si fa clip_mb è None)
        text_all = text_all[: clip_mb*1024*1024]
    return text_all.encode("latin1", errors="ignore")


def read_structural_jpeg_bytes(path, clip_kb=None, post_sos_window=1024):
    try:
        with open(path, "rb") as f: data = f.read()
    except Exception:
        return b""
    SOI=b"\xFF\xD8"; SOS=b"\xFF\xDA" #se non inizia con SOI(cioe 0xFFD8), prendi solo i primi 256 byte
    if not data.startswith(SOI):
        return data[:256]
    sos_idx = data.find(SOS) #trova l’indice di SOS(CIOE 0xFFDA, start of scan)
    header = data if sos_idx==-1 else data[:sos_idx] #prendi header fino a SOS (se c’è)
    if clip_kb is not None and len(header) > clip_kb*1024:
        header = header[: clip_kb*1024]
        #dai qui prendi i marker nella finestra post-SOS, che sono dati compressi dei pixel
        #che sono inutili per bfd, ma alcuni possono esserlo come i marker 0xFFxx, e qui prendiamo solo quelli
    markers = bytearray()
    if sos_idx != -1 and post_sos_window and post_sos_window>0:#estrai marker nella finestra post-SOS(finestra di 1024 byte di default)
        win = data[sos_idx : sos_idx+post_sos_window]
        i=0; L=len(win)
        while i+1<L:
            if win[i]==0xFF:
                j=i+1
                while j<L and win[j]==0xFF: j+=1#caso in cui ci sono piu 0xFF di fila
                if j<L and win[j]!=0x00:
                    markers.append(0xFF); markers.append(win[j])
                    i=j+1
                else:
                    break
            else:
                i+=1
    return bytes(header)+bytes(markers)



def read_structural_pdf_bytes(path, head_size=64*1024, tail_size=8*1024):
    try:
        with open(path, "rb") as f: data = f.read()
    except Exception:
        return b""
    text = data.decode("latin1", errors="ignore") #decodifica in latin1 per mantenere i byte originali
    text_clean = re.sub(r"stream(.*?)endstream", "", text, flags=re.DOTALL|re.IGNORECASE) #rimuovi stream /endstream
    head = text_clean[:head_size]; tail = text_clean[-tail_size:] #prendi solo head e tall del file
    return (head + tail).encode("latin1", errors="ignore")#ricodifica in byte

