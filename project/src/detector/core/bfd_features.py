import re, zipfile, numpy as np
from config.constants import NGRAM, BUCKETS

def read_structural_pdf_bytes(path, head_size=64*1024, tail_size=8*1024):
    try:
        with open(path, "rb") as f: data = f.read()
    except Exception:
        return b""
    text = data.decode("latin1", errors="ignore") #decodifica in latin1 per mantenere i byte originali
    text_clean = re.sub(r"stream(.*?)endstream", "", text, flags=re.DOTALL|re.IGNORECASE) #rimuovi stream /endstream
    head = text_clean[:head_size]; tail = text_clean[-tail_size:] #prendi solo head e tall del file
    return (head + tail).encode("latin1", errors="ignore")#ricodifica in byte

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

# ---- JPEG (pre-SOS + finestra post-SOS “solo marker”) ----
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

# ---- TXT helper ----
def _ensure_min_len(data: bytes, min_len: int = 256) -> bytes:
    if len(data)>=min_len or len(data)==0: return data
    reps = (min_len + len(data) - 1) // len(data)
    return (data*reps)[:min_len]

# ---- N-gram / BFD (comune) ----
def ngram_bfd_from_path(path, n=NGRAM, buckets=BUCKETS):
    low = path.lower()
    if   low.endswith(".pdf"):  data = read_structural_pdf_bytes(path)
    elif low.endswith(".docx"): data = read_structural_docx_bytes(path)
    elif low.endswith(".jpg") or low.endswith(".jpeg"):
        data = read_structural_jpeg_bytes(path)
    elif low.endswith(".txt"):
        try:
            with open(path,"rb") as f: data = f.read(128*1024) # leggi fino a 128KB del file txt
        except Exception:
            data = b""
        data = _ensure_min_len(data, 256)#se il file txt è troppo corto, ripeti il contenuto fino a 256 byte
    else:
        try:
            with open(path,"rb") as f: data = f.read()
        except Exception:
            data = b""
    if not data:
        return np.zeros(256 if n==1 else buckets, dtype=float)

    arr = np.frombuffer(data, dtype=np.uint8)
    if n==1:
        c = np.bincount(arr, minlength=256).astype(float); s=c.sum()
        return c/s if s>0 else c
    if len(arr)<n: return np.zeros(buckets, dtype=float)
    if n==2 and buckets==65536:
        idx = arr[:-1].astype(np.uint32)*256 + arr[1:].astype(np.uint32)
        c = np.bincount(idx, minlength=65536).astype(float); s=c.sum()
        return c/s if s>0 else c

    # rolling hash per n>2 o buckets!=65536
    B=int(buckets) if buckets and buckets>0 else 4096
    base=257; MOD=2**64; L=len(arr)-n+1
    idx=np.zeros(L,dtype=np.uint64); val=0
    for k in range(n): val=(val*base + int(arr[k]))%MOD
    idx[0]=val; pow_base=pow(base,n-1,MOD)
    for i in range(1,L):
        out_b=int(arr[i-1]); in_b=int(arr[i+n-1])
        val=((val - (out_b*pow_base)%MOD)*base + in_b)%MOD
        idx[i]=val
    idx=(idx % B).astype(np.int64)
    c=np.bincount(idx, minlength=B).astype(float); s=c.sum()
    return c/s if s>0 else c
