import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASETS_ROOT = PROJECT_ROOT / "data" / "detector" / "datasets"

DATASETS = {
    "pdf": {
        "real":   str(DATASETS_ROOT / "pdf data" / "PDF-total"),
        "random": str(DATASETS_ROOT / "pdf data" / "pdf_ranflood"),
    },

    "txt": {
        "real":   str(DATASETS_ROOT / "txt data" / "TXT-total"),
        "random": str(DATASETS_ROOT / "txt data" / "txt_ranflood"),
    },

    "jpg": {
        "real":   str(DATASETS_ROOT / "jpg data" / "JPG-total"),
        "random": str(DATASETS_ROOT / "jpg data" / "jpg_ranflood"),
    },

    "docx": {
        "real":   str(DATASETS_ROOT / "docx data" / "DOCX-total"),
        "random": str(DATASETS_ROOT / "docx data" / "docx_ranflood"),
    }
}



EXT2FMT = {".pdf":"pdf", 
           ".txt":"txt", 
           ".jpg":"jpg", 
           ".jpeg":"jpg",
           ".docx":"docx"}

def get_format(path: str):
    low = path.lower()
    for ext, fmt in EXT2FMT.items():
        if low.endswith(ext):
            return fmt
    return None


REAL_SUBDIRS = {
    "pdf":  ["pdf data/PDF-total"],
    "txt":  ["txt data/TXT-total"],
    "jpg":  ["jpg data/JPG-total"],
    "docx": ["docx data/DOCX-total"],
}
