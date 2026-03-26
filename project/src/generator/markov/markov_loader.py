import numpy as np
from pathlib import Path

def load_markov_matrix(path: Path) -> np.ndarray:
    P = np.load(path)
    return P

def load_docx_template(template_dir: Path) -> dict[str, bytes]:
    """
    Carica i file del template DOCX in memoria.
    """
    return {
        "[Content_Types].xml": (template_dir / "[Content_Types].xml").read_bytes(),
        "_rels/.rels": (template_dir / "_rels" / ".rels").read_bytes(),
        "word/document.xml": (template_dir / "word" / "document.xml").read_bytes(),
        "word/styles.xml": (template_dir / "word" / "styles.xml").read_bytes(),
        "word/_rels/document.xml.rels": (
            template_dir / "word" / "_rels" / "document.xml.rels"
        ).read_bytes(),
    }
