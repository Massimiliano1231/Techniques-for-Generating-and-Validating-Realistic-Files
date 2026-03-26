from pathlib import Path

def iter_files(root: str, suffix=".txt"):
    root = Path(root)
    for p in root.rglob(f"*{suffix}"):
        if p.is_file():
            yield p

def read_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()
    

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)
