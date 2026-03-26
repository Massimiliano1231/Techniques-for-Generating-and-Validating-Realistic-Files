from docx import Document
from pathlib import Path

OUTPUT = Path("template.docx")

def create_template():
    doc = Document()

    doc.save(OUTPUT)

    print(f"[+] Template DOCX creato: {OUTPUT.resolve()}")

if __name__ == "__main__":
    create_template()
