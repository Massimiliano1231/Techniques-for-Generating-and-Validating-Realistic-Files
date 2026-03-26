from pathlib import Path
import zipfile

def write_txt_file(path: Path, data: bytes):
    with open(path, "wb") as f:
        f.write(data)

def write_docx_file(out_path: Path, template_files: dict[str, bytes], document_xml: bytes, styles_xml: bytes):
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_STORED) as z:
        for name, content in template_files.items():
            if name == "word/document.xml":
                z.writestr(name, document_xml)
            elif name == "word/styles.xml":
                z.writestr(name, styles_xml)
            else:
                z.writestr(name, content)


def write_jpg_file(path: Path, data: bytes):
    with open(path, "wb") as f:
        f.write(data)



def write_pdf_file(path: Path, data: bytes):
    with open(path, "wb") as f:
        f.write(data)