from pdfminer.high_level import extract_text
import fitz  # PyMuPDF


def extract_pdf_text(pdf_path: str) -> str:
    """
    First try pdfminer.
    If result is too short, try PyMuPDF.
    """
    text = (extract_text(pdf_path) or "").strip()

    if len(text) >= 200:
        return text

    try:
        doc = fitz.open(pdf_path)
        pages = []
        for page in doc:
            pages.append(page.get_text("text"))
        fitz_text = "\n".join(pages).strip()

        if len(fitz_text) > len(text):
            return fitz_text
    except Exception:
        pass

    return text