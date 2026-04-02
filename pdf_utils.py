from pypdf import PdfReader
import io
from typing import Tuple

def extract_text_from_pdf(file_bytes: bytes, max_pages: int = 30) -> Tuple[str, int]:
    """
    Extract text from a PDF (text-based PDFs). Scanned PDFs may return little/no text.
    Returns (text, pages_processed).
    """
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = reader.pages[:max_pages]

    texts = []
    pages_processed = 0

    for i, page in enumerate(pages):
        t = page.extract_text() or ""
        t = " ".join(t.split())
        if t:
            texts.append(f"[PAGE {i+1}] {t}")
        pages_processed += 1

    return "\n".join(texts), pages_processed