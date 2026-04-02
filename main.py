import os
import uuid
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from pdf_utils import extract_text_from_pdf
from summarize import summarize_text, chat_with_paper

APP_TITLE = "PaperPal"
MAX_FILE_MB = 12
MAX_PAGES = 30

DOC_STORE: Dict[str, Dict[str, Any]] = {}

app = FastAPI(title=f"{APP_TITLE} (Local)")

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def ui_home():
    return FileResponse("static/index.html")

@app.get("/health")
def health():
    return {"status": "ok"}

class ChatRequest(BaseModel):
    doc_id: str
    message: str
    history: Optional[List[Dict[str, str]]] = None

class SummarizeRequest(BaseModel):
    doc_id: str

@app.post("/api/upload")
async def upload_pdf(pdf: UploadFile = File(...)):
    if pdf.content_type not in ["application/pdf", "application/octet-stream"]:
        raise HTTPException(status_code=400, detail="Please upload a PDF.")

    data = await pdf.read()
    if len(data) > MAX_FILE_MB * 1024 * 1024:
        raise HTTPException(status_code=400, detail=f"PDF too large. Max {MAX_FILE_MB}MB.")

    try:
        text, pages_processed = extract_text_from_pdf(data, max_pages=MAX_PAGES)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read PDF: {str(e)}")

    if not text.strip():
        raise HTTPException(status_code=400, detail="No extractable text found (may be scanned PDF).")

    doc_id = str(uuid.uuid4())
    DOC_STORE[doc_id] = {
        "filename": pdf.filename,
        "pages_processed": pages_processed,
        "text": text,
    }

    return {
        "doc_id": doc_id,
        "filename": pdf.filename,
        "pages_processed": pages_processed,
        "chars": len(text),
        "max_pages": MAX_PAGES
    }

@app.post("/api/summarize")
def summarize_doc(req: SummarizeRequest):
    doc = DOC_STORE.get(req.doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Unknown doc_id. Upload a PDF first.")

    try:
        summary = summarize_text(doc["text"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"doc_id": req.doc_id, "summary": summary}

@app.post("/api/chat")
def chat(req: ChatRequest):
    doc = DOC_STORE.get(req.doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Unknown doc_id. Upload a PDF first.")

    try:
        answer = chat_with_paper(doc["text"], req.message, req.history)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"doc_id": req.doc_id, "answer": answer}