"""
knowledge_pipeline/processor.py
---------------------------------
Document processing pipeline for the Construction Intelligence Platform.

Steps:
1. Read all PDFs from knowledge_base/ domain folders
2. Extract text using PyMuPDF (fitz) — page by page
3. OCR fallback for scanned/image-only pages using pytesseract
4. Clean text: strip headers, footers, page numbers, garbled unicode
5. Split into semantic chunks (800 token window, 100 overlap)
6. Preserve per-chunk metadata: title, domain, org, chapter, page, url, year
7. Save all chunks to knowledge_pipeline/chunks.json
"""

import json
import os
import re
import sys
from pathlib import Path

try:
    import fitz  # PyMuPDF
except ImportError:
    print("ERROR: PyMuPDF not installed. Run: pip install pymupdf")
    sys.exit(1)

BASE_DIR   = Path(__file__).resolve().parent.parent
KB_DIR     = BASE_DIR / "knowledge_base"
META_FILE  = BASE_DIR / "knowledge_pipeline" / "metadata.json"
LOG_FILE   = BASE_DIR / "knowledge_pipeline" / "download_log.json"
CHUNKS_OUT = BASE_DIR / "knowledge_pipeline" / "chunks.json"

CHUNK_SIZE    = 600   # characters (reduced for lower memory footprint)
CHUNK_OVERLAP = 80    # characters



def clean_text(text: str) -> str:
    """Remove noise common in academic/government PDFs."""
    # Remove repeated whitespace and non-printable characters
    text = re.sub(r"[^\x20-\x7E\n]", " ", text)
    # Collapse multiple spaces
    text = re.sub(r" {2,}", " ", text)
    # Remove page-number-like lines (lone numbers or short lines)
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip()
        # Skip lines that are purely page numbers or very short noise
        if re.match(r"^[\d\s\-–—|]{0,8}$", stripped):
            continue
        cleaned.append(stripped)
    return "\n".join(cleaned).strip()


def extract_text_from_pdf(pdf_path: Path) -> list[dict]:
    """
    Extract text from a PDF file page by page.
    Returns a list of {'page': int, 'text': str} dicts.
    Falls back to basic OCR detection if a page has no text.
    """
    pages = []
    try:
        doc = fitz.open(str(pdf_path))
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")
            text = clean_text(text)
            if len(text.strip()) < 30:
                # Page likely scanned — try OCR if pytesseract available
                try:
                    import pytesseract
                    from PIL import Image
                    import io
                    pix = page.get_pixmap(dpi=200)
                    img = Image.open(io.BytesIO(pix.tobytes("png")))
                    text = pytesseract.image_to_string(img, lang="eng")
                    text = clean_text(text)
                except Exception:
                    text = ""  # OCR not available or failed; skip page
            if text.strip():
                pages.append({"page": page_num + 1, "text": text})
        doc.close()
    except Exception as e:
        print(f"  [WARN] Could not open {pdf_path.name}: {e}")
    return pages


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """
    Generator that yields overlapping character-window chunks.
    Guarantees forward progress on every iteration.
    Tries to split at sentence boundaries when possible.
    """
    if not text:
        return
    min_advance = max(chunk_size - overlap, 50)  # must always move forward at least this much
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        if end < text_len:
            # Try sentence boundary
            boundary = text.rfind(". ", start + min_advance, end)
            if boundary == -1:
                boundary = text.rfind("\n", start + min_advance, end)
            if boundary != -1:
                end = boundary + 1
        chunk = text[start:end].strip()
        if len(chunk) > 50:
            yield chunk
        # Guaranteed forward progress: next start is at least min_advance ahead
        next_start = end - overlap
        start = max(next_start, start + min_advance)



def build_doc_meta_lookup(meta_file: Path, log_file: Path) -> dict:
    """Build a lookup from filename -> document metadata."""
    with open(meta_file, "r", encoding="utf-8") as f:
        meta = json.load(f)
    log = {}
    if log_file.exists():
        with open(log_file, "r", encoding="utf-8") as f:
            log = json.load(f)
    lookup = {}
    for doc in meta["documents"]:
        # Map by filename
        lookup[doc["filename"]] = {
            "doc_id":    doc["id"],
            "title":     doc["title"],
            "source_org": doc["source_org"],
            "domain":    doc["domain"],
            "doc_type":  doc["doc_type"],
            "pub_year":  doc["pub_year"],
            "source_url": doc.get("url", ""),
            "priority":  doc["priority"],
            "topics":    doc.get("topics", [])
        }
    return lookup


def process_all() -> int:
    """
    Walk all domain folders, extract text, chunk, and write chunks.json.
    Returns the total number of chunks produced.
    """
    doc_lookup = build_doc_meta_lookup(META_FILE, LOG_FILE)
    all_chunks = []
    chunk_id = 0

    domain_dirs = [d for d in KB_DIR.iterdir() if d.is_dir()]
    print(f"\nConstruction Intelligence Knowledge Base Processor")
    print(f"Domain folders found : {len(domain_dirs)}")
    print("-" * 60)

    for domain_dir in sorted(domain_dirs):
        pdfs = list(domain_dir.glob("*.pdf"))
        txts = list(domain_dir.glob("*.txt"))
        files = pdfs + txts
        if not files:
            print(f"  [SKIP] {domain_dir.name}/ — no documents found")
            continue
        print(f"\nProcessing domain: {domain_dir.name}/ ({len(pdfs)} PDF(s), {len(txts)} TXT(s))")

        # Process plain-text knowledge files (seeded content)
        for txt_path in txts:
            doc_meta = doc_lookup.get(txt_path.name, {
                "doc_id":     txt_path.stem,
                "title":      txt_path.stem.replace("_", " ").title(),
                "source_org": "Unknown",
                "domain":     domain_dir.name,
                "doc_type":   "knowledge_text",
                "pub_year":   2024,
                "source_url": "",
                "priority":   "High",
                "topics":     []
            })
            try:
                text = txt_path.read_text(encoding="utf-8")
                text = clean_text(text)
            except Exception as e:
                print(f"  [WARN]  Could not read {txt_path.name}: {e}")
                continue

            print(f"  Extracting: {txt_path.name}")
            doc_chunks_count = 0
            for chunk_text_str in chunk_text(text):
                all_chunks.append({
                    "chunk_id":   chunk_id,
                    "doc_id":     doc_meta["doc_id"],
                    "title":      doc_meta["title"],
                    "source_org": doc_meta["source_org"],
                    "domain":     doc_meta["domain"],
                    "doc_type":   doc_meta["doc_type"],
                    "pub_year":   doc_meta["pub_year"],
                    "source_url": doc_meta["source_url"],
                    "topics":     doc_meta["topics"],
                    "page":       1,
                    "text":       chunk_text_str
                })
                chunk_id += 1
                doc_chunks_count += 1
            print(f"    -> 1 page  |  {doc_chunks_count} chunks")

        for pdf_path in pdfs:
            doc_meta = doc_lookup.get(pdf_path.name, {
                "doc_id":     pdf_path.stem,
                "title":      pdf_path.stem.replace("_", " ").title(),
                "source_org": "Unknown",
                "domain":     domain_dir.name,
                "doc_type":   "pdf",
                "pub_year":   2000,
                "source_url": "",
                "priority":   "Medium",
                "topics":     []
            })

            print(f"  Extracting: {pdf_path.name}")
            pages = extract_text_from_pdf(pdf_path)
            if not pages:
                print(f"  [WARN]  No extractable text in {pdf_path.name}")
                continue

            doc_chunks_count = 0
            for page_info in pages:
                page_num = page_info["page"]
                page_text = page_info["text"]
                for chunk_text_str in chunk_text(page_text):
                    all_chunks.append({
                        "chunk_id":   chunk_id,
                        "doc_id":     doc_meta["doc_id"],
                        "title":      doc_meta["title"],
                        "source_org": doc_meta["source_org"],
                        "domain":     doc_meta["domain"],
                        "doc_type":   doc_meta["doc_type"],
                        "pub_year":   doc_meta["pub_year"],
                        "source_url": doc_meta["source_url"],
                        "topics":     doc_meta["topics"],
                        "page":       page_num,
                        "text":       chunk_text_str
                    })
                    chunk_id += 1
                    doc_chunks_count += 1

            print(f"    -> {len(pages)} pages  |  {doc_chunks_count} chunks")

    # Save chunks
    with open(CHUNKS_OUT, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print(f"Processing complete — Total chunks: {len(all_chunks)}")
    print(f"Chunks saved to: {CHUNKS_OUT}")
    return len(all_chunks)


if __name__ == "__main__":
    process_all()
