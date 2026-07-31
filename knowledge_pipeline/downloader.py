"""
knowledge_pipeline/downloader.py
---------------------------------
Automated document download pipeline for the Construction Intelligence Platform.

Features:
- Downloads PDFs from the curated metadata.json inventory
- SHA-256 deduplication (skips files already downloaded)
- Organises files into domain-specific sub-folders under knowledge_base/
- Writes a download_log.json with per-file status, size, timestamp
- Supports incremental updates (new entries in metadata.json trigger targeted downloads)
"""

import json
import os
import hashlib
import datetime
import time
import sys
import requests
from pathlib import Path

BASE_DIR    = Path(__file__).resolve().parent.parent
KB_DIR      = BASE_DIR / "knowledge_base"
META_FILE   = BASE_DIR / "knowledge_pipeline" / "metadata.json"
LOG_FILE    = BASE_DIR / "knowledge_pipeline" / "download_log.json"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}
TIMEOUT   = 60   # seconds per request
MAX_RETRY = 3


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_log() -> dict:
    if LOG_FILE.exists():
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_log(log: dict):
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)


def download_doc(doc: dict, log: dict) -> dict:
    """Download a single document, returning an updated log entry."""
    doc_id   = doc["id"]
    domain   = doc["domain"]
    filename = doc["filename"]
    dest_dir = KB_DIR / domain
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / filename

    # Skip if already downloaded and hash matches
    if dest_path.exists() and doc_id in log:
        existing_hash = sha256(dest_path)
        if existing_hash == log[doc_id].get("sha256"):
            print(f"  [SKIP]  {filename}  (already downloaded, hash verified)")
            return log[doc_id]

    urls_to_try = [doc.get("url"), doc.get("alt_url")]
    urls_to_try = [u for u in urls_to_try if u]

    for attempt, url in enumerate(urls_to_try):
        for retry in range(MAX_RETRY):
            try:
                print(f"  [DL]    {filename}  <- {url[:70]}...")
                resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT, stream=True)
                resp.raise_for_status()

                with open(dest_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=65536):
                        f.write(chunk)

                file_size = dest_path.stat().st_size
                file_hash = sha256(dest_path)

                entry = {
                    "id":           doc_id,
                    "domain":       domain,
                    "filename":     filename,
                    "url":          url,
                    "sha256":       file_hash,
                    "size_bytes":   file_size,
                    "downloaded_at": datetime.datetime.utcnow().isoformat(),
                    "status":       "success"
                }
                print(f"  [OK]    {filename}  ({file_size // 1024} KB)")
                return entry

            except requests.HTTPError as e:
                print(f"  [WARN]  HTTP {e.response.status_code} for {url}")
                break   # try alt_url
            except Exception as e:
                wait = 2 ** retry
                print(f"  [RETRY] {e} — retrying in {wait}s")
                time.sleep(wait)

    # All attempts failed
    print(f"  [FAIL]  Could not download {filename}")
    return {
        "id":           doc_id,
        "domain":       domain,
        "filename":     filename,
        "status":       "failed",
        "downloaded_at": datetime.datetime.utcnow().isoformat()
    }


def run(priority_filter: str | None = None):
    """
    Run the download pipeline.
    priority_filter: 'High' | 'Medium' | 'Low' | None (all)
    """
    with open(META_FILE, "r", encoding="utf-8") as f:
        meta = json.load(f)

    docs = meta["documents"]
    if priority_filter:
        docs = [d for d in docs if d.get("priority") == priority_filter]

    log = load_log()
    print(f"\nConstruction Intelligence Knowledge Base Downloader")
    print(f"Documents queued : {len(docs)}")
    print(f"Priority filter  : {priority_filter or 'All'}")
    print("-" * 60)

    success, skip, fail = 0, 0, 0
    for doc in docs:
        entry = download_doc(doc, log)
        log[doc["id"]] = entry
        if entry["status"] == "success":
            success += 1
        elif entry.get("sha256"):
            skip += 1
        else:
            fail += 1
        save_log(log)   # incremental save after each file

    print("\n" + "=" * 60)
    print(f"Download complete — Success: {success}  Skipped: {skip}  Failed: {fail}")
    print(f"Log saved to: {LOG_FILE}")


if __name__ == "__main__":
    pf = sys.argv[1] if len(sys.argv) > 1 else None
    run(priority_filter=pf)
