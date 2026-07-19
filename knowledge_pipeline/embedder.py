"""
knowledge_pipeline/embedder.py
---------------------------------
FAISS vector index builder for the Construction Intelligence Platform.

Steps:
1. Load chunks from knowledge_pipeline/chunks.json
2. Embed each chunk using sentence-transformers (all-MiniLM-L6-v2, offline)
3. Build a FAISS IndexFlatL2 vector index
4. Save the index to vector_store/index.faiss
5. Save chunk metadata to vector_store/chunk_metadata.json
6. Supports incremental update: appends new chunks without rebuilding

Usage:
  python knowledge_pipeline/embedder.py          # full rebuild
  python knowledge_pipeline/embedder.py --update  # append only new chunks
"""

import json
import sys
import os
import numpy as np
from pathlib import Path

try:
    import faiss
except ImportError:
    print("ERROR: faiss-cpu not installed. Run: pip install faiss-cpu")
    sys.exit(1)

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("ERROR: sentence-transformers not installed. Run: pip install sentence-transformers")
    sys.exit(1)

BASE_DIR       = Path(__file__).resolve().parent.parent
CHUNKS_FILE    = BASE_DIR / "knowledge_pipeline" / "chunks.json"
VS_DIR         = BASE_DIR / "vector_store"
INDEX_FILE     = VS_DIR / "index.faiss"
META_FILE      = VS_DIR / "chunk_metadata.json"

EMBED_MODEL    = "all-MiniLM-L6-v2"
BATCH_SIZE     = 64        # chunks per embedding batch
EMBED_DIM      = 384       # dimension for all-MiniLM-L6-v2


def load_chunks() -> list[dict]:
    if not CHUNKS_FILE.exists():
        print(f"ERROR: {CHUNKS_FILE} not found. Run processor.py first.")
        sys.exit(1)
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def load_existing_index() -> tuple:
    """Load existing FAISS index and metadata if present."""
    if INDEX_FILE.exists() and META_FILE.exists():
        index = faiss.read_index(str(INDEX_FILE))
        with open(META_FILE, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return index, meta
    return None, []


def build_index(update_only: bool = False):
    VS_DIR.mkdir(parents=True, exist_ok=True)
    chunks = load_chunks()

    existing_index, existing_meta = load_existing_index()

    if update_only and existing_index is not None:
        existing_ids = {m["chunk_id"] for m in existing_meta}
        new_chunks = [c for c in chunks if c["chunk_id"] not in existing_ids]
        if not new_chunks:
            print("Index is already up to date — no new chunks to add.")
            return
        print(f"\nIncremental update: {len(new_chunks)} new chunk(s) to add.")
        chunks_to_embed = new_chunks
        index = existing_index
        meta_store = existing_meta
    else:
        # Full rebuild
        print(f"\nFull rebuild: {len(chunks)} chunk(s) to embed.")
        chunks_to_embed = chunks
        index = faiss.IndexFlatL2(EMBED_DIM)
        meta_store = []

    print(f"Loading embedding model: {EMBED_MODEL} ...")
    model = SentenceTransformer(EMBED_MODEL)

    texts = [c["text"] for c in chunks_to_embed]
    all_embeddings = []

    print(f"Embedding {len(texts)} chunks in batches of {BATCH_SIZE}...")
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i: i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"  Batch {batch_num}/{total_batches} ({len(batch)} chunks)...")
        embs = model.encode(batch, convert_to_numpy=True, normalize_embeddings=True)
        all_embeddings.append(embs)

    embeddings_np = np.vstack(all_embeddings).astype("float32")
    index.add(embeddings_np)

    # Append metadata (strip heavy 'text' field, keep for citation)
    for i, chunk in enumerate(chunks_to_embed):
        meta_store.append({
            "chunk_id":   chunk["chunk_id"],
            "faiss_idx":  index.ntotal - len(chunks_to_embed) + i,
            "doc_id":     chunk["doc_id"],
            "title":      chunk["title"],
            "source_org": chunk["source_org"],
            "domain":     chunk["domain"],
            "doc_type":   chunk["doc_type"],
            "pub_year":   chunk["pub_year"],
            "source_url": chunk["source_url"],
            "topics":     chunk["topics"],
            "page":       chunk["page"],
            "text":       chunk["text"]   # keep for retrieval context
        })

    # Save
    faiss.write_index(index, str(INDEX_FILE))
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta_store, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print(f"FAISS index built — Total vectors: {index.ntotal}")
    print(f"Index saved to   : {INDEX_FILE}")
    print(f"Metadata saved to: {META_FILE}")
    print(f"Approx. index size: {index.ntotal * EMBED_DIM * 4 / 1024:.1f} KB")


if __name__ == "__main__":
    update_only = "--update" in sys.argv
    build_index(update_only=update_only)
