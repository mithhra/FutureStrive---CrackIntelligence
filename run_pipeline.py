"""
run_pipeline.py
----------------
Single-shot script that runs the full RAG pipeline:
  1. Seed knowledge base (if not already done)
  2. Process all domain text files into chunks
  3. Build FAISS vector index
  4. Run gap analysis
  5. Print summary

Run this once to bootstrap the knowledge base.
Subsequent runs auto-detect existing index and only process new files.
"""

import sys
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# ── Step 1: Ensure knowledge base is seeded ──────────────────────────────────
print("\n[1/4] Checking knowledge base seed files...")
kb_dir = BASE_DIR / "knowledge_base"
total_txt = sum(1 for p in kb_dir.rglob("*.txt"))
if total_txt < 10:
    print("  Seeding knowledge base from authoritative domain content...")
    import knowledge_pipeline.seed_knowledge_base as seeder
    seeder.seed()
else:
    print(f"  Found {total_txt} knowledge text files. Seed already complete.")

# ── Step 2: Process all documents into chunks ────────────────────────────────
print("\n[2/4] Processing documents into semantic chunks...")
import knowledge_pipeline.processor as processor
chunk_count = processor.process_all()

if chunk_count == 0:
    print("ERROR: No chunks produced. Check knowledge_base/ folder.")
    sys.exit(1)

# ── Step 3: Build FAISS vector index ─────────────────────────────────────────
print("\n[3/4] Building FAISS vector index...")
import knowledge_pipeline.embedder as embedder
embedder.build_index(update_only=False)

# ── Step 4: Run gap analysis ──────────────────────────────────────────────────
print("\n[4/4] Running knowledge gap analysis...")
import knowledge_pipeline.gap_detector as gap_detector
gap_detector.run()

print("\n" + "=" * 60)
print("Pipeline complete. Launch the app with:")
print("  streamlit run app.py --server.port 8501")
print("=" * 60)
