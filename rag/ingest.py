#!/usr/bin/env python3
"""
AI'dvisor McGill Advisor — RAG Ingestion Pipeline
Chunks scraped Markdown, embeds, and upserts into a persistent ChromaDB collection.

Usage:
    python rag/ingest.py                    # ingest everything in scraper/scraped_data/
    EMBED_BACKEND=local python rag/ingest.py  # use free SentenceTransformers embeddings

Exported API (used by router.py):
    query_vectorstore(question, n_results=6) -> list[dict]
"""
import hashlib
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Optional

# Allow running as a script from any directory
sys.path.insert(0, str(Path(__file__).parent.parent))

import chromadb
from chromadb.config import Settings

SCRAPED_DATA_DIR = Path(__file__).parent.parent / "scraper" / "scraped_data"
CHROMA_PERSIST_DIR = Path(__file__).parent.parent / "chroma_data"
COLLECTION_NAME = "mcgill_advisor"

CHUNK_SIZE_TOKENS = 600
CHUNK_OVERLAP_TOKENS = 80
CHARS_PER_TOKEN = 4          # rough English approximation
EMBED_BATCH_SIZE = 96        # stay inside OpenAI's 2 048 item limit

# Set EMBED_BACKEND=local to use SentenceTransformers instead of OpenAI
USE_OPENAI = os.getenv("EMBED_BACKEND", "openai").lower() == "openai"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ── lazy singletons ───────────────────────────────────────────────────────────

_openai_client = None
_st_model = None
_chroma_collection = None


def _get_openai():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        _openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _openai_client


def _get_st_model():
    global _st_model
    if _st_model is None:
        from sentence_transformers import SentenceTransformer
        log.info("Loading SentenceTransformer model all-MiniLM-L6-v2…")
        _st_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _st_model


# ── embedding ─────────────────────────────────────────────────────────────────

def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of strings. Dispatches to OpenAI or SentenceTransformers."""
    if not texts:
        return []
    if USE_OPENAI:
        client = _get_openai()
        resp = client.embeddings.create(
            model="text-embedding-3-small",
            input=texts,
        )
        return [item.embedding for item in resp.data]
    else:
        model = _get_st_model()
        embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return [e.tolist() for e in embeddings]


# ── chunking ──────────────────────────────────────────────────────────────────

def _last_section_header(text: str, before_pos: int) -> str:
    """Return the text of the most recent ATX heading before *before_pos*."""
    pattern = re.compile(r"^#{1,4}\s+(.+)$", re.MULTILINE)
    last = ""
    for m in pattern.finditer(text, 0, before_pos):
        last = m.group(1).strip()
    return last


def chunk_markdown(text: str, source_url: str) -> list[dict]:
    """
    Split *text* into overlapping windows of ~CHUNK_SIZE_TOKENS tokens.

    Each returned dict has:
        id          — stable MD5 of (url, start_char)
        text        — chunk content prefixed with its nearest section heading
        source_url  — origin URL
        start_char  — byte offset (for debugging)
    """
    target_chars = CHUNK_SIZE_TOKENS * CHARS_PER_TOKEN
    overlap_chars = CHUNK_OVERLAP_TOKENS * CHARS_PER_TOKEN
    text_len = len(text)
    chunks: list[dict] = []
    start = 0

    while start < text_len:
        end = min(start + target_chars, text_len)

        # Prefer breaking at paragraph boundaries
        if end < text_len:
            pb = text.rfind("\n\n", start, end)
            if pb != -1 and pb > start + overlap_chars:
                end = pb + 2

        chunk_text = text[start:end].strip()
        if chunk_text:
            section = _last_section_header(text, start)
            prefix = f"[Section: {section}]\n\n" if section else ""
            full_text = prefix + chunk_text

            chunk_id = hashlib.md5(f"{source_url}::{start}".encode()).hexdigest()
            chunks.append(
                {
                    "id": chunk_id,
                    "text": full_text,
                    "source_url": source_url,
                    "start_char": start,
                }
            )

        next_start = end - overlap_chars
        if next_start <= start:
            break
        start = next_start

    return chunks


# ── ChromaDB ──────────────────────────────────────────────────────────────────

def get_collection() -> chromadb.Collection:
    global _chroma_collection
    if _chroma_collection is None:
        CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(
            path=str(CHROMA_PERSIST_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
        _chroma_collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _chroma_collection


def query_vectorstore(question: str, n_results: int = 6) -> list[dict]:
    """
    Semantic search over the ChromaDB collection.

    Returns a list of dicts:
        [{"text": str, "source_url": str, "distance": float}, …]
    """
    collection = get_collection()
    embeddings = embed_texts([question])
    results = collection.query(
        query_embeddings=embeddings,
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    return [
        {
            "text": doc,
            "source_url": meta.get("source_url", ""),
            "distance": dist,
        }
        for doc, meta, dist in zip(docs, metas, dists)
    ]


# ── ingestion ─────────────────────────────────────────────────────────────────

def ingest_file(filepath: Path, source_url: str) -> int:
    """Chunk, embed, and upsert one Markdown file. Returns chunk count."""
    text = filepath.read_text(encoding="utf-8")
    if len(text.strip()) < 50:
        return 0

    chunks = chunk_markdown(text, source_url)
    if not chunks:
        return 0

    collection = get_collection()
    total = 0

    for i in range(0, len(chunks), EMBED_BATCH_SIZE):
        batch = chunks[i : i + EMBED_BATCH_SIZE]
        texts = [c["text"] for c in batch]
        embeddings = embed_texts(texts)

        collection.upsert(
            ids=[c["id"] for c in batch],
            embeddings=embeddings,
            documents=texts,
            metadatas=[{"source_url": c["source_url"]} for c in batch],
        )
        total += len(batch)

    return total


def run_ingestion(data_dir: Path = SCRAPED_DATA_DIR) -> None:
    manifest_path = data_dir / "_index.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found at {manifest_path}. Run the scraper first:\n"
            "  python scraper/scraper.py --quick"
        )

    index: list[dict] = json.loads(manifest_path.read_text(encoding="utf-8"))
    log.info(
        "Ingesting %d documents (embed backend: %s)",
        len(index), "openai" if USE_OPENAI else "sentence-transformers",
    )

    total_chunks = 0
    for entry in index:
        filepath = data_dir / entry["filename"]
        if not filepath.exists():
            log.warning("Missing file: %s — skipping", filepath.name)
            continue
        n = ingest_file(filepath, entry["url"])
        log.info("  %-70s  %d chunks", entry["filename"][:70], n)
        total_chunks += n

    log.info("Ingestion complete. Total chunks upserted: %d", total_chunks)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
    run_ingestion()
