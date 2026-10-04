"""ChromaDB vector store for project report chunks.

build_index() — embed all 31 report chunks and persist to ./chroma_db.
query()       — embed a question, return top_k matching chunks with scores.
"""

import logging
import os
import sys
from pathlib import Path

import chromadb
from dotenv import load_dotenv

from rag.document_loader import load_documents

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CHROMA_PATH = _PROJECT_ROOT / "chroma_db"
_COLLECTION_NAME = "project_docs"
_EMBEDDING_MODEL = "models/gemini-embedding-001"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_embeddings():
    """Load the API key and return a GoogleGenerativeAIEmbeddings instance.

    Raises EnvironmentError with a plain-English message if the key is absent
    or rejected by the API — never a raw stack trace.
    """
    load_dotenv(_PROJECT_ROOT / ".env")
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or api_key == "your-key-here":
        raise OSError(
            "GOOGLE_API_KEY is not set.\n"
            "Add your key to the .env file at the project root:\n"
            "  GOOGLE_API_KEY=AIza..."
        )
    try:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(
            model=_EMBEDDING_MODEL,
            google_api_key=api_key,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to initialise GoogleGenerativeAIEmbeddings: {exc}"
        ) from exc


def _open_collection(rebuild: bool = False) -> chromadb.Collection:
    client = chromadb.PersistentClient(path=str(_CHROMA_PATH))
    if rebuild:
        try:
            client.delete_collection(_COLLECTION_NAME)
            logger.info("Deleted existing collection for rebuild")
        except Exception:
            pass  # collection didn't exist yet — fine
    return client.get_or_create_collection(
        name=_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def _wrap_api_error(exc: Exception) -> Exception:
    msg = str(exc)
    if any(kw in msg.upper() for kw in ("API_KEY", "PERMISSION", "UNAUTHENTICATED", "401", "403")):
        return OSError(
            f"Google API key is missing or invalid. "
            f"Check GOOGLE_API_KEY in .env.\nDetail: {exc}"
        )
    return exc


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_index(rebuild: bool = False) -> None:
    """Embed all report chunks and store in ChromaDB.

    Skips the rebuild if the collection already contains data, unless
    rebuild=True is passed (or --rebuild on the CLI).
    """
    collection = _open_collection(rebuild=rebuild)

    existing = collection.count()
    if existing > 0 and not rebuild:
        logger.info(
            "Index already contains %d chunks — skipping rebuild "
            "(pass rebuild=True or --rebuild to force).",
            existing,
        )
        print(f"Index already built: {existing} chunks in {_CHROMA_PATH}")
        return

    chunks = load_documents()
    if not chunks:
        logger.error("No chunks returned from load_documents() — aborting")
        return

    logger.info("Embedding %d chunks with %s …", len(chunks), _EMBEDDING_MODEL)
    embeddings_model = _load_embeddings()

    try:
        vectors = embeddings_model.embed_documents([c["text"] for c in chunks])
    except Exception as exc:
        raise _wrap_api_error(exc) from exc

    ids = [f"{c['source_filename']}::{c['chunk_index']}" for c in chunks]
    metadatas = [
        {"source_filename": c["source_filename"], "chunk_index": c["chunk_index"]}
        for c in chunks
    ]

    collection.add(
        ids=ids,
        documents=[c["text"] for c in chunks],
        embeddings=vectors,
        metadatas=metadatas,
    )

    count = collection.count()
    logger.info("ChromaDB now contains %d chunks at %s", count, _CHROMA_PATH)
    print(f"Indexed {count} chunks -> {_CHROMA_PATH}")


def query(question: str, top_k: int = 3) -> list[dict]:
    """Embed question and return top_k matching chunks with metadata and scores.

    Each result dict has: text, source_filename, chunk_index, distance.
    Lower cosine distance = more similar.
    """
    collection = _open_collection()

    if collection.count() == 0:
        raise RuntimeError(
            "Vector index is empty. Run build_index() first."
        )

    embeddings_model = _load_embeddings()

    try:
        q_vector = embeddings_model.embed_query(question)
    except Exception as exc:
        raise _wrap_api_error(exc) from exc

    results = collection.query(
        query_embeddings=[q_vector],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
        strict=True,
    ):
        hits.append({
            "text": doc,
            "source_filename": meta["source_filename"],
            "chunk_index": int(meta["chunk_index"]),
            "distance": round(dist, 4),
        })
    return hits


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    build_index(rebuild="--rebuild" in sys.argv)
