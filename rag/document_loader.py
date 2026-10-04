"""Extract and chunk project report documents for RAG ingestion.

Globs recursively for .docx and .pdf files under the docs/ directory.
Returns a flat list of chunk dicts: {text, source_filename, chunk_index}.
"""

import logging
from pathlib import Path
from typing import TypedDict

import docx
import pypdf

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DOCS_DIR = _PROJECT_ROOT / "docs"

_CHUNK_WORDS = 500
_OVERLAP_WORDS = 50
_STEP = _CHUNK_WORDS - _OVERLAP_WORDS  # 450


class Chunk(TypedDict):
    text: str
    source_filename: str
    chunk_index: int


# ---------------------------------------------------------------------------
# Extractors
# ---------------------------------------------------------------------------

def _extract_docx(path: Path) -> str:
    doc = docx.Document(str(path))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragraphs)


def _extract_pdf(path: Path) -> str:
    reader = pypdf.PdfReader(str(path))
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n".join(pages)


def _extract_md(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------

def _chunk_text(text: str, source_filename: str) -> list[Chunk]:
    """Split text into overlapping word-based chunks."""
    words = text.split()
    if not words:
        return []

    chunks: list[Chunk] = []
    idx = 0
    start = 0
    while start < len(words):
        chunk_words = words[start : start + _CHUNK_WORDS]
        chunks.append(Chunk(
            text=" ".join(chunk_words),
            source_filename=source_filename,
            chunk_index=idx,
        ))
        idx += 1
        start += _STEP

    return chunks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_documents(docs_dir: Path = _DOCS_DIR) -> list[Chunk]:
    """Glob for .docx and .pdf files under docs_dir and return chunked text.

    Skips Word temp files (~$...) and files that fail extraction.
    Each returned chunk is a dict with keys: text, source_filename, chunk_index.
    """
    if not docs_dir.exists():
        logger.error("docs_dir does not exist: %s", docs_dir)
        return []

    # Collect files, excluding Word lock files (~$filename.docx)
    paths = sorted(
        p for p in (
            list(docs_dir.rglob("*.docx"))
            + list(docs_dir.rglob("*.pdf"))
            + list(docs_dir.rglob("*.md"))
        )
        if not p.name.startswith("~$")
    )

    if not paths:
        logger.warning("No .docx, .pdf, or .md files found under %s", docs_dir)
        return []

    # Deduplicate: group by stem; prefer .docx over .pdf when both exist.
    by_stem: dict[str, Path] = {}
    for path in paths:
        stem = path.stem
        existing = by_stem.get(stem)
        if existing is None:
            by_stem[stem] = path
        elif path.suffix.lower() == ".docx" and existing.suffix.lower() == ".pdf":
            # Upgrade pdf → docx for this stem
            logger.info(
                "Skipped %s (using .docx version instead)", existing.name
            )
            by_stem[stem] = path
        else:
            # existing is .docx (or same suffix) — skip the new path
            logger.info(
                "Skipped %s (using .docx version instead)", path.name
            )

    paths = sorted(by_stem.values())

    logger.info("Found %d unique document(s) to load:", len(paths))
    for p in paths:
        logger.info("  %s", p.relative_to(_PROJECT_ROOT))

    all_chunks: list[Chunk] = []
    for path in paths:
        try:
            if path.suffix.lower() == ".docx":
                text = _extract_docx(path)
            elif path.suffix.lower() == ".md":
                text = _extract_md(path)
            else:
                text = _extract_pdf(path)
        except Exception:
            logger.exception("Failed to extract text from %s — skipping", path.name)
            continue

        if not text.strip():
            logger.warning("No extractable text in %s — skipping", path.name)
            continue

        chunks = _chunk_text(text, path.name)
        logger.info("  %s → %d chunk(s)", path.name, len(chunks))
        all_chunks.extend(chunks)

    logger.info("Total: %d chunks from %d file(s)", len(all_chunks), len(paths))
    return all_chunks
