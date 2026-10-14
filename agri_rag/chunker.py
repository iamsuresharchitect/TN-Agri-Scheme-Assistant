"""Split documents into overlapping chunks that respect paragraph/sentence breaks."""
from . import config
from .models import Document

SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " "]


def _split_pieces(text: str, size: int, seps: list) -> list:
    if len(text) <= size:
        return [text]
    sep = next((s for s in seps if s in text), None)
    if sep is None:
        return [text[i:i + size] for i in range(0, len(text), size)]
    rest = seps[seps.index(sep) + 1:]
    parts = text.split(sep)
    pieces = []
    for i, part in enumerate(parts):
        piece = part + (sep if i < len(parts) - 1 else "")
        if not piece:
            continue
        if len(piece) > size:
            pieces.extend(_split_pieces(piece, size, rest))
        else:
            pieces.append(piece)
    return pieces


def split_text(text: str, size: int = None, overlap: int = None) -> list:
    size = size or config.CHUNK_SIZE
    overlap = config.CHUNK_OVERLAP if overlap is None else overlap
    text = text.strip()
    if not text:
        return []
    chunks, current = [], ""
    for piece in _split_pieces(text, size, SEPARATORS):
        if current and len(current) + len(piece) > size:
            chunks.append(current.strip())
            tail = current[-overlap:] if overlap else ""
            # start the overlap on a word boundary
            if tail and " " in tail:
                tail = tail[tail.index(" ") + 1:]
            current = tail
        current += piece
    if current.strip():
        chunks.append(current.strip())
    return [c for c in chunks if len(c) >= 20]


def chunk_documents(docs: list) -> list:
    """Documents in -> chunk Documents out (metadata copied, chunk index added).

    Tables are kept whole when they fit in 2x the chunk size, so a table row is
    not separated from its header.
    """
    out = []
    for doc in docs:
        text = doc.text.strip()
        if not text:
            continue
        if doc.metadata.get("content_type") == "table" and len(text) <= config.CHUNK_SIZE * 2:
            parts = [text]
        else:
            parts = split_text(text)
        for i, part in enumerate(parts):
            out.append(Document(part, {**doc.metadata, "chunk": i}))
    return out
