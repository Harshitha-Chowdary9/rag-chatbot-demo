"""Load documents and split them into overlapping word chunks."""
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Chunk:
    source: str
    index: int
    text: str


def chunk_text(text: str, size: int = 120, overlap: int = 30) -> list[str]:
    if size <= overlap:
        raise ValueError("size must be greater than overlap")
    words = text.split()
    if not words:
        return []
    step = size - overlap
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
    return chunks


def load_chunks(folder: str | Path, size: int = 120, overlap: int = 30) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(Path(folder).glob("*.txt")):
        for i, piece in enumerate(chunk_text(path.read_text(encoding="utf-8"), size, overlap)):
            chunks.append(Chunk(path.name, i, piece))
    return chunks
