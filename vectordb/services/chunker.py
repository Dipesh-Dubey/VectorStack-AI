from __future__ import annotations

from typing import List


def chunk_text(text: str, chunk_words: int = 250, overlap_words: int = 30) -> List[str]:
    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_words:
        return [text]

    step = max(chunk_words - overlap_words, 1)
    chunks: List[str] = []
    for start in range(0, len(words), step):
        end = min(start + chunk_words, len(words))
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
    return chunks
