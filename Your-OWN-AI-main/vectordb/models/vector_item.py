from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class VectorItem:
    id: int
    metadata: str
    category: str
    embedding: list[float]

    @property
    def emb(self) -> list[float]:
        return self.embedding
