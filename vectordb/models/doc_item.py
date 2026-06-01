from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class DocItem:
    id: int
    title: str
    text: str
    embedding: list[float]

    @property
    def emb(self) -> list[float]:
        return self.embedding
