from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

from ..models.vector_item import VectorItem
from ..utils.metrics import DistanceFn


@dataclass(slots=True)
class BruteForceSearch:
    items: List[VectorItem] = field(default_factory=list)

    def insert(self, item: VectorItem) -> None:
        self.items.append(item)

    def remove(self, item_id: int) -> None:
        self.items = [item for item in self.items if item.id != item_id]

    def knn(self, query: Sequence[float], k: int, dist_fn: DistanceFn) -> list[tuple[float, int]]:
        ranked = [(dist_fn(query, item.embedding), item.id) for item in self.items]
        ranked.sort(key=lambda pair: (pair[0], pair[1]))
        return ranked[:k]
