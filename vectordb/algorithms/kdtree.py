from __future__ import annotations

from dataclasses import dataclass
import heapq
from typing import List, Optional, Sequence

from ..models.vector_item import VectorItem
from ..utils.metrics import DistanceFn


@dataclass(slots=True)
class KDNode:
    item: VectorItem
    left: Optional["KDNode"] = None
    right: Optional["KDNode"] = None


class KDTree:
    def __init__(self, dims: int) -> None:
        self._dims = dims
        self._root: Optional[KDNode] = None

    def _insert(self, node: Optional[KDNode], item: VectorItem, depth: int) -> KDNode:
        if node is None:
            return KDNode(item)
        axis = depth % self._dims
        if item.embedding[axis] < node.item.embedding[axis]:
            node.left = self._insert(node.left, item, depth + 1)
        else:
            node.right = self._insert(node.right, item, depth + 1)
        return node

    def insert(self, item: VectorItem) -> None:
        self._root = self._insert(self._root, item, 0)

    def _knn(
        self,
        node: Optional[KDNode],
        query: Sequence[float],
        k: int,
        depth: int,
        dist_fn: DistanceFn,
        heap: list[tuple[float, int]],
    ) -> None:
        if node is None:
            return

        distance = dist_fn(query, node.item.embedding)
        entry = (-distance, node.item.id)
        if len(heap) < k:
            heapq.heappush(heap, entry)
        elif distance < -heap[0][0]:
            heapq.heapreplace(heap, entry)

        axis = depth % self._dims
        diff = query[axis] - node.item.embedding[axis]
        closer = node.left if diff < 0 else node.right
        farther = node.right if diff < 0 else node.left

        self._knn(closer, query, k, depth + 1, dist_fn, heap)
        if len(heap) < k or abs(diff) < -heap[0][0]:
            self._knn(farther, query, k, depth + 1, dist_fn, heap)

    def knn(self, query: Sequence[float], k: int, dist_fn: DistanceFn) -> list[tuple[float, int]]:
        if self._root is None or k <= 0:
            return []
        heap: list[tuple[float, int]] = []
        self._knn(self._root, query, k, 0, dist_fn, heap)
        ranked = [(-distance, item_id) for distance, item_id in heap]
        ranked.sort(key=lambda pair: (pair[0], pair[1]))
        return ranked

    def rebuild(self, items: Sequence[VectorItem]) -> None:
        self._root = None
        for item in items:
            self.insert(item)
