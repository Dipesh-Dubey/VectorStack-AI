from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Sequence

from ..algorithms.bruteforce import BruteForceSearch
from ..algorithms.hnsw import HNSW
from ..config import settings
from ..models.doc_item import DocItem
from ..models.vector_item import VectorItem
from ..utils.metrics import cosine


@dataclass(slots=True)
class DocumentSearchHit:
    distance: float
    document: DocItem


class DocumentDB:
    def __init__(self) -> None:
        self._store: dict[int, DocItem] = {}
        self._hnsw = HNSW(settings.hnsw_m, settings.hnsw_ef_construction)
        self._bruteforce = BruteForceSearch()
        self._lock = threading.Lock()
        self._next_id = 1
        self._dims = 0

    def insert(self, title: str, text: str, embedding: Sequence[float]) -> int:
        with self._lock:
            if self._dims == 0:
                self._dims = len(embedding)
            item = DocItem(
                id=self._next_id,
                title=title,
                text=text,
                embedding=[float(value) for value in embedding],
            )
            self._next_id += 1
            self._store[item.id] = item
            vector = VectorItem(id=item.id, metadata=item.title, category="doc", embedding=list(item.embedding))
            self._hnsw.insert(vector, cosine)
            self._bruteforce.insert(vector)
            return item.id

    def search(self, query: Sequence[float], k: int, max_dist: float = settings.doc_search_max_dist) -> list[tuple[float, DocItem]]:
        with self._lock:
            if not self._store:
                return []
            ranked = self._bruteforce.knn(query, k, cosine) if len(self._store) < 10 else self._hnsw.knn(query, k, settings.hnsw_ef_search, cosine)
            results: list[tuple[float, DocItem]] = []
            for distance, item_id in ranked:
                document = self._store.get(item_id)
                if document is None or distance > max_dist:
                    continue
                results.append((distance, document))
            return results

    def remove(self, item_id: int) -> bool:
        with self._lock:
            if item_id not in self._store:
                return False
            del self._store[item_id]
            self._hnsw.remove(item_id)
            self._bruteforce.remove(item_id)
            return True

    def all_documents(self) -> list[DocItem]:
        with self._lock:
            return list(self._store.values())

    def size(self) -> int:
        with self._lock:
            return len(self._store)

    def get_dims(self) -> int:
        with self._lock:
            return self._dims
