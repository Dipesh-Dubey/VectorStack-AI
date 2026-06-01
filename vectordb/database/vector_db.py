from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from time import perf_counter_ns
from typing import Sequence

from ..algorithms.bruteforce import BruteForceSearch
from ..algorithms.hnsw import HNSW
from ..algorithms.kdtree import KDTree
from ..config import settings
from ..models.vector_item import VectorItem
from ..utils.metrics import DistanceFn, get_distance_fn

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class SearchHit:
    id: int
    metadata: str
    category: str
    embedding: list[float]
    distance: float


@dataclass(slots=True)
class SearchResult:
    results: list[SearchHit]
    latency_us: int
    algo: str
    metric: str


@dataclass(slots=True)
class BenchmarkResult:
    bruteforceUs: int
    kdtreeUs: int
    hnswUs: int
    itemCount: int


class VectorDB:
    def __init__(self, dims: int) -> None:
        self.dims = dims
        self._store: dict[int, VectorItem] = {}
        self._bruteforce = BruteForceSearch()
        self._kdtree = KDTree(dims)
        self._hnsw = HNSW(settings.hnsw_m, settings.hnsw_ef_construction)
        self._lock = threading.Lock()
        self._next_id = 1

    def insert(self, metadata: str, category: str, embedding: Sequence[float], dist_fn: DistanceFn) -> int:
        with self._lock:
            item = VectorItem(
                id=self._next_id,
                metadata=metadata,
                category=category,
                embedding=[float(value) for value in embedding],
            )
            self._next_id += 1
            self._store[item.id] = item
            self._bruteforce.insert(item)
            self._kdtree.insert(item)
            self._hnsw.insert(item, dist_fn)
            return item.id

    def remove(self, item_id: int) -> bool:
        with self._lock:
            if item_id not in self._store:
                return False
            del self._store[item_id]
            self._bruteforce.remove(item_id)
            self._hnsw.remove(item_id)
            self._kdtree.rebuild(list(self._store.values()))
            return True

    def search(self, query: Sequence[float], k: int, metric: str, algo: str) -> SearchResult:
        with self._lock:
            dist_fn = get_distance_fn(metric)
            start = perf_counter_ns()
            normalized_algo = (algo or "hnsw").lower()
            if normalized_algo == "bruteforce":
                ranked = self._bruteforce.knn(query, k, dist_fn)
            elif normalized_algo == "kdtree":
                ranked = self._kdtree.knn(query, k, dist_fn)
            else:
                ranked = self._hnsw.knn(query, k, settings.hnsw_ef_search, dist_fn)
            latency_us = int((perf_counter_ns() - start) / 1000)

            results: list[SearchHit] = []
            for distance, item_id in ranked:
                item = self._store.get(item_id)
                if item is None:
                    continue
                results.append(
                    SearchHit(
                        id=item.id,
                        metadata=item.metadata,
                        category=item.category,
                        embedding=list(item.embedding),
                        distance=float(distance),
                    )
                )
            return SearchResult(results=results, latency_us=latency_us, algo=normalized_algo, metric=(metric or "cosine").lower())

    def benchmark(self, query: Sequence[float], k: int, metric: str) -> BenchmarkResult:
        with self._lock:
            dist_fn = get_distance_fn(metric)

            def elapsed_us(fn) -> int:
                start = perf_counter_ns()
                fn()
                return int((perf_counter_ns() - start) / 1000)

            bruteforce_us = elapsed_us(lambda: self._bruteforce.knn(query, k, dist_fn))
            kdtree_us = elapsed_us(lambda: self._kdtree.knn(query, k, dist_fn))
            hnsw_us = elapsed_us(lambda: self._hnsw.knn(query, k, settings.hnsw_ef_search, dist_fn))
            return BenchmarkResult(bruteforce_us, kdtree_us, hnsw_us, len(self._store))

    def all_items(self) -> list[VectorItem]:
        with self._lock:
            return list(self._store.values())

    def hnsw_info(self):
        with self._lock:
            return self._hnsw.get_info()

    def size(self) -> int:
        with self._lock:
            return len(self._store)
