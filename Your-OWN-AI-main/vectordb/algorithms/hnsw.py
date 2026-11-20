from __future__ import annotations

from dataclasses import dataclass, field
import heapq
import logging
import math
import random
from typing import Dict, Iterable, List, Sequence

from ..models.vector_item import VectorItem
from ..utils.metrics import DistanceFn

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class HNSWNode:
    item: VectorItem
    max_layer: int
    neighbors: list[list[int]] = field(default_factory=list)


@dataclass(slots=True)
class HNSWNodeInfo:
    id: int
    metadata: str
    category: str
    maxLyr: int


@dataclass(slots=True)
class HNSWEdgeInfo:
    src: int
    dst: int
    lyr: int


@dataclass(slots=True)
class HNSWGraphInfo:
    topLayer: int
    nodeCount: int
    nodesPerLayer: list[int]
    edgesPerLayer: list[int]
    nodes: list[HNSWNodeInfo]
    edges: list[HNSWEdgeInfo]


class HNSW:
    def __init__(self, m: int = 16, ef_construction: int = 200, seed: int = 42) -> None:
        self._m = m
        self._m0 = 2 * m
        self._ef_construction = ef_construction
        self._m_l = 1.0 / math.log(float(m))
        self._graph: Dict[int, HNSWNode] = {}
        self._entry_point = -1
        self._top_layer = -1
        self._rng = random.Random(seed)

    def _rand_level(self) -> int:
        value = self._rng.random()
        value = max(value, 1e-12)
        return int(math.floor(-math.log(value) * self._m_l))

    def _search_layer(
        self,
        query: Sequence[float],
        entry_id: int,
        ef: int,
        layer: int,
        dist_fn: DistanceFn,
    ) -> list[tuple[float, int]]:
        if entry_id not in self._graph:
            return []

        visited = {entry_id}
        candidates: list[tuple[float, int]] = []
        found: list[tuple[float, int]] = []

        entry_distance = dist_fn(query, self._graph[entry_id].item.embedding)
        heapq.heappush(candidates, (entry_distance, entry_id))
        heapq.heappush(found, (-entry_distance, entry_id))

        while candidates:
            current_distance, current_id = heapq.heappop(candidates)
            worst_found = -found[0][0]
            if len(found) >= ef and current_distance > worst_found:
                break

            node = self._graph.get(current_id)
            if node is None or layer >= len(node.neighbors):
                continue

            for neighbor_id in node.neighbors[layer]:
                if neighbor_id in visited or neighbor_id not in self._graph:
                    continue
                visited.add(neighbor_id)
                neighbor_distance = dist_fn(query, self._graph[neighbor_id].item.embedding)
                if len(found) < ef or neighbor_distance < -found[0][0]:
                    heapq.heappush(candidates, (neighbor_distance, neighbor_id))
                    heapq.heappush(found, (-neighbor_distance, neighbor_id))
                    if len(found) > ef:
                        heapq.heappop(found)

        ranked = [(-distance, item_id) for distance, item_id in found]
        ranked.sort(key=lambda pair: (pair[0], pair[1]))
        return ranked

    def _select_neighbors(self, candidates: Sequence[tuple[float, int]], max_m: int) -> list[int]:
        return [item_id for _, item_id in list(candidates)[:max_m]]

    def insert(self, item: VectorItem, dist_fn: DistanceFn) -> None:
        node_id = item.id
        layer = self._rand_level()
        self._graph[node_id] = HNSWNode(item=item, max_layer=layer, neighbors=[[] for _ in range(layer + 1)])

        if self._entry_point == -1:
            self._entry_point = node_id
            self._top_layer = layer
            return

        entry_id = self._entry_point
        for current_layer in range(self._top_layer, layer, -1):
            entry_node = self._graph.get(entry_id)
            if entry_node is None or current_layer >= len(entry_node.neighbors):
                continue
            candidates = self._search_layer(item.embedding, entry_id, 1, current_layer, dist_fn)
            if candidates:
                entry_id = candidates[0][1]

        for current_layer in range(min(self._top_layer, layer), -1, -1):
            candidates = self._search_layer(item.embedding, entry_id, self._ef_construction, current_layer, dist_fn)
            max_m = self._m0 if current_layer == 0 else self._m
            selected = self._select_neighbors(candidates, max_m)
            self._graph[node_id].neighbors[current_layer] = selected

            for neighbor_id in selected:
                neighbor = self._graph.get(neighbor_id)
                if neighbor is None:
                    continue
                while len(neighbor.neighbors) <= current_layer:
                    neighbor.neighbors.append([])
                connections = neighbor.neighbors[current_layer]
                connections.append(node_id)
                if len(connections) > max_m:
                    ranked = []
                    for connection_id in connections:
                        connected = self._graph.get(connection_id)
                        if connected is None:
                            continue
                        ranked.append((dist_fn(neighbor.item.embedding, connected.item.embedding), connection_id))
                    ranked.sort(key=lambda pair: (pair[0], pair[1]))
                    neighbor.neighbors[current_layer] = [connection_id for _, connection_id in ranked[:max_m]]

            if candidates:
                entry_id = candidates[0][1]

        if layer > self._top_layer:
            self._top_layer = layer
            self._entry_point = node_id

    def knn(self, query: Sequence[float], k: int, ef: int, dist_fn: DistanceFn) -> list[tuple[float, int]]:
        if self._entry_point == -1 or k <= 0:
            return []

        entry_id = self._entry_point
        for current_layer in range(self._top_layer, 0, -1):
            entry_node = self._graph.get(entry_id)
            if entry_node is None or current_layer >= len(entry_node.neighbors):
                continue
            candidates = self._search_layer(query, entry_id, 1, current_layer, dist_fn)
            if candidates:
                entry_id = candidates[0][1]

        results = self._search_layer(query, entry_id, max(ef, k), 0, dist_fn)
        return results[:k]

    def remove(self, item_id: int) -> None:
        if item_id not in self._graph:
            return

        for neighbor_id, node in self._graph.items():
            for layer_connections in node.neighbors:
                while item_id in layer_connections:
                    layer_connections.remove(item_id)

        del self._graph[item_id]

        if not self._graph:
            self._entry_point = -1
            self._top_layer = -1
            return

        self._entry_point = max(self._graph, key=lambda node_id: self._graph[node_id].max_layer)
        self._top_layer = self._graph[self._entry_point].max_layer

    def get_info(self) -> HNSWGraphInfo:
        max_layers = max(self._top_layer + 1, 1)
        nodes_per_layer = [0 for _ in range(max_layers)]
        edges_per_layer = [0 for _ in range(max_layers)]
        nodes: list[HNSWNodeInfo] = []
        edges: list[HNSWEdgeInfo] = []

        for node_id, node in self._graph.items():
            nodes.append(HNSWNodeInfo(id=node_id, metadata=node.item.metadata, category=node.item.category, maxLyr=node.max_layer))
            for layer in range(0, min(node.max_layer, max_layers - 1) + 1):
                nodes_per_layer[layer] += 1
                if layer < len(node.neighbors):
                    for neighbor_id in node.neighbors[layer]:
                        if node_id < neighbor_id:
                            edges_per_layer[layer] += 1
                            edges.append(HNSWEdgeInfo(src=node_id, dst=neighbor_id, lyr=layer))

        return HNSWGraphInfo(
            topLayer=self._top_layer,
            nodeCount=len(self._graph),
            nodesPerLayer=nodes_per_layer,
            edgesPerLayer=edges_per_layer,
            nodes=nodes,
            edges=edges,
        )

    def size(self) -> int:
        return len(self._graph)
