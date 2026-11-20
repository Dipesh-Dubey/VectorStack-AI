from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

DistanceFn = Callable[[Sequence[float], Sequence[float]], float]


def _as_array(values: Sequence[float]) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def euclidean(a: Sequence[float], b: Sequence[float]) -> float:
    left = _as_array(a)
    right = _as_array(b)
    diff = left - right
    return float(np.sqrt(np.dot(diff, diff)))


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    left = _as_array(a)
    right = _as_array(b)
    dot = float(np.dot(left, right))
    norm_left = float(np.dot(left, left))
    norm_right = float(np.dot(right, right))
    if norm_left < 1e-9 or norm_right < 1e-9:
        return 1.0
    return float(1.0 - dot / (np.sqrt(norm_left) * np.sqrt(norm_right)))


def manhattan(a: Sequence[float], b: Sequence[float]) -> float:
    left = _as_array(a)
    right = _as_array(b)
    return float(np.abs(left - right).sum())


def get_distance_fn(metric: str) -> DistanceFn:
    normalized = (metric or "euclidean").lower()
    if normalized == "cosine":
        return cosine
    if normalized == "manhattan":
        return manhattan
    return euclidean
