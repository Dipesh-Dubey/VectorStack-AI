from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Settings:
    demo_dims: int = 16
    hnsw_m: int = 16
    hnsw_ef_construction: int = 200
    hnsw_ef_search: int = 50
    doc_chunk_words: int = 250
    doc_chunk_overlap: int = 30
    doc_search_max_dist: float = 0.7
    ollama_host: str = "127.0.0.1"
    ollama_port: int = 11434
    http_host: str = "0.0.0.0"
    http_port: int = 8080

    @property
    def root(self) -> Path:
        return Path(__file__).resolve().parents[1]


settings = Settings()
