from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import requests

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class OllamaClient:
    host: str = "127.0.0.1"
    port: int = 11434
    embed_model: str = "nomic-embed-text"
    gen_model: str = "llama3.2"

    def _base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def is_available(self) -> bool:
        try:
            response = requests.get(f"{self._base_url()}/api/tags", timeout=2)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def embed(self, text: str) -> list[float]:
        payload = {"model": self.embed_model, "prompt": text}
        try:
            response = requests.post(f"{self._base_url()}/api/embeddings", json=payload, timeout=30)
            if response.status_code != 200:
                logger.warning("Ollama embedding request failed: %s", response.status_code)
                return []
            data = response.json()
            embedding = data.get("embedding", [])
            if isinstance(embedding, list):
                return [float(value) for value in embedding]
            return []
        except (requests.RequestException, ValueError, TypeError) as exc:
            logger.exception("Ollama embedding request failed: %s", exc)
            return []

    def generate(self, prompt: str) -> str:
        payload = {"model": self.gen_model, "prompt": prompt, "stream": False}
        try:
            response = requests.post(f"{self._base_url()}/api/generate", json=payload, timeout=180)
            if response.status_code != 200:
                return "ERROR: Ollama unavailable. Run: ollama serve"
            data = response.json()
            answer = data.get("response", "")
            return answer if isinstance(answer, str) else ""
        except (requests.RequestException, ValueError, TypeError) as exc:
            logger.exception("Ollama generation request failed: %s", exc)
            return "ERROR: Ollama unavailable. Run: ollama serve"
