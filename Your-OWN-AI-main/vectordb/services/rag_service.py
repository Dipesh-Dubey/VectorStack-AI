from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..config import settings
from ..database.document_db import DocumentDB
from ..services.ollama_client import OllamaClient


@dataclass(slots=True)
class RagService:
    document_db: DocumentDB
    ollama: OllamaClient

    def search(self, question: str, k: int) -> dict[str, Any]:
        question = question.strip()
        if not question:
            return {"error": "need question"}

        question_embedding = self.ollama.embed(question)
        if not question_embedding:
            return {"error": "Ollama unavailable"}

        hits = self.document_db.search(question_embedding, k)
        contexts = [
            {"id": doc.id, "title": doc.title, "distance": round(distance, 4)}
            for distance, doc in hits
        ]
        return {"contexts": contexts}

    def ask(self, question: str, k: int) -> dict[str, Any]:
        question = question.strip()
        if not question:
            return {"error": "need question"}

        question_embedding = self.ollama.embed(question)
        if not question_embedding:
            return {"error": "Ollama unavailable"}

        hits = self.document_db.search(question_embedding, k)
        context_blocks = []
        for index, (_, doc) in enumerate(hits, start=1):
            context_blocks.append(f"[{index}] {doc.title}:\n{doc.text}\n")

        prompt = (
            "You are a helpful assistant. Answer the user's question directly. "
            "Use the provided context if it contains relevant information. "
            "If it doesn't, just use your own general knowledge. "
            "IMPORTANT: Do NOT mention the 'context', 'provided text', or say things like 'the context doesn't mention'. "
            "Just answer the question naturally.\n\n"
            "Context:\n"
            + "\n".join(context_blocks)
            + f"Question: {question}\n\nAnswer:"
        )

        answer = self.ollama.generate(prompt)
        return {
            "answer": answer,
            "model": self.ollama.gen_model,
            "contexts": [
                {"id": doc.id, "title": doc.title, "text": doc.text, "distance": round(distance, 4)}
                for distance, doc in hits
            ],
            "docCount": self.document_db.size(),
        }
