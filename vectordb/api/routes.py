from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Annotated

import requests
from fastapi import APIRouter, Body, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from pydantic import BaseModel, Field

from ..config import settings
from ..data.demo_vectors import load_demo_vectors
from ..database.document_db import DocumentDB
from ..database.vector_db import VectorDB
from ..services.chunker import chunk_text
from ..services.ollama_client import OllamaClient
from ..services.rag_service import RagService
from ..utils.metrics import get_distance_fn

logger = logging.getLogger(__name__)


class InsertVectorRequest(BaseModel):
    metadata: str = Field(default="")
    category: str = Field(default="")
    embedding: list[float]


class SearchQuery(BaseModel):
    v: str
    k: int = 5
    metric: str = "cosine"
    algo: str = "hnsw"


class InsertDocumentRequest(BaseModel):
    title: str
    text: str


class AskDocumentRequest(BaseModel):
    question: str
    k: int = 3


class ProxyEmbeddingRequest(BaseModel):
    model: str | None = None
    prompt: str | None = None
    input: str | None = None


class ProxyGenerateRequest(BaseModel):
    model: str | None = None
    prompt: str | None = None
    stream: bool | None = False


class AppState:
    def __init__(self) -> None:
        self.vector_db = VectorDB(settings.demo_dims)
        self.document_db = DocumentDB()
        self.ollama = OllamaClient(settings.ollama_host, settings.ollama_port)
        self.rag = RagService(self.document_db, self.ollama)


state = AppState()


def parse_vec(raw: str) -> list[float]:
    values: list[float] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            values.append(float(part))
        except ValueError:
            continue
    return values


def escape_json_text(value: str) -> str:
    return json.dumps(value)


def cors_headers(response: Response) -> None:
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"


def create_app() -> FastAPI:
    app = FastAPI(title="VectorDB", version="1.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled exception: %s", exc)
        return JSONResponse(status_code=500, content={"error": "internal server error"})

    @app.options("/{path:path}")
    async def options_handler(path: str) -> Response:
        response = Response(status_code=204)
        cors_headers(response)
        return response

    @app.get("/search")
    async def search(v: str = Query(default=""), k: int = Query(default=5), metric: str = Query(default="cosine"), algo: str = Query(default="hnsw")) -> JSONResponse:
        query = parse_vec(v)
        if len(query) != settings.demo_dims:
            return JSONResponse(content={"error": f"need {settings.demo_dims}D vector"})
        result = state.vector_db.search(query, k, metric, algo)
        payload = {
            "results": [
                {
                    "id": hit.id,
                    "metadata": hit.metadata,
                    "category": hit.category,
                    "distance": round(hit.distance, 6),
                    "embedding": [round(value, 4) for value in hit.embedding],
                }
                for hit in result.results
            ],
            "latencyUs": result.latency_us,
            "algo": result.algo,
            "metric": result.metric,
        }
        return JSONResponse(content=payload)

    @app.post("/insert")
    async def insert_vector(body: InsertVectorRequest) -> JSONResponse:
        if len(body.embedding) != settings.demo_dims:
            return JSONResponse(content={"error": "invalid body"})
        vector_id = state.vector_db.insert(body.metadata, body.category, body.embedding, get_distance_fn("cosine"))
        return JSONResponse(content={"id": vector_id})

    @app.delete("/delete/{item_id}")
    async def delete_vector(item_id: int) -> JSONResponse:
        ok = state.vector_db.remove(item_id)
        return JSONResponse(content={"ok": ok})

    @app.get("/items")
    async def list_items() -> JSONResponse:
        items = state.vector_db.all_items()
        return JSONResponse(
            content=[
                {
                    "id": item.id,
                    "metadata": item.metadata,
                    "category": item.category,
                    "embedding": [round(value, 4) for value in item.embedding],
                }
                for item in items
            ]
        )

    @app.get("/benchmark")
    async def benchmark(v: str = Query(default=""), k: int = Query(default=5), metric: str = Query(default="cosine")) -> JSONResponse:
        query = parse_vec(v)
        if len(query) != settings.demo_dims:
            return JSONResponse(content={"error": f"need {settings.demo_dims}D vector"})
        result = state.vector_db.benchmark(query, k, metric)
        return JSONResponse(
            content={
                "bruteforceUs": result.bruteforceUs,
                "kdtreeUs": result.kdtreeUs,
                "hnswUs": result.hnswUs,
                "itemCount": result.itemCount,
            }
        )

    @app.get("/hnsw-info")
    async def hnsw_info() -> JSONResponse:
        info = state.vector_db.hnsw_info()
        return JSONResponse(
            content={
                "topLayer": info.topLayer,
                "nodeCount": info.nodeCount,
                "nodesPerLayer": info.nodesPerLayer,
                "edgesPerLayer": info.edgesPerLayer,
                "nodes": [
                    {"id": node.id, "metadata": node.metadata, "category": node.category, "maxLyr": node.maxLyr}
                    for node in info.nodes
                ],
                "edges": [{"src": edge.src, "dst": edge.dst, "lyr": edge.lyr} for edge in info.edges],
            }
        )

    @app.post("/doc/insert")
    async def insert_document(body: InsertDocumentRequest) -> JSONResponse:
        title = body.title.strip()
        text = body.text.strip()
        if not title or not text:
            return JSONResponse(content={"error": "need title and text"})

        chunks = chunk_text(text, settings.doc_chunk_words, settings.doc_chunk_overlap)
        ids: list[int] = []
        for index, chunk in enumerate(chunks, start=1):
            embedding = state.ollama.embed(chunk)
            if not embedding:
                return JSONResponse(
                    content={
                        "error": "Ollama unavailable. Install from https://ollama.com then run: ollama pull nomic-embed-text && ollama pull llama3.2",
                    }
                )
            chunk_title = f"{title} [{index}/{len(chunks)}]" if len(chunks) > 1 else title
            ids.append(state.document_db.insert(chunk_title, chunk, embedding))

        return JSONResponse(content={"ids": ids, "chunks": len(chunks), "dims": state.document_db.get_dims()})

    @app.delete("/doc/delete/{item_id}")
    async def delete_document(item_id: int) -> JSONResponse:
        ok = state.document_db.remove(item_id)
        return JSONResponse(content={"ok": ok})

    @app.get("/doc/list")
    async def list_documents() -> JSONResponse:
        documents = state.document_db.all_documents()
        payload = []
        for document in documents:
            preview = document.text[:120]
            if len(document.text) > 120:
                preview += "…"
            payload.append(
                {
                    "id": document.id,
                    "title": document.title,
                    "preview": preview,
                    "words": len(document.text.split()),
                }
            )
        return JSONResponse(content=payload)

    @app.post("/doc/search")
    async def search_documents(body: AskDocumentRequest) -> JSONResponse:
        result = state.rag.search(body.question, body.k)
        if "error" in result:
            return JSONResponse(content=result)
        return JSONResponse(content=result)

    @app.post("/doc/ask")
    async def ask_documents(body: AskDocumentRequest) -> JSONResponse:
        result = state.rag.ask(body.question, body.k)
        return JSONResponse(content=result)

    @app.get("/status")
    async def status() -> JSONResponse:
        return JSONResponse(
            content={
                "ollamaAvailable": state.ollama.is_available(),
                "embedModel": state.ollama.embed_model,
                "genModel": state.ollama.gen_model,
                "docCount": state.document_db.size(),
                "docDims": state.document_db.get_dims(),
                "demoDims": settings.demo_dims,
                "demoCount": state.vector_db.size(),
            }
        )

    @app.get("/stats")
    async def stats() -> JSONResponse:
        return JSONResponse(
            content={
                "count": state.vector_db.size(),
                "dims": settings.demo_dims,
                "algorithms": ["bruteforce", "kdtree", "hnsw"],
                "metrics": ["euclidean", "cosine", "manhattan"],
            }
        )

    @app.get("/")
    async def index() -> Response:
        index_path = Path(__file__).resolve().parents[1] / "index.html"
        if index_path.exists():
            return FileResponse(index_path, media_type="text/html")
        return HTMLResponse("<h1>VectorDB</h1>")

    @app.get("/api/tags")
    async def ollama_tags() -> JSONResponse:
        try:
            response = requests.get(f"{state.ollama._base_url()}/api/tags", timeout=10)
            return JSONResponse(content=response.json(), status_code=response.status_code)
        except requests.RequestException:
            return JSONResponse(content={"error": "ollama unavailable"}, status_code=502)

    @app.post("/api/embeddings")
    async def ollama_embeddings(body: ProxyEmbeddingRequest) -> JSONResponse:
        payload = body.model_dump(exclude_none=True)
        try:
            response = requests.post(f"{state.ollama._base_url()}/api/embeddings", json=payload, timeout=30)
            return JSONResponse(content=response.json(), status_code=response.status_code)
        except requests.RequestException:
            return JSONResponse(content={"error": "ollama unavailable"}, status_code=502)

    @app.post("/api/generate")
    async def ollama_generate(body: ProxyGenerateRequest) -> JSONResponse:
        payload = body.model_dump(exclude_none=True)
        try:
            response = requests.post(f"{state.ollama._base_url()}/api/generate", json=payload, timeout=180)
            return JSONResponse(content=response.json(), status_code=response.status_code)
        except requests.RequestException:
            return JSONResponse(content={"error": "ollama unavailable"}, status_code=502)

    load_demo_vectors(state.vector_db)
    logger.info("VectorDB ready: demo=%s docs=%s", state.vector_db.size(), state.document_db.size())
    return app


app = create_app()
