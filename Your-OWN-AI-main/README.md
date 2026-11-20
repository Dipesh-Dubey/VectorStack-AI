# VectorStack-AI

VectorStack-AI is a pure Python vector database and local RAG application. It
provides an interactive browser UI for experimenting with vector search,
document retrieval, and question answering through Ollama.

## Architecture

The application has one Python backend and one static frontend:

```text
Browser (index.html)
        │ HTTP
        ▼
FastAPI / Uvicorn (app.py)
        │
        ├── VectorDB
        │   ├── Brute Force
        │   ├── KD-Tree
        │   └── HNSW
        │
        └── DocumentDB / RAG service
            ├── document chunking
            ├── Ollama nomic-embed-text embeddings
            ├── similarity retrieval
            └── Ollama llama3.2 answers
```

All vector indexes and distance calculations are implemented in Python. The
application does not require a compiler or a native vector database.

## Features

- Brute Force, KD-Tree, and HNSW nearest-neighbor search.
- Euclidean, cosine, and Manhattan distance metrics.
- Vector insertion, deletion, listing, search, benchmarking, and HNSW
  inspection.
- Document chunking into 250-word chunks with 30-word overlap.
- Local embeddings with `nomic-embed-text`.
- Similarity-based document retrieval and RAG answers with `llama3.2`.
- A browser UI for vector visualization, document management, and RAG.

## Technology Stack

- Python 3.10+
- FastAPI and Pydantic
- Uvicorn
- NumPy
- Requests
- Ollama
- HTML, CSS, and JavaScript frontend

## Project Structure

```text
Your-OWN-AI-main/
├── app.py                         # Python FastAPI entrypoint
├── index.html                     # Browser frontend
├── requirements.txt
└── vectordb/
    ├── app.py                     # Application exports
    ├── config.py                  # Runtime settings
    ├── algorithms/
    │   ├── bruteforce.py
    │   ├── kdtree.py
    │   └── hnsw.py
    ├── api/
    │   └── routes.py              # FastAPI models and endpoints
    ├── database/
    │   ├── vector_db.py
    │   └── document_db.py
    ├── models/
    ├── services/
    │   ├── chunker.py
    │   ├── ollama_client.py
    │   └── rag_service.py
    └── utils/
        └── metrics.py
```

## Installation

Create and activate a virtual environment:

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the Python dependencies:

```bash
python -m pip install -r requirements.txt
```

Install Ollama separately, then download the models used by the RAG pipeline:

```bash
ollama pull nomic-embed-text
ollama pull llama3.2
```

## Running

Start Ollama if it is not already running:

```bash
ollama serve
```

From the project directory, start the Python backend:

```bash
python -m uvicorn app:app --host 0.0.0.0 --port 8080
```

Open [http://localhost:8080](http://localhost:8080) in a browser. The
frontend sends its requests to this FastAPI server.

## API

### Vector endpoints

| Method | Endpoint | Description |
| --- | --- | --- |
| GET | `/search?v=...&k=5&metric=cosine&algo=hnsw` | K-nearest-neighbor search |
| POST | `/insert` | Insert a vector |
| DELETE | `/delete/{id}` | Delete a vector |
| GET | `/items` | List vectors |
| GET | `/benchmark?v=...&k=5&metric=cosine` | Compare all algorithms |
| GET | `/hnsw-info` | Inspect the HNSW graph |
| GET | `/stats` | Vector database statistics |

Example vector insertion:

```bash
curl -X POST http://localhost:8080/insert \
  -H "Content-Type: application/json" \
  -d '{"metadata":"example","category":"demo","embedding":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]}'
```

### Document and RAG endpoints

| Method | Endpoint | Description |
| --- | --- | --- |
| POST | `/doc/insert` | Chunk, embed, and store a document |
| DELETE | `/doc/delete/{id}` | Delete a document chunk |
| GET | `/doc/list` | List stored documents |
| POST | `/doc/search` | Embed a question and retrieve contexts |
| POST | `/doc/ask` | Retrieve contexts and generate an answer |
| GET | `/status` | Ollama and database status |

Insert a document:

```bash
curl -X POST http://localhost:8080/doc/insert \
  -H "Content-Type: application/json" \
  -d '{"title":"Example","text":"Vector databases retrieve similar embeddings."}'
```

Ask a question:

```bash
curl -X POST http://localhost:8080/doc/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What do vector databases retrieve?","k":3}'
```

### Ollama proxy endpoints

`GET /api/tags`, `POST /api/embeddings`, and `POST /api/generate` proxy the
corresponding local Ollama APIs.

## Search Algorithms and Metrics

`VectorDB` keeps the custom Brute Force, KD-Tree, and HNSW implementations
available through the `algo` query parameter. Search uses the selected
Euclidean, cosine, or Manhattan distance function. Document retrieval uses
cosine distance and selects Brute Force for small collections and HNSW for
larger collections.

## RAG Flow

Documents are split into overlapping chunks, embedded with
`nomic-embed-text`, and stored in `DocumentDB`. A question is embedded with the
same model, relevant chunks are retrieved by similarity, and the chunks are
included in a prompt sent to `llama3.2`.

If Ollama is unavailable, vector search remains available and document/RAG
requests return an explicit availability error.
