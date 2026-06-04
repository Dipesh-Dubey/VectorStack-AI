# VectorDB in Python

A fully functional **Vector Database built from scratch in Python**, featuring **HNSW**, **KD-Tree**, and **Brute Force** search algorithms, semantic document retrieval, and a complete **Retrieval-Augmented Generation (RAG)** pipeline powered by local LLMs through Ollama.

This project demonstrates how modern vector databases such as Pinecone, Weaviate, Chroma, and Milvus operate internally, while providing an interactive web interface for experimentation, benchmarking, document indexing, and AI-powered question answering.

---

## Features

### Vector Search Engine

* HNSW (Hierarchical Navigable Small World) implementation from scratch
* KD-Tree implementation from scratch
* Brute Force nearest-neighbor search
* Side-by-side algorithm benchmarking
* Multiple distance metrics:

  * Cosine Similarity
  * Euclidean Distance
  * Manhattan Distance

### Semantic Search

* 20 preloaded demonstration vectors across multiple domains
* Real-time nearest-neighbor retrieval
* Semantic similarity exploration
* PCA-based visualization of vector space

### Document Intelligence

* Automatic document chunking
* 250-word chunks with 30-word overlap
* Embedding generation using Ollama
* Vector indexing for semantic retrieval
* Document management APIs

### RAG Pipeline

* Question embedding generation
* Context retrieval using HNSW
* Local LLM response generation
* Fully offline AI workflow
* No external API dependencies

### REST API

* CRUD operations for vectors
* Document ingestion and retrieval
* Search and benchmarking endpoints
* HNSW graph inspection
* Ollama integration endpoints
* System statistics and monitoring

---

## Architecture

```text
Your Document
      │
      ▼
Ollama Embeddings
(nomic-embed-text)
      │
      ▼
Document Chunking
250 words + overlap
      │
      ▼
HNSW Index (Python)
      │
      ▼
Semantic Retrieval
Top-K Relevant Chunks
      │
      ▼
Ollama LLM
(llama3.2)
      │
      ▼
Generated Answer
```

---

## Project Structure

```text
VectorDB-Python/
│
├── app.py                # Main FastAPI application
├── index.html            # Frontend UI
├── requirements.txt      # Python dependencies
│
├── algorithms/
│   ├── hnsw.py
│   ├── kdtree.py
│   └── bruteforce.py
│
├── database/
│   ├── vectordb.py
│   └── documentdb.py
│
├── services/
│   ├── embeddings.py
│   ├── rag.py
│   └── ollama_client.py
│
└── README.md
```

---

## Requirements

### Software

* Python 3.10+
* Ollama
* Git

### Ollama Models

Pull the required models:

```bash
ollama pull nomic-embed-text
ollama pull llama3.2
```

Verify installation:

```bash
ollama list
```

---

## Installation

### Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/VectorDB-Python.git

cd VectorDB-Python
```

### Create Virtual Environment

#### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

#### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Application

Start Ollama:

```bash
ollama serve
```

Run the FastAPI server:

```bash
python -m uvicorn app:app \
  --host 0.0.0.0 \
  --port 8080 \
  --reload
```

Open:

```text
http://localhost:8080
```

---

## API Endpoints

### Vector Search

| Method | Endpoint     | Description           |
| ------ | ------------ | --------------------- |
| GET    | /search      | K-NN vector search    |
| POST   | /insert      | Insert vector         |
| DELETE | /delete/{id} | Delete vector         |
| GET    | /items       | List vectors          |
| GET    | /benchmark   | Compare algorithms    |
| GET    | /hnsw-info   | HNSW graph statistics |
| GET    | /stats       | Database statistics   |

### Documents

| Method | Endpoint         |
| ------ | ---------------- |
| POST   | /doc/insert      |
| DELETE | /doc/delete/{id} |
| GET    | /doc/list        |
| POST   | /doc/search      |

### RAG

| Method | Endpoint |
| ------ | -------- |
| POST   | /doc/ask |
| GET    | /status  |

### Ollama-Compatible APIs

| Method | Endpoint        |
| ------ | --------------- |
| GET    | /api/tags       |
| POST   | /api/embeddings |
| POST   | /api/generate   |

---

## Search Algorithms

### HNSW

Hierarchical Navigable Small World graphs provide near-logarithmic search complexity and power many modern vector databases.

Advantages:

* Extremely fast approximate search
* Scales to millions of vectors
* Excellent high-dimensional performance

### KD-Tree

A binary space-partitioning structure that performs efficient exact nearest-neighbor search in lower-dimensional spaces.

Advantages:

* Exact results
* Efficient for low-dimensional vectors
* Memory efficient

### Brute Force

Computes distances against every vector.

Advantages:

* Always exact
* Useful as a baseline benchmark
* Simple and predictable

---

## Why HNSW?

As vector dimensionality increases, KD-Tree pruning becomes less effective due to the curse of dimensionality.

HNSW avoids this limitation by navigating a graph structure rather than partitioning geometric space, making it the preferred choice for high-dimensional embedding search.

---

## Technology Stack

### Backend

* Python
* FastAPI
* NumPy
* Uvicorn

### AI & Retrieval

* Ollama
* nomic-embed-text
* llama3.2

### Algorithms

* HNSW
* KD-Tree
* Brute Force

### Frontend

* HTML
* CSS
* JavaScript

---

## Educational Goals

This project is designed to help developers understand:

* How vector databases work internally
* Approximate nearest-neighbor search
* HNSW graph construction
* Semantic embeddings
* Retrieval-Augmented Generation (RAG)
* Local LLM deployment
* Production search system architecture

---

## Future Improvements

* Persistent storage layer
* Hybrid keyword + vector search
* Multi-user support
* GPU acceleration
* Distributed indexing
* Metadata filtering
* Hybrid retrieval ranking
* Quantization and compression

---

## License

MIT License

Feel free to use, modify, and extend this project for educational, research, or commercial purposes.
