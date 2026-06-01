# VectorDB in Python

This repository now contains a from-scratch Python conversion of the original C++ VectorDB project.

## Run

```bash
pip install -r requirements.txt
uvicorn app:app --reload
```

## Endpoints

- `GET /search`
- `POST /insert`
- `DELETE /delete/{id}`
- `GET /items`
- `GET /benchmark`
- `GET /hnsw-info`
- `POST /doc/insert`
- `DELETE /doc/delete/{id}`
- `GET /doc/list`
- `POST /doc/search`
- `POST /doc/ask`
- `GET /status`
- `GET /stats`
- `GET /api/tags`
- `POST /api/embeddings`
- `POST /api/generate`

## Notes

- HNSW, KD-Tree, and brute force are implemented from scratch in Python.
- Document chunking uses 250-word chunks with 30-word overlap.
- The root `/` route serves the existing `index.html` UI.
- The main entrypoint is the root `app.py` file.
