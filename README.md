# VectorDB in Python

This repository now contains a from-scratch Python conversion of the original C++ VectorDB project.

## Run

```bash
# Create and activate a virtual environment (recommended)
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the app with uvicorn (production: drop --reload)
python -m uvicorn app:app --host 0.0.0.0 --port 8080 --reload
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
