# Offline Agentic RAG: Cyber Security Act 2024 (Act 854)

An assistant that answers questions about Malaysia's Cyber Security Act 2024 and two of its
regulations (P.U.(A) 220/2024 and P.U.(A) 219/2024), citing the exact sections. It runs fully
offline on one laptop. Requirements are in [PRD.md](PRD.md); working rules in [CLAUDE.md](CLAUDE.md).

Answers are informational and cite the source text; they are not legal advice.

> **Status: M0 (project setup) complete.** The database, schema, config and `/health` endpoint work.
> Ingestion (M1), question answering (M2+) and the Streamlit UI are not built yet; this README
> gains steps as each milestone lands.

## How it fits together

| Piece | Runs where | What it does |
|---|---|---|
| Postgres + pgvector | Docker (`db` service) | Stores documents, chunks and their embeddings |
| Ollama | Natively on Windows | Runs the chat model and the embedding model locally |
| Backend (FastAPI) | Docker (`backend` service), or on the host for development | API; talks to Postgres and Ollama only |
| Frontend (Streamlit) | *(added at M2)* | UI; talks to the backend only |

## Prerequisites (install once)

1. **Python 3.11.3.** Check with `py -3.11 --version`.
2. **Docker Desktop** (WSL2 backend). Start it and wait until it says *Engine running*.
3. **Ollama for Windows**, running (it sits in the system tray). Pull the models:

   ```powershell
   ollama pull gemma4:e2b
   ollama pull qwen3-embedding:0.6b
   ```

   These downloads are the last time anything needs the internet. After that, the system
   makes no network calls except to local Postgres and Ollama.
4. **The source PDFs** in `data/raw/` (not committed to git): `Act-854.pdf`, `PUA-219.pdf`, `PUA-220.pdf`.

## First-time setup

All commands are PowerShell, run from the repository root unless stated.

**1. Create your settings file.**

```powershell
Copy-Item .env.example .env
```

`.env` holds model names, the database URL and limits. The defaults work as-is.
`EMBED_MODEL` and `EMBED_DIM` must always change together, and changing them means re-running ingestion.

**2. Create the Python virtual environment and install the backend.**

A virtual environment (`.venv`) is a private folder of Python packages for this project only,
so nothing is installed into your system Python.

```powershell
cd backend
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

After `Activate.ps1`, your prompt starts with `(.venv)`; `python` and `pip` now mean the
project's copies. Run `deactivate` to leave it.

If PowerShell says *running scripts is disabled on this system*, allow local scripts for your
user once with `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, or skip activation and
call the venv's Python directly: `.\.venv\Scripts\python -m pip install -e ".[dev]"`.

`-e` installs the backend in editable mode, so code changes take effect without reinstalling.
`[dev]` adds the test tools.

## Running it

There are two ways. Use **A** while developing (fast reload), **B** to run everything the way the demo runs.

### A. Development: database in Docker, backend on your machine

```powershell
# Terminal 1, repo root: start Postgres
docker compose up -d db

# Terminal 1, backend/ with the venv active: create the tables (safe to re-run)
cd backend
.\.venv\Scripts\Activate.ps1
python -m app.db.schema

# Start the API; --reload restarts it when you edit code
uvicorn app.api.main:app --reload
```

### B. Full stack in Docker

```powershell
docker compose up -d --build
```

This builds the backend image, starts `db` and `backend`, and creates the tables automatically.
Ollama stays outside Docker; the container reaches it through `host.docker.internal`.
Use A or B, not both at once: both serve the API on port 8000.

### Check it is working

Open <http://localhost:8000/health> (or `curl http://localhost:8000/health`). A healthy system
returns HTTP 200 with `"status": "ok"`. HTTP 503 means something is wrong, and the response says what:

| Field | Meaning when it fails |
|---|---|
| `database.ok: false` | Postgres not running: `docker compose up -d db` |
| `ollama.ok: false` | Ollama not running: start it from the Start menu |
| `ollama.missing_models` | A configured model is not pulled: `ollama pull <name>` |
| `embedding.status: "mismatch"` | Stored vectors came from a different embedding model than `.env` says. The API also refuses to start in this case |

`embedding.status: "not_ingested"` is normal until ingestion (M1) has run.

Interactive API docs: <http://localhost:8000/docs>.

## Tests

From `backend/` with the venv active:

```powershell
python -m pytest                  # unit + API tests; Postgres and Ollama are faked, nothing needs to run
python -m pytest -m integration   # against the real Postgres and Ollama (start db and Ollama first)
```

## Stopping and resetting

```powershell
docker compose down      # stop containers; the database is kept
docker compose down -v   # stop AND delete the database (you would re-run the schema and ingestion)
```

## Repository layout

```
backend/
  app/
    api/       FastAPI app and /health
    config.py  settings loaded from .env
    db/        connection, schema, embedding-model record
    ingest/    PDF parsing and chunking (M1)
    llm/       Ollama client
  tests/       unit, api, integration
data/
  corpus/      per-document configs and table-of-contents checklists
  raw/         source PDFs (not committed)
docker-compose.yml
.env.example
```
