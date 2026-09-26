# Offline Agentic RAG: Cyber Security Act 2024 (Act 854)

An assistant that answers questions about Malaysia's Cyber Security Act 2024 and two of its
regulations (P.U.(A) 220/2024 and P.U.(A) 219/2024), citing the exact sections. It runs fully
offline on one laptop. Requirements are in [PRD.md](PRD.md); working rules in [CLAUDE.md](CLAUDE.md).

Answers are informational and cite the source text; they are not legal advice.

> **Status: M2 (traditional mode) complete.** Ask a question in the Streamlit UI and get an answer
> with clickable citations, using hybrid retrieval (vector + BM25). Agentic mode arrives at M4;
> this README gains steps as each milestone lands.

## How it fits together

| Piece | Runs where | What it does |
|---|---|---|
| Postgres + pgvector | Docker (`db` service) | Stores documents, chunks and their embeddings |
| Ollama | Natively on Windows | Runs the chat model and the embedding model locally |
| Backend (FastAPI) | Docker (`backend` service), or on the host for development | API; talks to Postgres and Ollama only |
| Frontend (Streamlit) | Docker (`frontend` service), or on the host for development | UI; talks to the backend only |

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

**3. Install the frontend into the same venv** (only needed for running the UI outside Docker):

```powershell
pip install -r ..rontendequirements.txt
```

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

# Terminal 2, frontend/ with the same venv active: start the UI
cd frontend
..ackend\.venv\Scripts\Activate.ps1
streamlit run app.py
```

Open <http://localhost:8501>. The UI calls the API at `http://localhost:8000` (change with `BACKEND_URL`).

### B. Full stack in Docker

```powershell
docker compose up -d --build
```

This builds the images, starts `db`, `backend` and `frontend`, and creates the tables automatically.
Open <http://localhost:8501>. Ollama stays outside Docker; the backend reaches it through `host.docker.internal`.
Use A or B, not both at once: they use the same ports (8000 API, 8501 UI).
Ingestion (below) still runs from your machine, as in A; the containers see the same database.

### Check it is working

Open <http://localhost:8000/health> (or `curl http://localhost:8000/health`). A healthy system
returns HTTP 200 with `"status": "ok"`. HTTP 503 means something is wrong, and the response says what:

| Field | Meaning when it fails |
|---|---|
| `database.ok: false` | Postgres not running: `docker compose up -d db` |
| `ollama.ok: false` | Ollama not running: start it from the Start menu |
| `ollama.missing_models` | A configured model is not pulled: `ollama pull <name>` |
| `embedding.status: "mismatch"` | Stored vectors came from a different embedding model than `.env` says. The API also refuses to start in this case |

`embedding.status: "not_ingested"` means ingestion has not run yet (next section).

## Loading the documents (ingestion)

Run once after first setup, and again whenever a PDF, a config in `data/corpus/`, or the
embedding model changes. Needs the database and Ollama running. From `backend/` with the venv active:

```powershell
# 1. Read the PDFs, split them into chunks (one per section/regulation/definition),
#    check them against the table-of-contents checklists, and save them to the database.
python -m app.ingest.run_chunk --corpus-dir ../data/corpus --dump ../data/chunks.jsonl

# 2. Turn each chunk into an embedding (a list of 1024 numbers used for search) with Ollama.
python -m app.ingest.run_embed
```

Expected output: `act854: 85 chunks`, `pua219: 3 chunks`, `pua220: 3 chunks`, then `91 chunks to embed`.
Afterwards `/health` shows `embedding.status: "match"`.

- Both steps are safe to re-run. Chunks are matched by ID, so nothing is duplicated, and step 2
  only embeds chunks that are new or whose text changed.
- `--dump` writes every chunk to `data/chunks.jsonl`, handy for reading what the parser produced.
  Add `--no-db` to step 1 to only parse and check, without touching the database.
- If a PDF does not match its checklist, step 1 stops with a message naming the section and page;
  nothing is written.
- After changing `EMBED_MODEL`/`EMBED_DIM` in `.env`: `docker compose exec db psql -U rag -d rag -c "DROP TABLE chunks"`,
  then `python -m app.db.schema`, step 1, and step 2 with `--all`.

Interactive API docs: <http://localhost:8000/docs>.

## Asking a question

In the UI, type a question, keep **traditional** mode, and press **Ask**. The status box shows each
step as it runs (retrieving, writing the answer). The answer appears once, complete, followed by its
citations: click one to see the exact section text, document and page.

- The first question after starting takes ~40 s while Ollama loads the models into GPU memory;
  later ones take ~7–16 s (measured on an RTX 3050, 4 GB).
- Traditional mode answers are labelled *Not checked*: the automated source check comes with agentic mode.
- **Agentic** mode is not built yet (M4) and returns an error message.
- If the answer is not in the documents, the reply is *Not found in the provided documents*.

The API can also be called directly; it streams server-sent events:

```powershell
curl.exe -N -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d '{\"question\": \"How quickly must an incident be reported?\", \"mode\": \"traditional\"}'
```

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
    ingest/    PDF parsing, chunking, database insert, embedding
    llm/       Ollama client
  tests/       unit, api, integration
data/
  corpus/      per-document configs and table-of-contents checklists
  raw/         source PDFs (not committed)
docker-compose.yml
.env.example
```
