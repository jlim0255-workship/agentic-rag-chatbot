# Offline Agentic RAG: Cyber Security Act 2024 (Act 854)

An assistant that answers questions about Malaysia's Cyber Security Act 2024 and two of its
regulations (P.U.(A) 220/2024 and P.U.(A) 219/2024), citing the exact sections. It runs fully
offline on one laptop. Requirements are in [PRD.md](PRD.md); working rules in [CLAUDE.md](CLAUDE.md).

Answers are informational and cite the source text; they are not legal advice.

> **Status: M5 complete.** Ask a question in the Streamlit UI and get an answer with clickable
> citations, in **traditional** mode (retrieve once, answer) or **agentic** mode (split the question
> into sub-questions, grade the retrieved sources, rewrite the search and retry when evidence is
> missing, or answer "not found"). Source check and router come in M6–M7.

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
pip install -r ..rontend
equirements.txt
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
..backend\.venv\Scripts\Activate.ps1
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

Run once after first setup, and again whenever a PDF, a document card in `data/corpus/`, or the
embedding model changes. Needs the database and Ollama running. From `backend/` with the venv active:

```powershell
# 1. Read the PDFs, build each one's table of contents (outline) from the PDF, check it equals the
#    reviewed outline in data/corpus/, split into chunks (one per section/regulation/definition)
#    and save them to the database.
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
- Each document has a hand-written **document card** (`data/corpus/<doc>.toml`: title, page offset,
  running headers, the exact lines where the body starts and ends) and a generated, reviewed
  **outline** (`data/corpus/<doc>.outline.md`: Part, number and heading of every unit). Step 1 also
  writes `data/work/<doc>.lines.txt`, every PDF line with its printed page, for checking the card.
- If the outline built from the PDF differs from the committed one, or the body does not match the
  outline, step 1 stops with a message naming the unit and page; nothing is written. For a new or
  changed PDF, run step 1 with `--no-db --write-outline`, review the outline against the PDF, commit it.
- After changing `EMBED_MODEL`/`EMBED_DIM` in `.env`: `docker compose exec db psql -U rag -d rag -c "DROP TABLE chunks"`,
  then `python -m app.db.schema`, step 1, and step 2 with `--all`.

### Demo add-on: staff-directory tool (fictional data, `new_prd.md`)

In agentic mode the agent can call a `lookup_staff` tool (LangGraph `ToolNode`) on a table of 10 fictional
people; the contacts are appended under the answer by code. Load the table once (from `backend/`):

```powershell
python -m app.db.schema       # adds the demo_staff table (safe to re-run)
python -m app.db.seed_staff   # 10 fictional people
```

Try: "Ransomware has encrypted our patient records. Who do I contact, and by when?"

Interactive API docs: <http://localhost:8000/docs>.

## Asking a question

The UI is a chat page: pick **traditional** or **agentic** mode in the sidebar, type in the box at the
bottom and press Enter. While a question runs, a status box shows each step live. The answer then appears
once, complete, with its label, timing, and two collapsed sections: **Sources** (the exact section text,
document and page for each citation) and **Trace** (every step's decision). Earlier questions and answers
stay on the page for this browser tab; **Clear conversation** in the sidebar removes them. Each question
is answered on its own: the model never sees earlier turns, so include the details you need.

- The first question after starting takes ~40 s while Ollama loads the models into GPU memory;
  later ones take ~7–16 s (measured on an RTX 3050, 4 GB).
- **Agentic** answers go through an automated source check before they are shown: every cited ID must
  be one of the sources given, every factual sentence must have a citation, and the model checks each
  cited sentence against its source. The label reads *Passed automated source check*, or says how many
  sentences were flagged; flagged sentences carry a visible ⚠ note. It is a small model judging, so it
  can be wrong both ways: check the cited source text. **Traditional** answers are labelled *Not checked*.
- **Agentic** mode first routes the question: greetings and off-topic questions (e.g. "what is 1+1?")
  get a fixed reply with no search; simple questions skip the grading loop; complex ones get the full
  agent (sub-questions, grading with verified quotes, rewritten searches). It makes more LLM calls, so it
  is slower. Expand **Trace** under an answer to see every decision.
- The API server logs the same steps, one line each, tagged with a short request id:
  `docker compose logs -f backend` (or the terminal running `uvicorn`).
- If the answer is not in the documents, the reply is *Not found in the provided documents*.

The API can also be called directly; it streams server-sent events:

```powershell
curl.exe -N -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d '{\"question\": \"How quickly must an incident be reported?\", \"mode\": \"traditional\"}'
```

## Evaluation

The golden set is `backend/eval/golden.jsonl`: 30 questions (10 single-hop, 10 multi-hop, 5 not in the
documents, 5 ambiguous), each with the sections that answer it and the key facts a correct answer contains.
From `backend/` with the venv active (database ingested, Ollama running):

```powershell
# Experiment A: retrieval only, no LLM (~1 minute). Compares semantic, BM25 and hybrid search.
python -m eval.run retrieval

# Experiment B: traditional vs agentic, full answers through the API (API must be running; ~25 minutes).
python -m eval.run e2e
# ... or one mode only
python -m eval.run e2e --mode agentic
```

Reports are written to `backend/eval/reports/`, timestamped (Markdown + CSV):

- `*-retrieval.md`: a scoreboard (✅ all needed sections found / 🟡 some / ❌ none, per method), results by
  category, a per-question table, and the questions where the methods disagree.
- `*-e2e.md`: the same layout for traditional vs agentic (evidence given to the writer, final status,
  source check), plus the rewrite-similarity table used to calibrate the drift gate.
- `*-e2e-<mode>-answers.md`: every answer next to the key facts a correct answer contains, for manual review.

Every rate is shown with its count, e.g. `7/10 (70%)`. Experiment B talks to whatever API is running:
after changing backend code, rebuild the container first (`docker compose up -d --build backend`).

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
    api/         FastAPI app: /chat (SSE), /chunks, /health
    config.py    settings loaded from .env
    db/          connection, schema, embedding-model record
    ingest/      PDF parsing, chunking, database insert, embedding
    llm/         Ollama client, structured (JSON-schema) output
    retrieval/   hybrid search: vector + BM25 + RRF
    pipeline/    how a question is answered: START with flow.py (flow map of both modes)
      core.py    shared plumbing: state, time budget, trace/SSE events, citations
      nodes/     one file per step: planner, retrieve, grade, rewrite, not_found, generate, grounding
  eval/          golden set, evaluation runner, reports/
  tests/         unit, api, pipeline (LLM faked), integration (real services); fakes.py = shared fakes
frontend/        Streamlit UI (talks to the API only)
data/
  corpus/      per document: card (<doc>.toml, hand-written) and outline (<doc>.outline.md, reviewed)
  work/        extracted PDF lines for checking (generated, not committed)
  raw/         source PDFs (not committed)
docker-compose.yml
.env.example
```
