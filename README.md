# Offline Agentic RAG: Cyber Security Act 2024 (Act 854)

Answers questions about Malaysia's Cyber Security Act 2024 and two of its regulations
(P.U.(A) 219/2024 and P.U.(A) 220/2024), citing the exact sections. Runs fully offline on one laptop.
Two modes: **traditional** (retrieve once, answer) and **agentic** (route, plan sub-questions, grade the
sources, rewrite the search and retry, or answer "not found"). Answers are informational, not legal advice.

| Piece | Runs where | What it does |
|---|---|---|
| Postgres + pgvector | Docker (`db`) | Documents, chunks, embeddings |
| Ollama | Natively on Windows | Chat model and embedding model |
| Backend (FastAPI + LangGraph) | Docker (`backend`) or host | API; talks to Postgres and Ollama only |
| Frontend (Streamlit) | Docker (`frontend`) or host | UI; talks to the backend only |

## Setup

Prerequisites: Python 3.11.3, Docker Desktop (WSL2), Ollama for Windows, and the source PDFs in `data/raw/`
(`Act-854.pdf`, `PUA-219.pdf`, `PUA-220.pdf`; not committed). Pull the models once; after that nothing needs
the internet:

```powershell
ollama pull gemma4:e2b
ollama pull qwen3-embedding:0.6b
```

From the repo root (PowerShell):

```powershell
Copy-Item .env.example .env          # defaults work as-is
cd backend
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1         # if scripts are blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install -e ".[dev]"
pip install -r ..\frontend\requirements.txt   # only to run the UI outside Docker
```

`EMBED_MODEL` and `EMBED_DIM` in `.env` are tied: change both together, then re-ingest.

## Running

**Development** (database in Docker, backend and UI on the host):

```powershell
docker compose up -d db
cd backend
python -m app.db.schema                  # create tables; safe to re-run
uvicorn app.api.main:app --reload        # API on http://localhost:8000
# second terminal, same venv:
cd frontend
streamlit run app.py                     # UI on http://localhost:8501
```

**Full stack in Docker:** `docker compose up -d --build` (creates the tables itself). Ollama stays outside
Docker; the backend reaches it through `host.docker.internal`. Use one way or the other: both use ports 8000 and 8501.

**Health:** `GET http://localhost:8000/health` returns 200 with `"status": "ok"`, or 503 saying what is wrong:

| Field | Meaning when it fails |
|---|---|
| `database.ok: false` | Postgres not running |
| `ollama.ok: false` | Ollama not running |
| `ollama.missing_models` | A configured model is not pulled |
| `embedding.status: "mismatch"` | Stored vectors come from a different embedding model than `.env` (the API also refuses to start) |
| `embedding.status: "not_ingested"` | Ingestion has not run yet |

Other endpoints: `POST /chat` (server-sent events), `GET /chunks/{id}`, docs at <http://localhost:8000/docs>.

## Ingestion

Run once, and again when a file in `data/llamaparsed/` or the embedding model changes. From `backend/`:

```powershell
python -m app.ingest.run_chunk --dump ../data/chunks.jsonl   # parse + chunk + save (add --no-db to only check)
python -m app.ingest.run_embed                               # embed new or changed chunks (--all for every chunk)
```

Expected: `act854: 85 chunks`, `pua219: 3 chunks`, `pua220: 3 chunks`, then `91 chunks to embed`. Both steps are
safe to re-run (chunks are matched by ID).

The input is `data/llamaparsed/<doc>.md`: each PDF converted once by hand with LlamaParse (never called by the
app) and corrected into a fixed format (`# Part X: Title`, `## N. Heading`, `<!-- page N -->` markers, text as
printed), with a document card between `+++` lines at the top. `run_chunk` stops, writing nothing, if a file
breaks the format (unknown heading, non-consecutive section numbers, a missing `## N.` heading).

After changing the embedding model: `docker compose exec db psql -U rag -d rag -c "DROP TABLE chunks"`, then
`python -m app.db.schema`, `run_chunk`, and `run_embed --all`.

## Asking a question

In the UI, pick **traditional** or **agentic** in the sidebar and type a question. A status box shows each
step live; the answer then shows its label, timing, **Sources** (cited section text, document, page) and
**Trace** (every step's decision). Each question is answered on its own: the model does not see earlier turns.

- First question after start-up takes ~40 s (models loading); later ones ~7–16 s on an RTX 3050 4 GB.
- Answers are labelled *Not checked: no automated source check*. Only sources given to the writer can be cited.
- Agentic mode routes first: off-topic questions get a fixed reply with no search. It makes more LLM calls than traditional.
- If the documents do not answer the question, the reply is *Not found in the provided documents*.
- The API logs every step with a request id (`docker compose logs -f backend` or the `uvicorn` terminal).

```powershell
curl.exe -N -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d '{\"question\": \"How quickly must an incident be reported?\", \"mode\": \"traditional\"}'
```

**Staff-directory tool (demo, fictional data).** In agentic mode the agent can call `lookup_staff` (LangGraph
`ToolNode`) on 10 fictional people; the contacts are appended under the answer by code. Load them once from
`backend/` with `python -m app.db.seed_staff` (creates the `employee_demo` database; safe to re-run). Try:
"Ransomware has encrypted our patient records. Who do I contact, and by when?"

## Evaluation

Golden set: `backend/eval/golden.jsonl`, 30 questions (10 single-hop, 10 multi-hop, 5 not in the documents,
5 ambiguous), each with the sections that answer it and the key facts a correct answer contains. From `backend/`:

```powershell
python -m eval.run retrieval                      # semantic vs BM25 vs hybrid, no LLM (~1 min)
python -m eval.run e2e [--mode agentic]           # traditional vs agentic through the running API (~25 min)
python -m eval.run_cloud --model gemma4:31b-cloud # same, in-process, with another model
python -m eval.judge eval/reports/<stem>-rows.jsonl [...] [--judge gpt-oss:120b-cloud]   # LLM-as-judge
```

Reports go to `backend/eval/reports/`, timestamped (Markdown + CSV):

- `*-retrieval.md`: FULL / PART / MISS (all / some / none of the needed sections found) per method, by category, best-match similarity.
- `*-e2e*.md`: the same for traditional vs agentic, plus status, latency, LLM calls; `*-answers.md` lists every answer next to its key facts.
- `*-rows.jsonl`: raw answers, input to `eval.judge`.
- `*-judge-<model>.md`: judged correctness, faithfulness and relevance per model × mode, and why answers were wrong.

Every rate is shown with its count, e.g. `7/10 (70%)`. Judge scores come from an LLM and can be wrong in both
directions. `-cloud` models send questions and chunks to Ollama Cloud: evaluation only, the app stays offline.
`eval.run e2e` tests whatever API is running, so rebuild it after code changes (`docker compose up -d --build backend`).

## Tests

```powershell
python -m pytest                  # unit, API and pipeline tests; Postgres and Ollama are faked
python -m pytest -m integration   # against real Postgres and Ollama
```

## Stopping

```powershell
docker compose down      # stop; the database is kept
docker compose down -v   # stop and delete the database (re-run schema and ingestion)
```

## Layout

```
backend/
  app/
    api/         FastAPI: /chat (SSE), /chunks, /health
    config.py    settings from .env (all model names, URLs, limits)
    db/          connection, schema, embedding-model record, staff demo seed
    ingest/      Markdown reader, chunker, database insert, embedding
    llm/         Ollama client, structured (JSON-schema) output with retry
    retrieval/   hybrid search: pgvector + BM25, fused with RRF
    pipeline/    flow.py = map of both modes (start here); core.py = shared plumbing; tools.py = lookup_staff
      nodes/     one file per step: planner, tool_agent, retrieve, grade, rewrite, not_found, generate
  eval/          golden set, runners, judge, reports/
  tests/         unit, api, pipeline (LLM faked), integration; fakes.py
frontend/        Streamlit UI
data/
  llamaparsed/   ingestion input: <doc>.md per document
  raw/           source PDFs (not committed)
docker-compose.yml, .env.example
```
