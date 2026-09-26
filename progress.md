# Progress

Running log of what was built, decided and found, milestone by milestone (PRD Section 15).
Newest milestone at the top of each section. Commits are made by Jlim after each milestone.

## Milestone status

| # | Milestone | Status | Tests at completion |
|---|---|---|---|
| M0 | Project setup | Done (2026-09-26) | 24 unit/API + 1 integration |
| M1 | Ingestion | Done (2026-09-26) | 51 unit/API + 2 integration |
| M2 | Traditional MVP | Done (2026-09-26) | 78 unit/API + 5 integration |
| M3 | Evaluation baseline | Next: golden set being drafted | |
| M4 | Grade + rewrite + not-found loop | | |
| M5 | Planner | | |
| M6 | Grounding check + regenerate | | |
| M7 | Router | | |
| M8 | Trace panel + compare view | | |
| M9 | Polish | | |

## M2: Traditional MVP

**Built**
- Hybrid retrieval (`backend/app/retrieval/`): pgvector cosine + BM25 (stdlib, ~20 lines), fused with RRF;
  optional document/Part filters applied to both searches before fusion.
- Generate (`backend/app/graph/generate.py`): answer from the retrieved chunks only, citing chunk IDs; `NOT_FOUND` when unanswerable.
- Traditional pipeline (`backend/app/graph/traditional.py`) yielding SSE events in PRD 11.2 order.
- API: `POST /chat` (SSE), `GET /chunks/{id}`.
- Streamlit UI (`frontend/app.py`): question, mode toggle, live status line, answer, clickable citations, metrics.
- Compose `frontend` service.

**Decisions** (were `TODO(Jlim)`; awaiting Jlim's confirmation)
- BM25: own stdlib implementation instead of `bm25s`/`rank_bm25` (no dependency).
- `TOP_K_VECTOR=20`, `TOP_K_BM25=20`, `TOP_K_FINAL=5`.
- `LLM_CALL_TIMEOUT_S=90` from measurement; `QUESTION_BUDGET_S=300` (revisit at M4); `LLM_NUM_CTX=8192`.
- Chat calls use `think: false`, `temperature: 0`.

**Deviations from the PRD**
- BM25 is built per request, not at startup: milliseconds for 91 chunks, never stale after re-ingestion.
- Vector search is exact (no HNSW use): instant at this size, and avoids pgvector returning too few rows with a filter.

**Found and fixed**
- Streamlit called `checkip.amazonaws.com` at startup (offline-rule violation). Fixed with
  `browser.serverAddress = "localhost"` in `frontend/.streamlit/config.toml`; verified no call is made.
- Model writes citations like `[act854:s22:1(1)(a)]`; parser now extracts the ID.
- Model sometimes writes a sentence then `NOT_FOUND`; treated as not found when nothing valid is cited.

**Measured** (RTX 3050 4 GB): first question ~41 s (models loading into VRAM), then 7–25 s per question.

**Weaknesses seen (for M3 to measure)**
- "Which sectors are NCII sectors?" misses the Schedule: BM25 has no stemming ("sectors" ≠ "sector").
- Answers are long, copied from the source, and sometimes skip part of a multi-part question.

## M1: Ingestion

**Built**
- Parser (`backend/app/ingest/`): page-by-page extraction, running headers stripped, checklist-guided
  section detection, Parts, definition splitting, contextual headers, deterministic chunk IDs.
- DB insert (idempotent upsert; embeddings kept when text unchanged) and embed step (records model in `ingest_meta`).

**Results**: Act 854 → 85 chunks (long title + 63 sections + 20 definitions + Schedule); P.U.(A) 219 → 3; P.U.(A) 220 → 3.
Re-run identical; 10/10 spot check of pages correct; 91 chunks embedded in ~29 s.

**Decisions**
- s4 intro sentence dropped from definition chunks (header carries "Section 4: Interpretation — “term”").
- `part` stored as Roman numeral (`IV`) plus new `part_title` column (title case). PRD 8.3 updated.
- Config paths relative to each config file.
- Dropped the `pgvector` Python package (vectors passed as text); only `pypdf` added.
- New settings: `EMBED_BATCH_SIZE=16`, `EMBED_TIMEOUT_S=120`.

**Known cosmetic**: six chunks keep a stray space before punctuation from the PDF (e.g. `Gazette .`).

## M0: Project setup

**Built**: settings from `.env` (`backend/app/config.py`), Compose (`db` pgvector/pg16, `backend`),
schema created from Python with `EMBED_DIM`, `ingest_meta` table, `/health` (DB, Ollama, models,
embedding-model match), startup refusal on embedding-model mismatch.

**Decisions**: embedding model `qwen3-embedding:0.6b` (1024 dims, checked with `ollama show`);
Python 3.11.3; `psycopg[binary]` driver; `httpx`; PRD 8.3/8.5 updated (Insert keeps unchanged embeddings; `ingest_meta`).

## Open items

- Confirm the M2 decisions above.
- M3: golden set drafted in `backend/eval/golden.jsonl` (25 questions: 9 single-hop, 8 multi-hop,
  5 not in corpus, 3 ambiguous). Every entry has `"reviewed": false` until Jlim checks it.
  Finding while drafting: multi-hop questions written in the Act's own vocabulary were fully retrieved by
  traditional mode (7/8), so they were rewritten in lay language / with a bridging step, as the PRD's
  user would ask. Retrieval-only check of the rewritten set: 5 of 8 multi-hop miss at least one needed section.
- `PRD.md` and `CLAUDE.md` to be committed at the last milestone.
- Ponytail skill enabled on the account but not loadable in the current session.
