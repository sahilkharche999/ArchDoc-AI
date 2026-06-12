# DAX Steel Estimator — Backend

DAX is an AI-powered backend that analyzes structural steel construction drawings and automatically generates a **Bill of Materials (BOM)** with quantities, weights, and costs from multi-page PDFs.

The system combines:

* Vision models (Grounding DINO symbol detection + Gemini OCR)
* LLM reasoning (Google Gemini 2.5 Pro / Flash-Lite)
* A graph database with vector search (Memgraph)
* A stateful multi-agent workflow (LangGraph)

---

## Features

* Upload structural drawing PDFs (trimmed to a page range on upload)
* Automatic page classification (text / floor / section)
* MinerU layout + table detection
* Symbol detection using **Grounding DINO + Gemini OCR**
* Detail and schedule extraction using **Gemini vision models**
* GraphRAG retrieval of schedules, rules, and detail BOMs (Memgraph vector search)
* Multi-agent pipeline orchestrated by **LangGraph** with pause/resume HITL
* Human-in-the-loop review of page classification and detection boxes
* Final Bill of Materials with pricing, written to disk as JSON
* Real-time workflow streaming via **Server-Sent Events** (Redis pub-sub)
* Per-user encrypted Gemini API keys; JWT auth with invite-code registration

---

## Technology Stack

| Layer | Technology |
| --- | --- |
| Pipeline orchestration | LangGraph (stateful multi-agent graph) |
| LLM / Vision | Google Gemini 2.5 Pro, Gemini 2.5 Flash-Lite |
| PDF extraction | MinerU 3.1.8 (layout + table detection) |
| Symbol detection | Grounding DINO + Gemini OCR |
| Graph database | Memgraph (Neo4j-compatible, vector search) |
| Checkpoint store | PostgreSQL via LangGraph PostgresSaver |
| Job queue / pub-sub | Redis |
| API | FastAPI + Uvicorn |
| Serving | Nginx (static frontend) + FastAPI (API) |
| Infrastructure | AWS EC2 t3.large, Ubuntu 24, systemd |

---

## Project Structure

```
backend
│
├── src
│   ├── api                     # FastAPI layer
│   │   ├── routes
│   │   │   ├── auth.py          # JWT register/login/me, save Gemini key
│   │   │   ├── upload.py        # PDF upload + page-range trim
│   │   │   ├── jobs.py          # start, SSE stream, HITL submit, result, BOM patch
│   │   │   ├── projects.py      # list/get/rename/delete projects
│   │   │   └── pdf_fixes.py     # page rotation fix
│   │   └── main.py              # FastAPI app + JWT auth middleware
│   │
│   ├── db                      # Postgres helpers (jobs, jobs_progress, users)
│   │   └── checkpoint.py        # LangGraph PostgresSaver setup
│   │
│   ├── infrastructure
│   │   ├── graph_db.py          # Memgraph wrapper (lookup, semantic search, writes)
│   │   └── symbol_detection.py  # Grounding DINO + Gemini OCR
│   │
│   ├── workflow
│   │   ├── common               # state.py, schemas.py, utils.py
│   │   └── workflows
│   │       └── estimation        # graph.py, nodes.py (Agents 0–4), prompt.py
│   │
│   ├── logger.py                # Central logging setup
│   ├── redis_conn.py            # Redis client
│   └── service.py               # LangGraph runner (stream_estimation)
│
├── requirements.txt
├── requirements-ml.txt          # Heavy ML deps (DINO/torch), kept separate
└── README.md
```

> Note: LangGraph state is persisted in **PostgreSQL** (via `PostgresSaver`). The legacy `checkpoints.sqlite` checkpointer has been retired.

---

## Requirements

* Python **3.10+**
* CUDA GPU (optional but recommended for Grounding DINO)
* Memgraph (Neo4j-compatible) database
* PostgreSQL database
* Redis
* A Google Gemini API key (per user, supplied via the app)

---

## Environment Variables

Create a `.env` file in the backend root.

| Variable | Purpose |
| --- | --- |
| `GOOGLE_API_KEY` | Optional server-level Gemini key. Not set in production; each user provides their own key. |
| `JWT_SECRET` | JWT signing secret |
| `ENCRYPTION_KEY` | Fernet key for encrypting per-user Gemini keys |
| `DAX_INVITE_CODE` | Registration invite code |
| `NEO4J_URI` | Memgraph connection URI (e.g. `bolt://localhost:7687`) |
| `POSTGRES_*` | Postgres connection params |
| `ASSETS_DIR` | PDF / image storage (e.g. `/data/assets`) |
| `BOM_STORAGE_PATH` | BOM JSON output directory (e.g. `/data/bom`) |
| `EXCEL_PATH` | Material weights / pricing Excel file |

> Each user supplies their own Gemini API key, encrypted at rest with Fernet and decrypted per-job at start. There is no server-level fallback configured in production — a job with no user key will fail at the first Gemini call.

---

## Installation

```
git clone https://github.com/sahilkharche999/ArchDoc-AI.git
cd ArchDoc-AI/backend
```

```
python -m venv venv
source venv/bin/activate
```

```
pip install -r requirements.txt
pip install -r requirements-ml.txt
```

---

## Running the Backend

```
uvicorn src.api.main:app --reload
```

Server starts at `http://localhost:8000`, API docs at `http://localhost:8000/docs`.

---

## API Endpoints

All routes require a JWT (`Authorization: Bearer {token}`) except `/health`, `/auth/*`, `/assets/*`, and the SSE `/stream` route (which uses the `job_id` in the URL as a capability token).

### Auth

```
POST /auth/register     # invite-code gated account creation
POST /auth/login        # returns JWT (HS256, 30-day expiry)
GET  /auth/me           # verify token + user info
POST /auth/api-key      # save encrypted Gemini key
```

### Upload

```
POST /upload
```

Form data: `file` (PDF), `start_page`, `end_page`. Trims the PDF to the range and creates a job record.

### Jobs

```
POST /jobs/start            # decrypt key, acquire Redis lock, launch pipeline
GET  /jobs/{id}/stream      # Server-Sent Events progress stream
POST /jobs/{id}/hitl        # submit HITL corrections (resume pipeline)
GET  /jobs/{id}/result      # final BOM JSON
PATCH /jobs/{id}/bom        # save edited BOM back to server
```

### Projects

```
GET    /projects/           # list jobs
GET    /projects/{id}       # single project status/metadata
PUT    /projects/{id}       # rename
DELETE /projects/{id}       # delete + trigger cleanup
POST   /pdf/fix             # apply page rotation fix
```

---

## AI Workflow

A compiled LangGraph `StateGraph` with five nodes and two conditional self-loops. State is persisted in Postgres via `PostgresSaver`, enabling pause/resume across HITL interrupts.

```
START → classify → process_text → process_plans ⟲ → process_details ⟲ → agent_4_merger → END
```

| Node | Agent | Type | HITL |
| --- | --- | --- | --- |
| classify | Agent 0 | Single-pass | Yes — page classification |
| process_text | Agent 1 | Single-pass | No |
| process_plans | Agent 2 | Loop (floor pages) | Yes — bbox review |
| process_details | Agent 3 | Loop (section pages) | Yes — bbox review |
| agent_4_merger | Agent 4 | Single-pass | No |

> `remaining_pages` and `remaining_section_pages` are intentionally **not** initialized in the initial state. The loop nodes populate them lazily on first entry via an absence-based guard (`if 'remaining_pages' not in state`). Initializing them to `[]` would cause the guard never to fire, and no pages would process.

### Agent 0 — Page Classifier
Renders each page to a 300 DPI PNG and asks Gemini Flash to classify it as `text`, `floor`, or `section`. Results cached in `page_map_cache.json` (per job). A HITL interrupt lets the user correct misclassifications.

### Agent 1 — Text Rule Extractor
Runs MinerU on `text` pages, classifies each table image as `Schedule` or `Keyed_Notes`, and stores rows in Memgraph as `Definition` nodes with embeddings. Freeform prose is intentionally ignored. Definitions use `sheet_number = job_id` (project-global). Agent 1 never mutates pipeline state.

### Agent 2 — Floor Plan Processor
Per floor page: render PNG + PDF, extract sheet number (Gemini Pro), run MinerU, present annotated bboxes for HITL, apply corrections to the MinerU JSON (`_hitl.json`), map crops to detail groups, then classify each crop:
* **Schedule** → written to Memgraph
* **Plan_View** → accumulated in `floor_plan_images` (consumed by Agent 4)
* **Detail** → accumulated in `detected_details` (consumed by Agent 3's final pass)

The queue `pop(0)` happens at the **end** of the node, so a crash never silently drops a page. `plan_box_indices` lets the user force a crop to be treated as a plan.

### Agent 3 — Section Detail Processor
Processes section pages in a loop (MinerU → HITL → group → `extract_single_detail` → `add_detail_bom`). On the **final** pass it also resolves Agent 2's `detected_details`: classifies each, runs `_process_dependent_details` (extracts text references, resolves them against the graph, inherits sub-BOMs, merges fabrication counts), and `_process_plan_like_details`. With zero section pages, the loop body is skipped but the final-pass logic still runs, so the pipeline always reaches Agent 4.

### Agent 4 — BOM Merger
Runs Grounding DINO on each floor plan to detect callout symbols, reads each via Gemini Flash OCR, dedups + counts occurrences, resolves each against the graph, then passes enriched symbols + plan context to Gemini Pro for final BOM extraction. Per-image failures are isolated — one bad plan does not discard other plans' results. Output is written to `{BOM_STORAGE_PATH}/{job_id}.json` with keys `bom`, `unreferenced_details`, and an optional `message`.

---

## HITL Mechanism

Implemented with LangGraph's `interrupt()`. When a node calls `interrupt(review_data)`, the pipeline pauses and the checkpoint captures full state in Postgres. Review data is stored in Redis (`hitl:{job_id}`) and a `waiting_for_user` event is relayed to the frontend via SSE. On `POST /jobs/{id}/hitl`, the pipeline resumes via `Command(resume=payload)`; LangGraph re-runs the interrupted node from the top, but `interrupt()` now returns the payload instead of pausing, and the node applies the corrections.

| Type | Node | User reviews | Can change |
| --- | --- | --- | --- |
| classify_review | Agent 0 | Page type map | Correct misclassified pages |
| bbox_review | Agent 2 | MinerU bboxes (floor) | Add crops, delete boxes, flag as plan |
| section_review | Agent 3 | MinerU bboxes (section) | Add crops, delete boxes |

---

## Data Layer

**PostgreSQL** — LangGraph checkpoints (via `PostgresSaver`) plus app tables: `jobs`, `jobs_progress`, `users` (with encrypted `gemini_api_key`).

**Memgraph** — labeled property graph with 3072-dim embeddings (`gemini-embedding-001`):
`Project` → `Sheet` → `Definition:Schedule` / `Definition:Detail` / `Definition:Rule` (vestigial), with `Component` nodes linked from details via `CONTAINS` edges.

**Redis** — `hitl:{job_id}` (pending review), `dax:processing_lock` (single-job mutex), `sheet_prefix:{job_id}`, `gemini_key:{job_id}`, and the `{job_id}` pub-sub channel for progress events.

Only one job runs at a time, enforced by `dax:processing_lock`. New jobs queue as `pending`; when the active job finishes, `start_next_pending_job()` starts the oldest pending one.

---

## Authentication & Security

* Registration is invite-code gated (`DAX_INVITE_CODE`).
* Passwords hashed with bcrypt.
* JWTs are HS256, 30-day expiry, verified in FastAPI middleware for all non-public routes.
* Gemini API keys are encrypted at rest with Fernet (`ENCRYPTION_KEY`), decrypted per-request at job start.
* The `/stream` route bypasses JWT middleware (SSE cannot send Authorization headers); the `job_id` acts as a capability token.

---

## Deployment

The backend runs as a systemd service (`dax-backend`) on EC2. Postgres, Redis, and Memgraph run as Docker containers with persistent volumes under `/data/`.

| Service | Host/Docker | Port |
| --- | --- | --- |
| FastAPI (uvicorn) | Host — systemd | 127.0.0.1:8000 |
| Nginx | Host — systemd | 0.0.0.0:80 |
| Postgres | Docker (postgres_db) | 5433 |
| Redis | Docker (dax-redis) | 127.0.0.1:6379 |
| Memgraph | Docker (dax-memgraph) | 7687 |

### Day-to-day deploy

```bash
# Pull latest code
cd /home/ubuntu/ArchDoc-AI && git pull origin main

# Sync backend (code changes only)
rsync -a --delete --exclude='venv' --exclude='.env' \
  /home/ubuntu/ArchDoc-AI/backend/ /opt/dax/backend/
systemctl restart dax-backend

# Full redeploy (new packages or fresh EC2)
bash scripts/deploy_ec2_t3large.sh
```

---

## Output Example

```
Beam W24X62
Total Length: 240 ft
Quantity: 10
Total Weight: 14,880 lbs
Total Cost: $22,320
```