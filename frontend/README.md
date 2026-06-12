# DAX Estimation Dashboard

A modern **React + TypeScript dashboard** for the DAX AI-powered structural steel estimation system. The frontend provides the interface for uploading drawings, reviewing pages and detections (HITL), tracking live processing status, and viewing and editing the generated Bill of Materials (BOM).

---

## Overview

The DAX dashboard allows users to:

* Register/log in (JWT auth) and save a personal Gemini API key
* Upload PDF drawings, fix page rotation, and trim to a page range
* Review and correct page classifications and detection boxes during processing (Human-in-the-Loop)
* Monitor document processing in real time via Server-Sent Events
* View the extracted BOM across Summary, Detailed BOM, Traceability, and Untracked tabs
* Edit BOM rows inline and save changes back to the server
* Export the BOM to Excel

The UI is designed for **structural steel estimation workflows**.

---

## Tech Stack

| Area | Technology |
| --- | --- |
| Framework | React + TypeScript |
| Build tool | Vite |
| Styling | TailwindCSS + Emotion |
| Excel export | SheetJS (xlsx) |
| HTTP | Native fetch API |
| Real-time | Server-Sent Events (EventSource) |
| Auth | JWT stored in localStorage, verified on load |
| Serving | Nginx (static files built by Vite) |

---

## Project Structure

```
src
 ├── app
 │   ├── App.tsx                      # Root router, defines routes, fetches project list
 │   └── context
 │       └── AuthContext.tsx          # Global auth state: token, user, login/logout
 │
 ├── assets
 │   └── dax_mfg_logo.jpeg
 │
 ├── components
 │   ├── BOMTable.tsx                 # Editable BOM table with inline inputs
 │   ├── DashboardHeader.tsx
 │   ├── EditableCell.tsx
 │   ├── EstimationDashboard.tsx      # BOM display: Summary, Detailed BOM, Traceability, Untracked
 │   ├── ProcessingView.tsx           # SSE listener, HITL canvas UI, progress tracker
 │   ├── QueuedWatcher.tsx            # SSE watcher for queued jobs, detects when processing starts
 │   ├── Sidebar.tsx                  # Project list, per-project SSE listeners, rename/delete
 │   ├── SummaryTab.tsx
 │   ├── TraceabilityView.tsx         # Expandable traceability rows with extraction logic
 │   └── UploadState.tsx              # PDF upload, rotation fix, page range input
 │
 ├── pages
 │   ├── ProjectPage.tsx              # Main project view: processing/completed/failed/queued
 │   ├── LoginPage.tsx                # Login form
 │   ├── RegisterPage.tsx             # Registration form with invite code
 │   └── SettingsPage.tsx             # Gemini API key save
 │
 ├── styles
 │   ├── fonts.css
 │   ├── index.css
 │   ├── tailwind.css
 │   └── theme.css
 │
 ├── types
 │   ├── bom.ts                       # BOMItem
 │   ├── project.ts                   # Project
 │   └── summary.ts                   # SummaryTabProps
 │
 ├── main.tsx
 └── vite-env.d.ts
```

---

## Installation

Clone the repository:

```
git clone https://github.com/sahilkharche999/ArchDoc-AI
cd frontend
```

Install dependencies:

```
npm install
```

---

## Run Development Server

```
npm run dev
```

The app will be available at:

```
http://localhost:5173
```

---

## Build for Production

```
npm run build
```

The output files are generated in:

```
dist/
```

---

## Environment Variables

Create a `.env` file in the project root.

```
VITE_API_URL=http://localhost:8000
```

All API calls use the native fetch API with this base URL.

---

## Authentication

Auth is handled via JWT tokens stored in localStorage under the key `dax_token`.

* On app load, `AuthProvider` checks localStorage for a saved token.
* If found, it calls `GET /auth/me` to verify the token is still valid.
* If valid, user state is hydrated and the app proceeds normally.
* If invalid or expired, the token is cleared and the user is redirected to `/login`.
* All API calls include `Authorization: Bearer {token}` in their headers.
* The SSE stream endpoint bypasses JWT middleware (EventSource cannot send custom headers) — the `job_id` in the URL acts as a capability token.

Each user supplies their own Gemini API key via the Settings page; it is saved encrypted on the backend and used for that user's jobs.

---

## Real-Time Progress (SSE)

`ProcessingView` opens an `EventSource` connection to `GET /jobs/{job_id}/stream`. The backend publishes events to a Redis pub-sub channel and the SSE endpoint relays them to the browser.

| Event | Frontend Action |
| --- | --- |
| `step: classify, status: processing` | Set step 0 as active (spinning) |
| `step: process_text, status: completed` | Mark steps 0–1 done, advance active |
| `step: hitl_review, data: {type: classify_review}` | Show page classification modal |
| `step: hitl_review, data: {type: bbox_review}` | Show canvas bbox review modal |
| `step: agent_4_merger, status: completed` | Close SSE, fetch result, call `onComplete` |
| `status: failed` | Close SSE, call `onFailed` with error message |

On SSE reconnect (after a HITL submit), `connectSSE()` is called again — it closes the old `EventSource` and opens a new one, which handles the resume correctly.

---

## HITL Canvas UI

The bbox review modal in `ProcessingView` uses an HTML5 Canvas overlaid on the annotated page image from the backend.

* **Green boxes** — MinerU auto-detected bboxes from the backend payload (`hitlData.bboxes`).
* **Red boxes** — user-drawn boxes; click and drag on the canvas to draw them.
* **Coordinate system** — display px → image px via a scale factor (`imageNaturalW / canvas.getBoundingClientRect().width`). All boxes are stored and submitted in image-pixel space.
* **X button** (top-right of a box) — deletes the box.
* **P button** (top-left of a box) — flags the crop as a Plan View (floor plan) rather than a detail drawing.
* **On Submit** — surviving green boxes + user-drawn red boxes are combined into `corrected_bboxes`; deleted green box indices are sent as `deleted_mineru_bboxes`; plan-flagged indices are sent as `plan_box_indices`.

---

## State Management

The app uses React local state (`useState`) — no external state library.

| Component | State Owned |
| --- | --- |
| AuthContext | token, user, isLoading (shared globally via context) |
| App.tsx | projects list, search string |
| ProjectPage.tsx | status (loading/processing/completed/failed/queued), bomData, filePath |
| ProcessingView.tsx | SSE connection, completedSteps, hitlData, canvas bboxes |
| EstimationDashboard.tsx | editableBom, projectMeta, zoom, activeTab |
| Sidebar.tsx | projects list (own copy), per-project SSE listeners |

Note: projects are fetched in both `App.tsx` (project grid) and `Sidebar.tsx` (sidebar list). They maintain separate copies and refresh independently.

---

## Backend Integration

Key endpoints called by the frontend:

| Endpoint | Called From | Purpose |
| --- | --- | --- |
| POST /auth/register | RegisterPage | Create account with invite code |
| POST /auth/login | LoginPage | Get JWT token |
| GET /auth/me | AuthContext, LoginPage | Verify token + get user info |
| POST /auth/api-key | SettingsPage | Save encrypted Gemini key |
| POST /upload | UploadState | Upload + trim PDF, create job |
| POST /jobs/start | App.tsx | Start pipeline, acquire Redis lock |
| GET /jobs/{id}/stream | ProcessingView, Sidebar, QueuedWatcher | SSE progress stream |
| GET /jobs/{id}/result | ProcessingView (on complete) | Fetch final BOM JSON |
| POST /jobs/{id}/hitl | ProcessingView | Submit HITL corrections |
| PATCH /jobs/{id}/bom | EstimationDashboard | Save edited BOM back to server |
| GET /projects/ | App.tsx, Sidebar | List all projects |
| GET /projects/{id} | ProjectPage, EstimationDashboard | Get single project status/metadata |
| PUT /projects/{id} | App.tsx, Sidebar | Rename project |
| DELETE /projects/{id} | App.tsx, Sidebar, ProjectPage | Delete project + trigger cleanup |
| POST /pdf/fix | UploadState | Apply page rotation fix to PDF |

---

## Build & Deploy

The frontend is built with Vite and served as static files by Nginx.

* **Build** — `npm run build` produces a `dist/` folder.
* **Deploy** — `dist/` is copied to `/var/www/dax/` on the EC2 host.
* **Nginx** serves the static files and proxies all `/api/` requests to FastAPI at `127.0.0.1:8000`.
* **Routing** — React Router uses browser history mode; Nginx is configured with `try_files $uri $uri/ /index.html` so all routes return `index.html` and React Router handles client-side navigation.