# API Reference

Base URL: `http://localhost:8000/api`

Interactive docs: `http://localhost:8000/api/docs` (Swagger) | `http://localhost:8000/api/redoc` (ReDoc)

---

## Health Check

### `GET /api/health`

Returns server status and index info.

**Response** `200 OK`

```json
{
  "status": "ok",
  "index_loaded": true,
  "n_chunks": 868,
  "version": "2.0.0"
}
```

| Field          | Type    | Description                                |
| -------------- | ------- | ------------------------------------------ |
| `status`       | string  | Always `"ok"`                              |
| `index_loaded` | boolean | Whether FAISS index is loaded and ready    |
| `n_chunks`     | integer | Number of indexed chunks (0 if not loaded) |
| `version`      | string  | Application version                        |

---

## Question Answering

### `POST /api/query`

Submit a question against the indexed document. Returns a complete answer with citations.

**Request Body** `application/json`

```json
{
  "question": "What is systems engineering?",
  "top_k": 10,
  "alpha": 0.7,
  "use_parent": true
}
```

| Field        | Type    | Required | Default | Constraints  | Description                                           |
| ------------ | ------- | -------- | ------- | ------------ | ----------------------------------------------------- |
| `question`   | string  | Yes      | —       | 1–2000 chars | Natural language question                             |
| `top_k`      | integer | No       | 10      | 1–50         | Number of source chunks to retrieve                   |
| `alpha`      | float   | No       | 0.7     | 0.0–1.0      | Dense/BM25 weight (1.0 = pure dense, 0.0 = pure BM25) |
| `use_parent` | boolean | No       | true    | —            | Resolve child → parent chunks for richer LLM context  |

**Response** `200 OK`

```json
{
  "question": "What is systems engineering?",
  "answer": "Systems engineering is an interdisciplinary approach that enables the realization of successful systems [Source 1]. It encompasses...",
  "citations": [
    {
      "index": 1,
      "citation": "Page 5, §2.3 Systems Engineering Process, Text",
      "score": 0.8234,
      "page": 5,
      "pdf_name": "NASA-SP-2016-6105-Rev2.pdf"
    },
    {
      "index": 2,
      "citation": "Page 12, §3.1 Requirements, Table, Table 3.1-1",
      "score": 0.7621,
      "page": 12,
      "pdf_name": "NASA-SP-2016-6105-Rev2.pdf"
    }
  ],
  "scores": [0.8234, 0.7621, 0.7103]
}
```

| Field                  | Type                | Description                                       |
| ---------------------- | ------------------- | ------------------------------------------------- |
| `question`             | string              | Echo of the input question                        |
| `answer`               | string              | LLM-generated answer with `[Source N]` references |
| `citations`            | array               | Source chunks used for the answer                 |
| `citations[].index`    | integer             | Source number (matches `[Source N]` in answer)    |
| `citations[].citation` | string              | Citation label: page, section, type, caption      |
| `citations[].score`    | float               | Hybrid retrieval score (0–1)                      |
| `citations[].page`     | integer/string/null | PDF page number                                   |
| `citations[].pdf_name` | string/null         | Original PDF filename                             |
| `scores`               | array[float]        | All retrieval scores (top-k)                      |

**Error** `503 Service Unavailable`

```json
{ "detail": "Index not loaded. Ingest a PDF first." }
```

---

### `POST /api/query/stream`

Same parameters as `/api/query`, but returns a **Server-Sent Events** stream.

**Request Body** — identical to `POST /api/query`

**Response** `200 OK` — `text/event-stream`

```
data: {"type": "token", "content": "Systems"}

data: {"type": "token", "content": " engineering"}

data: {"type": "token", "content": " is an"}

...

data: {"type": "citations", "citations": [{"index": 1, "citation": "Page 5, §2.3", "score": 0.82, "page": 5, "pdf_name": "handbook.pdf"}]}

data: {"type": "done"}
```

**Event Types:**

| Type        | Payload                                     | Description                                  |
| ----------- | ------------------------------------------- | -------------------------------------------- |
| `token`     | `{"type": "token", "content": "..."}`       | Incremental text token from LLM              |
| `citations` | `{"type": "citations", "citations": [...]}` | Citation array (sent once, after all tokens) |
| `done`      | `{"type": "done"}`                          | Stream complete signal                       |

---

## PDF Ingestion

### `POST /api/ingest`

Upload a PDF file for background processing. Returns immediately with a task ID for polling.

**Request** `multipart/form-data`

| Field  | Type | Required | Description                 |
| ------ | ---- | -------- | --------------------------- |
| `file` | file | Yes      | PDF file to ingest (`.pdf`) |

**Response** `202 Accepted`

```json
{
  "task_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "status": "queued"
}
```

---

### `GET /api/ingest/{task_id}`

Poll the status of a running ingestion task. Includes step-level progress for UI display.

**Response** `200 OK`

```json
{
  "task_id": "a1b2c3d4-...",
  "status": "running",
  "error": null,
  "current_step": 1,
  "total_steps": 5,
  "step_message": "Parsing PDF with AdvancedPDFParser",
  "step_detail": "Extracting page 45 of 297"
}
```

| Field          | Type        | Description                                         |
| -------------- | ----------- | --------------------------------------------------- |
| `task_id`      | string      | Task identifier                                     |
| `status`       | string      | `queued` → `running` → `completed` or `failed`      |
| `error`        | string/null | Error message (only if `failed`)                    |
| `current_step` | integer     | Current pipeline step (1–5)                         |
| `total_steps`  | integer     | Always 5                                            |
| `step_message` | string      | Current step name (e.g., "Building embeddings")     |
| `step_detail`  | string      | Sub-step detail (e.g., "Extracting page 45 of 297") |

**Pipeline Steps:**

| Step | `step_message` example                               | `step_detail` examples                              |
| ---- | ---------------------------------------------------- | --------------------------------------------------- |
| 1    | Parsing PDF with AdvancedPDFParser                   | Scanning page 45 of 297, Extracting page 100 of 297 |
| 2    | Building embeddings (openai, text-embedding-3-small) | Embedding 868 chunks...                             |
| 3    | Building FAISS IndexFlatIP                           | —                                                   |
| 4    | Building BM25 sparse index                           | —                                                   |
| 5    | Saving artefacts to data/                            | —                                                   |

---

### `GET /api/ingest/{task_id}/result`

Get the result of a completed ingestion.

**Response** `200 OK` (only when status = completed)

```json
{
  "task_id": "a1b2c3d4-...",
  "status": "completed",
  "n_chunks": 868,
  "n_parents": 285,
  "n_tables": 26,
  "n_images_described": 42,
  "embedding_dim": 1536
}
```

| Field                | Type    | Description                    |
| -------------------- | ------- | ------------------------------ |
| `n_chunks`           | integer | Total child chunks indexed     |
| `n_parents`          | integer | Parent chunks created          |
| `n_tables`           | integer | Tables detected and parsed     |
| `n_images_described` | integer | Images described by Vision LLM |
| `embedding_dim`      | integer | Embedding vector dimensions    |

**Error** `202 Accepted` — task still in progress
**Error** `404 Not Found` — task ID not found

---

## PDF Serving

### `GET /api/pdf/{filename}`

Serve an uploaded PDF file for browser viewing. Used by the frontend for citation links.

**Response** `200 OK` — `application/pdf` (inline disposition)

Opens in browser PDF viewer. Supports `#page=N` fragment for direct page navigation.

**Example:** `GET /api/pdf/NASA-SP-2016-6105-Rev2.pdf#page=5`

**Error** `404 Not Found` — file not found in `data/uploads/`

---

## Settings

### `GET /api/settings`

Read current configuration (secrets redacted).

**Response** `200 OK`

```json
{
  "EMBEDDING_PROVIDER": "openai",
  "LLM_PROVIDER": "groq",
  "VISION_PROVIDER": "openai",
  "GROQ_LLM_MODEL": "llama-3.3-70b-versatile",
  "OPENAI_EMBEDDING_MODEL": "text-embedding-3-small",
  "HYBRID_ALPHA": 0.7,
  "RETRIEVAL_TOP_K": 10,
  "CHUNK_MAX_TOKENS": 256,
  "has_groq_key": true,
  "has_openai_key": true
}
```

API keys are never returned — only `has_groq_key` and `has_openai_key` booleans.

---

### `PUT /api/settings`

Update configuration. Writes to `.env` file. Model/provider changes require server restart.

**Request Body** `application/json`

```json
{
  "LLM_PROVIDER": "openai",
  "HYBRID_ALPHA": 0.8
}
```

**Response** `200 OK`

```json
{ "status": "ok" }
```

---

## Static SPA

### `GET /` (catch-all)

Serves the React SPA from `frontend/build/`. Any path not matching `/api/*` returns `index.html` for client-side routing.

---

## Running

```powershell
# Development (two terminals)
uvicorn backend.server.app:app --reload --port 8000    # backend
cd frontend; npm run dev                                 # frontend (proxies /api → :8000)

# Production (single process)
cd frontend; npm run build; cd ..
uvicorn backend.server.app:app --host 0.0.0.0 --port 8000
# SPA served at http://localhost:8000
```
