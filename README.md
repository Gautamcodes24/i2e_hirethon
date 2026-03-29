# Tech Manual QA

Production-grade RAG (Retrieval-Augmented Generation) system for answering questions from technical PDF manuals. Features an advanced 8-phase ingestion pipeline, hybrid dense+sparse retrieval, multi-provider LLM support, and a React UI with real-time streaming.

## Features

- **Advanced PDF ingestion** — 8-phase parser: OCR, table detection & multi-page stitching, image/vector diagram extraction, Vision LLM descriptions, caption linking, parent-child chunking, 7-dimension quality scoring, MinHash deduplication
- **Hybrid search** — 70% dense (FAISS cosine) + 30% sparse (BM25) retrieval with configurable alpha, parent-chunk resolution for richer LLM context
- **Citation-aware answers** — LLM generates grounded answers with `[Source N]` references linked to specific PDF pages
- **Multi-provider** — Switch between Groq and OpenAI for embeddings, LLM, and Vision via environment variables
- **React UI** — Dark-themed SPA with SSE streaming, chat history, PDF upload with live 5-step progress stepper, clickable citations that open PDF at the cited page, collapsible sidebar, settings panel
- **FastAPI backend** — 11 API endpoints, background ingestion with progress callbacks, Swagger/ReDoc docs, SPA serving

## Quick Start

### 1. Install dependencies

```powershell
git clone [https://github.com/Gautamcodes24/tech-manual-qa.git](https://github.com/Gautamcodes24/i2e_hirethon.git)
cd tech-manual-qa

python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Start the server

```powershell
uvicorn backend.server.app:app --reload --port 8000
```

Open **http://localhost:8000** in the browser — the frontend is pre-built and served live from the backend.

> **Note:** The PDF (NASA Systems Engineering Handbook) has already been ingested and the data artefacts are included in the repo. Just start the backend and start asking questions — no ingestion step needed for testing.

---

## Data Ingestion

The PDF was ingested using **OpenAI embeddings** (`text-embedding-3-small`) with a personal API key. The resulting artefacts (`faiss.index`, `chunks.pkl`, `parent_chunks.pkl`, `bm25_corpus.pkl`) are already present in `data/`.

If you want to re-ingest or ingest a different PDF, add your own OpenAI key to `.env`:

```
OPENAI_API_KEY=sk-...
```

Then use the **Upload PDF** button in the sidebar, or via API:

---

## Asking Questions

Type a question in the chat and press **Ctrl+Enter** or click Send. Answers stream in token-by-token with expandable citation cards that link directly to the referenced PDF page.

---

## Project Structure

```
tech-manual-qa/
├── backend/
│   ├── config.py              # Centralized Settings (@dataclass, .env)
│   ├── logger.py              # Logging + StepTracker with callbacks
│   ├── ingestion/             # Pipeline 1: PDF → FAISS
│   │   ├── pipeline.py        # 5-step orchestrator
│   │   ├── advanced_parser.py # 8-phase PDF parser
│   │   ├── parser.py          # Table detection & stitching
│   │   ├── chunkers/          # Recursive + parent-child chunking
│   │   ├── extractors/        # Images, captions, OCR, links, sections
│   │   └── processors/        # Quality scoring, dedup, text cleaning
│   ├── qna/                   # Pipeline 2: Question → Answer
│   │   ├── pipeline.py        # Retriever + Generator orchestrator
│   │   ├── retriever.py       # Hybrid FAISS + BM25 fusion
│   │   └── generator.py       # Citation-aware LLM prompting + streaming
│   ├── server/                # FastAPI (11 endpoints)
│   │   ├── app.py             # Routes, CORS, SPA mount
│   │   └── models.py          # Pydantic schemas
│   └── utils/
│       ├── embeddings.py      # Multi-provider (OpenAI / local)
│       └── llm_client.py      # Multi-provider (Groq / OpenAI)
├── frontend/                  # React 18 + Vite 5
│   └── src/
│       ├── App.jsx            # Layout + state management
│       ├── api.js             # API client (8 functions)
│       ├── components/        # ChatView, IngestPanel, Settings, Sidebar
│       └── hooks/             # useChat (SSE), useToast
├── data/                      # Artefacts (gitignored)
├── docs/
│   ├── architecture.md        # Full pipeline details
│   └── api-reference.md       # Endpoint documentation
└── .env.example               # Environment template
```

---

## How It Works

### Ingestion Pipeline (5 Steps)

```
Step 1: Parse PDF (AdvancedPDFParser — 8 phases)
        → Document structure → header/footer removal → per-page extraction
        → OCR → Vision descriptions → caption linking → chunking
        → Quality scoring → deduplication
Step 2: Build embeddings (OpenAI text-embedding-3-small or BGE-large-en-v1.5)
Step 3: Build FAISS index (IndexFlatIP, cosine via inner product)
Step 4: Build BM25 index (BM25Okapi, stopword-filtered tokens)
Step 5: Save artefacts (faiss.index, chunks.pkl, parent_chunks.pkl, bm25_corpus.pkl)
```

### QnA Pipeline

```
Question → HybridRetriever
           ├── Dense: FAISS top-50 (cosine similarity)
           ├── Sparse: BM25 top-50 (keyword matching)
           ├── Fusion: α·dense + (1-α)·BM25 → rank → top-k
           └── Parent resolution: child.parent_id → parent chunk (1024 tok)
         → AnswerGenerator
           ├── Context: [Source N: Page X, §Y, type] + content
           ├── LLM: temperature=0, max_tokens=1024
           └── SSE stream: token events → citations → done
```

---

## Environment Variables

Copy `.env.example` to `.env` and configure:

| Variable                 | Default                   | Description                                  |
| ------------------------ | ------------------------- | -------------------------------------------- |
| `GROQ_API_KEY`           | —                         | Groq API key (for LLM + Vision)              |
| `OPENAI_API_KEY`         | —                         | OpenAI API key (for embeddings, LLM, Vision) |
| `EMBEDDING_PROVIDER`     | `openai`                  | `openai` or `local` (HuggingFace)            |
| `LLM_PROVIDER`           | `groq`                    | `groq` or `openai`                           |
| `VISION_PROVIDER`        | `openai`                  | `groq` or `openai`                           |
| `GROQ_LLM_MODEL`         | `llama-3.3-70b-versatile` | Groq chat model                              |
| `OPENAI_LLM_MODEL`       | `gpt-4o-mini`             | OpenAI chat model                            |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small`  | OpenAI embedding model                       |
| `HYBRID_ALPHA`           | `0.7`                     | Dense vs BM25 weight (0.0–1.0)               |
| `RETRIEVAL_TOP_K`        | `10`                      | Results returned per query                   |
| `CHUNK_MAX_TOKENS`       | `256`                     | Child chunk size                             |
| `PARENT_MAX_TOKENS`      | `1024`                    | Parent chunk size                            |
| `QUALITY_MIN_SCORE`      | `0.30`                    | Minimum chunk quality threshold              |

Full list in [docs/architecture.md](docs/architecture.md#configuration-reference).

## Tech Stack

| Layer             | Technology                                                                  |
| ----------------- | --------------------------------------------------------------------------- |
| **Backend**       | Python 3.10+, FastAPI, uvicorn                                              |
| **Vector Search** | FAISS IndexFlatIP (cosine via inner product)                                |
| **Sparse Search** | rank-bm25 (BM25Okapi)                                                       |
| **Embeddings**    | OpenAI `text-embedding-3-small` (1536d) or `BAAI/bge-large-en-v1.5` (1024d) |
| **LLM**           | Groq `llama-3.3-70b-versatile` or OpenAI `gpt-4o-mini`                      |
| **Vision**        | OpenAI `gpt-4o-mini` or Groq `llama-4-scout-17b`                            |
| **PDF Parsing**   | PyMuPDF (fitz), pdfplumber                                                  |
| **OCR**           | Tesseract 5.x via pytesseract                                               |
| **Frontend**      | React 18, Vite 5, lucide-react, react-markdown                              |

## Documentation

- [Architecture & Pipeline Details](docs/architecture.md) — full data flow, algorithms, chunk schema, configuration reference
- [API Reference](docs/api-reference.md) — all 11 endpoints with request/response examples
