# Architecture Overview

Production-grade RAG (Retrieval-Augmented Generation) system for PDF technical manuals. Two main pipelines — **Ingestion** (PDF → searchable index) and **QnA** (question → grounded answer) — served by a FastAPI backend with a React frontend.

---

## Project Structure

```
tech-manual-qa/
├── backend/
│   ├── config.py                          # Centralized Settings dataclass (.env)
│   ├── logger.py                          # Structured logging + StepTracker
│   │
│   ├── ingestion/                         # Pipeline 1: PDF → FAISS
│   │   ├── pipeline.py                    # IngestPipeline — 5-step orchestrator
│   │   ├── advanced_parser.py             # AdvancedPDFParser — 8-phase PDF extraction
│   │   ├── parser.py                      # PDFLayoutParser — table detection & stitching
│   │   ├── chunkers/
│   │   │   ├── chunker.py                 # recursive_chunk() — recursive text splitting
│   │   │   └── parent_child_chunker.py    # 2-tier parent (1024 tok) / child (256 tok)
│   │   ├── extractors/
│   │   │   ├── image_extractor.py         # Raster + vector diagram extraction + Vision
│   │   │   ├── caption_extractor.py       # Figure/Table/Exhibit caption regex
│   │   │   ├── section_tracker.py         # Section heading detection + SectionIndex
│   │   │   ├── link_extractor.py          # Hyperlinks + in-text citation patterns
│   │   │   └── ocr.py                     # Tesseract OCR for scanned pages
│   │   └── processors/
│   │       ├── text_cleaner.py            # Unicode normalization, header/footer removal
│   │       ├── quality_scorer.py          # 7-dimension quality scoring + filtering
│   │       ├── deduplicator.py            # MinHash near-dup + exact-dup removal
│   │       └── doc_structure.py           # Page classification (TOC, body, appendix…)
│   │
│   ├── qna/                               # Pipeline 2: Question → Answer
│   │   ├── pipeline.py                    # QnAPipeline — retriever + generator
│   │   ├── retriever.py                   # HybridRetriever — FAISS + BM25 fusion
│   │   └── generator.py                   # AnswerGenerator — citation-aware LLM
│   │
│   ├── server/                            # FastAPI application
│   │   ├── app.py                         # 11 API endpoints + SPA serving
│   │   └── models.py                      # Pydantic request/response schemas
│   │
│   └── utils/
│       ├── embeddings.py                  # Multi-provider embeddings (local / OpenAI)
│       └── llm_client.py                  # LLM provider abstraction (Groq / OpenAI)
│
├── frontend/                              # React 18 SPA (Vite 5)
│   └── src/
│       ├── App.jsx                        # Main layout + routing
│       ├── api.js                         # API client (8 functions, SSE streaming)
│       ├── index.css                      # Full dark-theme stylesheet
│       ├── components/
│       │   ├── ChatView.jsx               # Chat messages + citation cards
│       │   ├── IngestPanel.jsx            # PDF upload + 5-step live stepper
│       │   ├── SettingsPanel.jsx          # Provider/model configuration
│       │   ├── Sidebar.jsx                # Chat history list
│       │   └── ToastContainer.jsx         # Notification toasts
│       └── hooks/
│           ├── useChat.js                 # Chat state + SSE streaming + localStorage
│           └── useToast.js                # Auto-dismiss toast notifications
│
├── data/                                  # Artefacts (gitignored)
│   ├── uploads/                           # Uploaded PDFs
│   ├── faiss.index                        # Dense vector index
│   ├── chunks.pkl                         # Child chunk metadata
│   ├── parent_chunks.pkl                  # Parent chunk metadata
│   └── bm25_corpus.pkl                    # BM25 tokenized corpus
│
├── .env.example                           # Environment variable template
├── requirements.txt                       # Python dependencies
└── README.md
```

---

## High-Level Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                     INGESTION PIPELINE                          │
│                                                                 │
│  PDF ──► AdvancedPDFParser (8 phases) ──► chunks + parents      │
│            │ Text, tables, images, OCR, captions, structure     │
│            │ Quality scoring, deduplication                      │
│            ▼                                                     │
│         Embeddings (OpenAI / local) ──► float32 vectors         │
│            │                                                     │
│            ├──► FAISS IndexFlatIP (dense cosine search)         │
│            ├──► BM25Okapi (sparse keyword search)               │
│            └──► Pickle artefacts to data/                       │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                       QnA PIPELINE                              │
│                                                                 │
│  Question ──► HybridRetriever                                   │
│                ├── Dense: embed query → FAISS top-50            │
│                ├── Sparse: tokenize → BM25 top-50               │
│                ├── Fuse: α·dense + (1-α)·BM25 → top-k          │
│                └── Parent resolution: child → parent chunk      │
│              ──► AnswerGenerator                                 │
│                ├── Build prompt with [Source N] markers          │
│                ├── LLM call (Groq/OpenAI, temp=0)               │
│                └── Stream tokens + citations via SSE             │
│              ──► { answer, citations[], scores[] }              │
└─────────────────────────────────────────────────────────────────┘
```

---

# Pipeline 1: Data Ingestion (PDF → Searchable Index)

The ingestion pipeline transforms a raw PDF into a searchable knowledge base in **5 orchestrated steps**, with the PDF parsing itself comprising **8 internal phases**.

## Step 1: Parse PDF — `AdvancedPDFParser.ingest()`

The parser is the most complex component. It processes a PDF through 8 ordered phases:

### Phase 0: Pre-scan

Extracts raw text from every page using PyMuPDF (`fitz`). This fast first pass provides text for structure analysis without heavy processing.

```
for each page in PDF:
    raw_text[page] = fitz_page.get_text("text")
    report progress: "Scanning page X of Y"
```

### Phase 1: Document Structure Analysis

Classifies every page into one of 11 categories to decide which pages contain actual content:

| Category       | Detection                                 | Include in chunks? |
| -------------- | ----------------------------------------- | ------------------ |
| `cover`        | First page, short text                    | No                 |
| `title_page`   | NASA/org keywords on early pages          | No                 |
| `toc`          | "TABLE OF CONTENTS", dot leaders          | No                 |
| `front_matter` | Preface, Foreword, List of Figures/Tables | No                 |
| `body`         | Main content (default)                    | **Yes**            |
| `appendix`     | "Appendix A", "APPENDIX" patterns         | **Yes**            |
| `glossary`     | "Glossary", "Definitions"                 | **Yes**            |
| `index_page`   | "Index", typical keyword-page format      | No                 |
| `references`   | "References", "Bibliography"              | **Yes**            |
| `blank`        | < 50 characters                           | No                 |
| `unknown`      | Unclassified                              | **Yes**            |

**Implementation:** `doc_structure.py` → `analyze_document_structure()` uses 18 regex patterns, detects `body_start` and `appendix_start` page numbers, and marks non-content pages as `is_content=False`.

### Phase 2: Header/Footer Detection

Identifies repeated text appearing on ≥30% of pages:

1. Extract top 3 and bottom 3 lines from each page
2. Normalize (strip digits, collapse whitespace)
3. Count occurrences across all pages
4. Patterns exceeding threshold → removed during text cleaning

Examples caught: "NASA Systems Engineering Handbook", "Page X of Y", "Rev 2", copyright footers.

### Phase 3: Per-Page Extraction

The heaviest phase — processes each **content** page through multiple extractors:

```
for each content page:
    1. Extract native text (PyMuPDF)
       → Clean: remove headers/footers, normalize Unicode, rejoin hyphenated words
       → Strip page numbers, watermarks

    2. Detect tables (pdfplumber)
       → PDFLayoutParser: find_tables() + multi-page stitching
       → Column matching: snap x-coords to 10px tolerance
       → Stitch if: adjacent pages + ≥60% column overlap
       → Extract text between table bounding boxes

    3. Extract images (PyMuPDF)
       → Skip decorative: width or height < 100px
       → Max 5 images per page
       → Detect vector diagrams: ≥5 PDF path operations + no raster → render at 144 DPI
       → Convert all to PNG, base64 encode

    4. OCR scanned regions (Tesseract)
       → Heuristic: < 20 native chars AND ≥ 1 embedded image → scanned page
       → Render at 300 DPI → pytesseract.image_to_string()

    5. Extract links & references
       → PyMuPDF get_links(): URIs, internal goto, named destinations
       → Regex: [1], [1-5], (Smith, 2020), footnote patterns

    6. Track section headings
       → Numbered: "1.2.3 Title"
       → Chapter: "Chapter 3 – Title"
       → ALL-CAPS: ≥4 words, ≤80 chars
       → Each chunk tagged with most-recent heading

    7. Detect captions
       → Regex: "Figure 2.3-1: Title", "Table 4.2: Title", "Exhibit A-1: Title"

    report progress: "Extracting page X of Y"
```

### Phase 4: Vision Descriptions (Batched)

All extracted images are sent to a Vision LLM for natural-language descriptions:

- **Model:** configurable — OpenAI `gpt-4o-mini` or Groq `llama-4-scout-17b`
- **Prompt:** "Describe this technical diagram, workflow, chart, or figure in detail…"
- **Retry:** 3 attempts with exponential backoff (2× base delay)
- **Rate limiting:** configurable `vision_delay` between API calls (default 1.0s)
- The descriptions become searchable text chunks, making diagrams retrievable by natural language

### Phase 5: Caption Linking

Associates detected captions with their corresponding images and tables:

- **Images:** Match `Figure X` caption to the nearest image on the same or adjacent page
- **Tables:** Match `Table X` caption to the nearest `TableObject` by page proximity
- Linked captions appear in chunk metadata (`caption_label`: "Figure 2.3-1: Systems Engineering Process")

### Phase 6: Chunk Assembly

Consolidates all extracted content into a flat list of chunk dictionaries:

**Text chunks** — Recursive splitting with overlap:

```
recursive_chunk(page_text, max_tokens=256, overlap_tokens=30)
Separator cascade: "\n\n" → "\n" → ". " → " " → "" (hard split)
```

**Parent-child hierarchy** (if enabled):

```
Parent: ~1024 tokens (large context for LLM)
  └── Child: ~256 tokens (precise embeddings for retrieval)
      linked via parent_id → chunk_id (UUID[:12])
```

**Rationale:** Small child embeddings avoid dilution and match precisely. Parent chunks provide rich surrounding context for LLM answer generation.

**Table chunks** — One chunk per table row, pipe-delimited columns.

**Image chunks** — Full vision description as content, tagged `type: "image"`.

**OCR chunks** — Recognized text tagged `type: "ocr"`, `source: "ocr"`.

### Phase 7: Quality Scoring & Filtering

Every chunk is scored across 7 dimensions (each 0.0–1.0):

| Dimension       | Weight | What it measures                                                  |
| --------------- | ------ | ----------------------------------------------------------------- |
| **Length**      | 15%    | Optimal at 50–500 tokens; penalizes very short/long               |
| **Entropy**     | 20%    | Character distribution (Shannon entropy); filters repetitive text |
| **Alpha ratio** | 15%    | Fraction of alphanumeric + whitespace; catches garbled content    |
| **Uniqueness**  | 15%    | Unique words / total words; filters boilerplate repetition        |
| **Coherence**   | 10%    | Average word length 2.5–12.0 chars; catches OCR garbage           |
| **Boilerplate** | 25%    | NOT matching patterns: page numbers, TOC lines, watermarks        |

**Composite score:**

$$Q = 0.15L + 0.20E + 0.15A + 0.15U + 0.10C + 0.25B$$

- **Threshold:** $Q \geq 0.30$ → kept; below → rejected
- **Special:** Tables and images get a floor of $Q_{min} = 0.60$ (always valuable)

### Phase 8: Deduplication

Two-pass duplicate removal:

**Pass 1 — Exact dedup:** O(n) single-pass exact string matching.

**Pass 2 — Near-dedup (MinHash):**

1. Tokenize each chunk → lowercase, alpha-only words
2. Create 3-word shingles (sliding window)
3. Compute 128 independent MinHash signatures per chunk
4. Estimate Jaccard similarity: `matching_hashes / 128`
5. If similarity ≥ 0.85 → duplicate → keep the "richer" chunk (more metadata, longer content, has caption/links)

$$\text{Jaccard}(A, B) \approx \frac{|\{i : h_i(A) = h_i(B)\}|}{128}$$

### Output of Step 1

```python
{
    "chunks":        [...],    # Child chunks ready for embedding
    "parent_chunks": [...],    # Parent chunks for LLM context
    "tables":        [...],    # Raw TableObject data
    "stats":         ParseStats(total_pages=297, total_chunks=868, ...),
    "captions":      [...],    # Caption objects
    "doc_structure": {...},    # DocumentStructure
    "section_index": [...]     # Heading list
}
```

---

## Step 2: Build Embeddings

```python
embedder = Embeddings(provider=EMBEDDING_PROVIDER)
texts = [chunk["content"] for chunk in chunks]
embeddings = embedder.embed(texts)  # → np.ndarray(N, D), float32
```

**Providers:**

| Provider | Model                    | Dimensions | Notes                        |
| -------- | ------------------------ | ---------- | ---------------------------- |
| `openai` | `text-embedding-3-small` | 1536       | API-based, fast, low cost    |
| `local`  | `BAAI/bge-large-en-v1.5` | 1024       | HuggingFace, runs on CPU/GPU |

**Post-processing:** L2-normalize so inner product = cosine similarity:

```python
norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
embeddings /= norms
```

---

## Step 3: Build FAISS Index

```python
index = faiss.IndexFlatIP(dimension)   # Inner product = cosine (after L2-norm)
index.add(embeddings)                  # O(N) insertion, O(N) search
```

- **Index type:** `IndexFlatIP` — exact nearest-neighbour search via inner product
- **Why not HNSW/IVF?** At < 10K chunks, flat search is fast enough and guarantees exact results

---

## Step 4: Build BM25 Sparse Index

```python
corpus = [tokenize_bm25(chunk["content"]) for chunk in chunks]
bm25_index = BM25Okapi(corpus)
```

**Tokenizer:** lowercase → regex word extraction → remove 45+ English stopwords → alpha-only filter.

**Why BM25 alongside dense search?** Dense embeddings capture semantic similarity but can miss exact keyword matches. BM25 catches acronyms (e.g., "SE", "MBSE"), specific terms, and table content that embeddings may not represent well.

---

## Step 5: Save Artefacts

Four files persisted to `data/`:

| File                | Format        | Contents                       |
| ------------------- | ------------- | ------------------------------ |
| `faiss.index`       | FAISS binary  | Dense vector index             |
| `chunks.pkl`        | Python pickle | All child chunks with metadata |
| `parent_chunks.pkl` | Python pickle | Parent chunks                  |
| `bm25_corpus.pkl`   | Python pickle | Tokenized BM25 corpus          |

---

## Chunk Data Schema

Every chunk is a dictionary:

```python
{
    "content": str,                          # The actual text
    "type": "text" | "table" | "image" | "ocr",
    "page": int,                             # 1-based page number
    "section_heading": str,                  # Most recent section heading
    "chunk_id": str,                         # UUID[:12] (parent chunks only)
    "parent_id": str,                        # Links to parent (children only)
    "is_parent": bool,
    "metadata": {
        "pages": [int],                      # Pages spanned
        "start_char": int,
        "end_char": int,
        "source": "native" | "ocr" | "vision_llm",
        "chunk_level": "parent" | "child",
        "quality_scores": {
            "length_score": float,
            "entropy_score": float,
            "alpha_ratio": float,
            "uniqueness_score": float,
            "coherence_score": float,
            "boilerplate_score": float,
            "quality_score": float           # Composite
        },
        "caption": str,
        "caption_label": str,                # "Figure 2.3: Title"
        "table_id": str,
        "header": [str],                     # Table column headers
        "row": int,
        "image_b64": str,                    # Base64 image data
        "bbox": tuple,
        "is_vector": bool,
        "links": [str]
    }
}
```

---

# Pipeline 2: QnA (Question → Answer)

## Step 1: Hybrid Retrieval — `HybridRetriever.search()`

### 1a. Dense Search (FAISS)

```python
query_embedding = embedder.embed([question])       # (1, D)
query_embedding /= np.linalg.norm(query_embedding)  # L2-normalize
scores, indices = index.search(query_embedding, SEARCH_K)  # top-50
```

Returns cosine similarity scores in range [0, 1].

### 1b. Sparse Search (BM25)

```python
query_tokens = tokenize(question)
bm25_scores = bm25_index.get_scores(query_tokens)  # array[N]
bm25_scores /= max(bm25_scores)                    # normalize to [0, 1]
```

### 1c. Score Fusion

$$\text{fused}[i] = \alpha \times \text{dense}[i] + (1 - \alpha) \times \text{bm25}[i]$$

Default $\alpha = 0.7$ → 70% dense (semantic), 30% BM25 (keyword).

```python
# Combine candidates from both searches
for each chunk index i in (FAISS top-50 ∪ BM25 top-50):
    fused[i] = alpha * dense_score[i] + (1-alpha) * bm25_score[i]

# Rank by fused score, return top-k (default k=10)
results = sorted(fused.items(), key=score, reverse=True)[:k]
```

### 1d. Parent Resolution

For each retrieved child chunk that has a `parent_id`:

```python
parent = parent_lookup[child.parent_id]    # O(1) dict lookup
result["parent_content"] = parent["content"]  # ~1024 tokens of context
```

**Why?** The child chunk (~256 tokens) matched the query precisely, but the parent chunk (~1024 tokens) provides surrounding context for a better answer.

---

## Step 2: Answer Generation — `AnswerGenerator.generate()`

### 2a. Context Building

For each of the top-k results (up to 10):

```
[Source 1: Page 5, §2.3 Systems Engineering Process, Text]
{parent_content or chunk content}

---

[Source 2: Page 12, §3.1 Requirements, Table, Table 3.1-1: Requirements Matrix]
{chunk content}
```

Each source label includes: page number, section heading, chunk type, caption (if any).

### 2b. LLM Prompt

```
System: "You are an expert technical assistant. Answer accurately using
         ONLY the provided context. Cite sources using [Source N] format.
         If insufficient info, say so clearly."

User:   "Context (retrieved from document):

         [Source 1: Page 5, §2.3, Text]
         {content}
         ---
         [Source 2: Page 12, §3.1, Table]
         {content}
         ---

         Question: {user's question}

         Instructions: Answer using context above. Cite all sources used."
```

### 2c. LLM Call

| Setting     | Value                                                  |
| ----------- | ------------------------------------------------------ |
| Temperature | 0.0 (deterministic)                                    |
| Max tokens  | 1024                                                   |
| Model       | Groq `llama-3.3-70b-versatile` or OpenAI `gpt-4o-mini` |
| Streaming   | Yes (SSE token events)                                 |

### 2d. Streaming Response (SSE)

```
data: {"type": "token", "content": "The systems"}
data: {"type": "token", "content": " engineering process"}
...
data: {"type": "citations", "citations": [{index:1, citation:"Page 5, §2.3", score:0.82, page:5, pdf_name:"handbook.pdf"}]}
data: {"type": "done"}
```

### 2e. Citation Object

```python
{
    "index": 1,
    "citation": "Page 5, §2.3 Systems Engineering, Text",
    "score": 0.8234,
    "page": 5,
    "pdf_name": "NASA-SP-2016-6105-Rev2.pdf"
}
```

---

# Configuration Reference

All settings live in `backend/config.py` as a frozen `@dataclass`, loaded from `.env`:

## API Keys

| Variable         | Required for                                       |
| ---------------- | -------------------------------------------------- |
| `GROQ_API_KEY`   | LLM (Groq), Vision (Groq)                          |
| `OPENAI_API_KEY` | Embeddings (OpenAI), LLM (OpenAI), Vision (OpenAI) |

## Provider Selection

| Variable             | Options           | Default  | Purpose                  |
| -------------------- | ----------------- | -------- | ------------------------ |
| `EMBEDDING_PROVIDER` | `openai`, `local` | `openai` | Embedding model provider |
| `LLM_PROVIDER`       | `groq`, `openai`  | `groq`   | Answer generation LLM    |
| `VISION_PROVIDER`    | `groq`, `openai`  | `openai` | Image description model  |

## Model Overrides

| Variable                 | Default                                     |
| ------------------------ | ------------------------------------------- |
| `GROQ_LLM_MODEL`         | `llama-3.3-70b-versatile`                   |
| `GROQ_VISION_MODEL`      | `meta-llama/llama-4-scout-17b-16e-instruct` |
| `OPENAI_LLM_MODEL`       | `gpt-4o-mini`                               |
| `OPENAI_VISION_MODEL`    | `gpt-4o-mini`                               |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small`                    |
| `OPENAI_EMBEDDING_DIM`   | `1536`                                      |
| `EMBEDDING_MODEL`        | `BAAI/bge-large-en-v1.5`                    |

## Ingestion Settings

| Variable                  | Default | Purpose                                      |
| ------------------------- | ------- | -------------------------------------------- |
| `CHUNK_MAX_TOKENS`        | 256     | Child chunk size                             |
| `CHUNK_OVERLAP_TOKENS`    | 30      | Child chunk overlap                          |
| `PARENT_MAX_TOKENS`       | 1024    | Parent chunk size                            |
| `PARENT_OVERLAP_TOKENS`   | 100     | Parent chunk overlap                         |
| `ENABLE_OCR`              | `True`  | Tesseract OCR for scanned pages              |
| `ENABLE_VISION`           | `True`  | Vision LLM for image descriptions            |
| `ENABLE_PARENT_CHILD`     | `True`  | 2-tier chunk hierarchy                       |
| `ENABLE_QUALITY_FILTER`   | `True`  | Quality scoring & filtering                  |
| `ENABLE_DEDUP`            | `True`  | MinHash + exact deduplication                |
| `ENABLE_TEXT_CLEANING`    | `True`  | Header/footer removal, Unicode normalization |
| `ENABLE_CAPTIONS`         | `True`  | Figure/Table caption detection               |
| `ENABLE_DOC_STRUCTURE`    | `True`  | Page classification                          |
| `ENABLE_VECTOR_DETECTION` | `True`  | Detect vector diagrams                       |
| `QUALITY_MIN_SCORE`       | 0.30    | Minimum quality threshold                    |
| `VISION_DELAY`            | 1.0     | Seconds between Vision API calls             |
| `OCR_DPI`                 | 300     | OCR rendering resolution                     |

## Retrieval Settings

| Variable                | Default | Purpose                                          |
| ----------------------- | ------- | ------------------------------------------------ |
| `RETRIEVAL_TOP_K`       | 10      | Final results returned                           |
| `RETRIEVAL_SEARCH_K`    | 50      | Candidate pool size before fusion                |
| `HYBRID_ALPHA`          | 0.7     | Dense weight (1.0 = pure dense, 0.0 = pure BM25) |
| `USE_PARENT_RESOLUTION` | `True`  | Resolve child → parent for richer LLM context    |

---

# Key Design Decisions

1. **Config in one place** — `backend/config.py` is the single source of truth. No hardcoded secrets, paths, or model names anywhere else.

2. **Two-tier chunks** — 256-token children for precise retrieval + 1024-token parents for rich LLM context. Small embeddings avoid semantic dilution; large contexts improve answer quality.

3. **Hybrid search** — 70% dense + 30% BM25 catches both semantic similarity (paraphrased questions) and exact keyword matches (acronyms, table values, specific terms).

4. **Lazy loading** — FAISS index and embedding models load on first query, not at server startup.

5. **SPA served by FastAPI** — React build served via `StaticFiles` mount. In dev, Vite proxy forwards `/api` to the backend. Single process in production.

6. **Step-by-step progress tracking** — `StepTracker` with `on_step` + `on_detail` callbacks enables real-time UI updates during ingestion (page-level granularity).

7. **Multi-provider abstraction** — Unified interfaces for embeddings and LLM allow switching between Groq and OpenAI with a single env var change.
