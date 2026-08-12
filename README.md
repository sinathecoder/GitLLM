# Git History RAG System

A complete pipeline to extract git history changes, index them in a vector database, and query them with an LLM using Retrieval-Augmented Generation (RAG).

## Pipeline Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ 1. Extract      │ →  │ 2. Build Index   │ →  │ 3. Query with LLM│ →  │ 4. Get Answers  │
│ git_history_    │    │ build_rag_index  │    │ query_git_rag    │    │ about code      │
│ extractor.py    │    │ .py              │    │ .py              │    │ changes         │
└─────────────────┘    └──────────────────┘    └──────────────────┘    └─────────────────┘
```

## Files

| File | Purpose |
|------|---------|
| `config.py` | Single source of truth for all settings (paths, models, chunking, server) |
| `git_history_extractor.py` | Extracts full git history (with all code diffs) from a local repo to a text file |
| `build_rag_index.py` | Chunks the history file, embeds it, and stores it in ChromaDB |
| `query_git_rag.py` | Retrieves relevant chunks and uses an LLM to answer questions |
| `app.py` | Flask API + web console for testing prompts/answers on the RAG |
| `webui/index.html` | Single-page web UI served at `GET /ui` |
| `requirements.txt` | Python dependencies |

## Configuration

All settings live in one place: **`config.py`**. Edit it to change:

- **Paths** — `CHROMA_DIR` (vector DB), `WEBUI_DIR` (web UI assets)
- **ChromaDB** — `COLLECTION_NAME`, `COLLECTION_DESCRIPTION`, `CHROMA_BATCH_SIZE`
- **Chunking** — `DEFAULT_CHUNK_SIZE`, `DEFAULT_CHUNK_OVERLAP`
- **Gemini (web UI / API)** — `GEMINI_API_KEY`, `GEMINI_MODEL`, `GEMINI_TEMPERATURE`, `GEMINI_MAX_OUTPUT_TOKENS`
- **OpenAI-compatible CLI LLM** — `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL`, `LLM_TEMPERATURE`
- **Retrieval** — `DEFAULT_TOP_K`
- **Extractor** — `HISTORY_OUTPUT_DEFAULT`
- **Web server** — `HOST`, `PORT`, `DEBUG`

Every value can also be overridden with an environment variable of the same name
(e.g. `CHROMA_DIR=./my_db python3 app.py`, `GEMINI_MODEL=gemini-2.0-flash python3 app.py`).
Every script reads its defaults from `config.py`, so a change there applies everywhere
without editing the individual tools.

## Installation

```bash
pip install -r requirements.txt
```

## Step 1: Extract Git History

```bash
# Extract full history with all code changes (diffs)
python git_history_extractor.py /path/to/repo -o git_history.txt

# Metadata only (no diffs)
python git_history_extractor.py /path/to/repo -o git_history.txt --no-diff
```

## Step 2: Build the RAG Index

```bash
# Build vector index from the history file
python build_rag_index.py git_history.txt

# Custom database location
python build_rag_index.py git_history.txt --db ./my_db

# Custom chunking
python build_rag_index.py git_history.txt --chunk-size 2000 --chunk-overlap 300
```

This creates a ChromaDB vector database (default: `./git_history_db`) with:
- Each commit split into per-file chunks
- Metadata (commit hash, subject, file path) attached to every chunk
- Embeddings via ONNX MiniLM-L6-v2 (no PyTorch required)

## Step 3: Query with an LLM

### Option A: OpenAI API

```bash
export OPENAI_API_KEY='your-api-key'

# Single question
python query_git_rag.py "What changes were made to the clustering code?"

# Interactive mode
python query_git_rag.py --interactive

# Custom model
export OPENAI_MODEL='gpt-4o'
python query_git_rag.py "What changed in pom.xml?"
```

### Option B: Local LLM (Ollama)

```bash
# Install Ollama: https://ollama.com
ollama pull llama3

export OPENAI_BASE_URL='http://localhost:11434/v1'
export OPENAI_MODEL='llama3'

python query_git_rag.py "What changes were made to the clustering code?"
```

### Option C: Any OpenAI-compatible API

```bash
export OPENAI_BASE_URL='https://your-api-endpoint/v1'
export OPENAI_API_KEY='your-key'
export OPENAI_MODEL='your-model'

python query_git_rag.py "Your question here"
```

## Web UI — Test Prompts & Answers

The included Flask server ships a browser console for testing the RAG end-to-end:

```bash
python app.py
# open http://localhost:5000/ui
```

The console has four tabs:

- **Ask** — enter a prompt; the RAG retrieves context and the model answers. Each answer
  lists the exact chunks used as sources (commit, file, similarity distance) with
  expandable full text. Top-K, model, source citations, and a raw-JSON toggle are available.
- **Retrieval** — run the vector retrieval step *without* calling the LLM (`/api/retrieve`)
  to inspect exactly which chunks would be pulled for a prompt and how they rank. Great for
  tuning a prompt or top-K before spending a model call.
- **Ingest** — paste new text (or a full git-history report) to chunk, embed, and store it.
- **Stats** — collection totals and configuration.

Useful API endpoints:

| Endpoint | Purpose |
|----------|---------|
| `GET /ui` | Browser console (this UI) |
| `POST /api/ingest` | Send text to chunk and store in ChromaDB |
| `POST /api/query` | Ask a question — RAG retrieves context and the model answers |
| `POST /api/retrieve` | Retrieval-only preview of the top-K chunks (no LLM call) |
| `GET /api/stats` | View index statistics |

Set `CHROMA_DIR` to point at a different collection, e.g. for the CLI-built index:
`CHROMA_DIR=./git_history_db python app.py`.

## Example Questions

- "What changes were made to the clustering code?"
- "Which files were modified in the last commit?"
- "What dependencies were added or removed?"
- "How did the pom.xml change over time?"
- "What was the initial commit about?"

## How It Works

1. **Extraction**: `git log --all --patch` captures the complete history including exact line-level diffs for every file in every commit.

2. **Chunking**: Each commit is split into per-file diff sections. If a section is too large, it's further split into overlapping chunks. Every chunk carries metadata (commit hash, subject, file path).

3. **Embedding**: Chunks are embedded using the ONNX MiniLM-L6-v2 model (384-dim vectors) and stored in ChromaDB.

4. **Retrieval**: When you ask a question, the system finds the most semantically similar chunks using vector similarity search.

5. **Generation**: The retrieved chunks are injected into a prompt with your question, and the LLM generates an answer grounded in the actual code changes.

## Output Format

The query script shows:
- The LLM's answer
- Source citations (commit hash, subject, file, similarity distance) for transparency