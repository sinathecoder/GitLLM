#!/usr/bin/env python3
"""
Central configuration for the GitLLM RAG project.

All tunable settings live here so you can adjust behaviour in one place instead
of editing every script. Every value can also be overridden with an environment
variable (see ``os.environ.get`` calls below).

Import in any module with::

    import config
    # or
    from config import CHROMA_DIR, GEMINI_MODEL

Adjust paths/values here, then re-run whichever script you need:
    python3 build_rag_index.py <history_file>
    python3 query_git_rag.py "your question"
    python3 app.py            # open http://localhost:5000/ui
"""

import os
from pathlib import Path

# ============================================================
# Paths
# ============================================================
# Absolute directory that contains this file (the project root).
BASE_DIR = Path(__file__).resolve().parent

# Default ChromaDB vector database directory (the populated RAG index).
CHROMA_DIR = os.environ.get("CHROMA_DIR", "./git_history_db")

# ChromaDB collection name used across every script.
COLLECTION_NAME = "git_history"

# Metadata description attached to the collection when it is first created.
COLLECTION_DESCRIPTION = "Git history changes for RAG"

# ChromaDB add/upsert batch size.
CHROMA_BATCH_SIZE = 100

# ============================================================
# Text chunking
# ============================================================
DEFAULT_CHUNK_SIZE = 1500          # target chunk size (in characters)
DEFAULT_CHUNK_OVERLAP = 200        # overlap between chunks (in characters)

# ============================================================
# Embedding model
# ============================================================
# Name of the ChromaDB embedding function. Currently we use the bundled ONNX
# MiniLM-L6-v2 model (no PyTorch/torch required).
EMBEDDING_MODEL = "ONNXMiniLM_L6_V2"

# ============================================================
# LLM — Gemini (used by the Flask API / web UI at /api/query)
# ============================================================
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
GEMINI_API_KEY = os.environ.get(
    "GEMINI_API_KEY", "AQ.Ab8RN6IdgLzFMViYHrHa8QpptA3o71fO2CZ4gsPNFPxfxKFopw"
)
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3-flash-preview")
GEMINI_TEMPERATURE = 0.2
GEMINI_MAX_OUTPUT_TOKENS = 2048

# ============================================================
# LLM — OpenAI-compatible (used by query_git_rag.py CLI)
# ============================================================
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
LLM_TEMPERATURE = 0.2

# ============================================================
# Retrieval
# ============================================================
DEFAULT_TOP_K = 5                  # number of chunks retrieved by default

# ============================================================
# Git history extraction
# ============================================================
HISTORY_OUTPUT_DEFAULT = "git_history.txt"   # default --output for the extractor

# ============================================================
# Web server (Flask / app.py)
# ============================================================
HOST = "0.0.0.0"
PORT = 5000
DEBUG = True

# Directory serving the web UI assets at GET /ui.
WEBUI_DIR = str(BASE_DIR / "webui")
