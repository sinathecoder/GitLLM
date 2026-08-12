#!/usr/bin/env python3
"""
Git History RAG API
-------------------
Flask application with two endpoints:

1. POST /api/ingest  - Accept text, chunk it, and store in ChromaDB
2. POST /api/query   - Accept a prompt, retrieve relevant chunks from ChromaDB (RAG),
                       and answer using Google Gemini

Usage:
    python app.py
"""

import re
import uuid
from typing import Any

import chromadb
import requests
from chromadb.utils import embedding_functions
from flask import Flask, jsonify, request, send_from_directory

import config

app = Flask(__name__)

# ============================================================
# Configuration (single source of truth: config.py)
# ============================================================
# Directory containing the web UI assets served at /ui
WEBUI_DIR = config.WEBUI_DIR

GEMINI_API_BASE = config.GEMINI_API_BASE
GEMINI_API_KEY = config.GEMINI_API_KEY
GEMINI_MODEL = config.GEMINI_MODEL
GEMINI_TEMPERATURE = config.GEMINI_TEMPERATURE
GEMINI_MAX_OUTPUT_TOKENS = config.GEMINI_MAX_OUTPUT_TOKENS
GEMINI_URL = f"{GEMINI_API_BASE}/{GEMINI_MODEL}:generateContent"

CHROMA_DIR = config.CHROMA_DIR
COLLECTION_NAME = config.COLLECTION_NAME
COLLECTION_DESCRIPTION = config.COLLECTION_DESCRIPTION
DEFAULT_CHUNK_SIZE = config.DEFAULT_CHUNK_SIZE
DEFAULT_CHUNK_OVERLAP = config.DEFAULT_CHUNK_OVERLAP

# ============================================================
# Embedding Function (ONNX MiniLM — no torch needed)
# ============================================================
def get_embedding_function():
    try:
        return embedding_functions.ONNXMiniLM_L6_V2()
    except Exception:
        return embedding_functions.DefaultEmbeddingFunction()


def get_collection():
    """Get or create the ChromaDB collection."""
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        return client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=get_embedding_function(),
        )
    except Exception:
        return client.create_collection(
            name=COLLECTION_NAME,
            embedding_function=get_embedding_function(),
            metadata={"description": COLLECTION_DESCRIPTION},
        )


# ============================================================
# Text Chunking
# ============================================================
def chunk_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE,
               chunk_overlap: int = DEFAULT_CHUNK_OVERLAP) -> list[str]:
    """
    Split text into overlapping chunks.
    If the text looks like a git history report (with diff blocks),
    chunk per-file first; otherwise use simple character overlap.
    """
    text = text.strip()
    if not text:
        return []

    # If the text contains git diffs, chunk by file sections
    if "diff --git" in text:
        return chunk_git_history(text, chunk_size, chunk_overlap)

    # Simple character-based chunking with overlap
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - chunk_overlap

    return chunks


def chunk_git_history(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split git history text into per-commit, per-file chunks."""
    # Split into commit blocks (separator followed by "Commit:")
    commit_blocks = re.split(r"\n={10,}\n(?=Commit:)", text)

    # If we didn't find commit separators, treat the whole text as one block
    if len(commit_blocks) == 1 and "Commit:" not in commit_blocks[0][:500]:
        commit_blocks = [text]

    all_chunks: list[str] = []
    for commit in commit_blocks:
        commit = commit.strip()
        if not commit:
            continue

        # Extract metadata
        commit_hash = ""
        subject = ""
        hash_match = re.search(r"Commit:\s*([a-f0-9]+)", commit)
        if hash_match:
            commit_hash = hash_match.group(1)

        subject_match = re.search(r"Subject:\s*(.+)", commit)
        if subject_match:
            subject = subject_match.group(1).strip()

        # Split commit into file diff sections
        file_sections = re.split(r"(?=diff --git )", commit)

        for section in file_sections:
            section = section.strip()
            if not section:
                continue

            # Extract file path
            file_path = ""
            file_match = re.search(r"diff --git a/(\S+) b/", section)
            if file_match:
                file_path = file_match.group(1)

            # Prepend commit context to file-only sections
            if "Commit:" not in section and commit_hash:
                section = (
                    f"Commit: {commit_hash}\n"
                    f"Subject: {subject}\n"
                    f"File: {file_path}\n"
                    f"{section}"
                )

            # Chunk if larger than max size
            if len(section) <= chunk_size:
                all_chunks.append(section)
                continue

            # Overlapping word-based chunking for large sections
            words = section.split()
            current = []
            current_len = 0
            overlap_words: list[str] = []

            for word in words:
                current.append(word)
                current_len += len(word) + 1

                if current_len >= chunk_size:
                    all_chunks.append(" ".join(current))
                    overlap_words = current[-chunk_overlap // 10:]
                    current = list(overlap_words)
                    current_len = sum(len(w) + 1 for w in current)

            if current:
                all_chunks.append(" ".join(current))

    return all_chunks


# ============================================================
# Gemini API
# ============================================================
def query_gemini(prompt: str) -> str:
    """Send a prompt to Google Gemini and return the response text."""
    payload = {
        "contents": [
            {
                "parts": [{"text": prompt}]
            }
        ],
        "generationConfig": {
            "temperature": GEMINI_TEMPERATURE,
            "maxOutputTokens": GEMINI_MAX_OUTPUT_TOKENS,
        }
    }

    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(
            f"{GEMINI_URL}?key={GEMINI_API_KEY}",
            json=payload,
            headers=headers,
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()

        # Extract text from Gemini response
        candidates = data.get("candidates", [])
        if not candidates:
            return "No response generated."

        parts = candidates[0].get("content", {}).get("parts", [])
        return "".join(part.get("text", "") for part in parts)

    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Error calling Gemini API: {e}")


def build_rag_prompt(question: str, chunks: list[dict[str, Any]]) -> str:
    """Build a prompt with retrieved context for the LLM."""
    context_parts = []
    for i, chunk in enumerate(chunks):
        meta = chunk.get("metadata", {})
        header = (
            f"--- Chunk {i + 1} "
            f"| Commit: {meta.get('commit', 'unknown')} "
            f"| Subject: {meta.get('subject', '')} "
            f"| File: {meta.get('file', '')} ---"
        )
        context_parts.append(f"{header}\n{chunk.get('text', '')}")

    context = "\n\n".join(context_parts)

    return f"""You are an expert code analyst. You are given relevant sections of a Git repository's commit history (including code diffs). Answer the user's question based ONLY on the provided context. If the answer cannot be found in the context, say so clearly.

CONTEXT (Git history changes):
{context}

QUESTION:
{question}

ANSWER:
"""


# ============================================================
# API Endpoints
# ============================================================
@app.route("/api/ingest", methods=["POST"])
def ingest_text():
    """
    Accept text, chunk it, and store it in ChromaDB.

    Request body (JSON):
        {
            "text": "the text content to store",
            "chunk_size": 1500,      # optional
            "chunk_overlap": 200,    # optional
            "metadata": {            # optional, extra metadata
                "source": "git_history.txt",
                "repo": "MyRepo"
            }
        }
    """
    data = request.get_json(silent=True)
    if not data or not data.get("text"):
        return jsonify({"error": "Missing 'text' in request body"}), 400

    text = data["text"].strip()
    if not text:
        return jsonify({"error": "Text cannot be empty"}), 400

    chunk_size = int(data.get("chunk_size", DEFAULT_CHUNK_SIZE))
    chunk_overlap = int(data.get("chunk_overlap", DEFAULT_CHUNK_OVERLAP))
    extra_metadata = data.get("metadata", {}) or {}

    # Chunk the text
    chunks = chunk_text(text, chunk_size, chunk_overlap)

    if not chunks:
        return jsonify({"error": "No chunks were generated from the text"}), 400

    # Store in ChromaDB
    collection = get_collection()
    ids = [f"chunk_{uuid.uuid4().hex}" for _ in chunks]
    metadatas = []
    for i, chunk in enumerate(chunks):
        meta = {"index": i, **extra_metadata}

        # Extract useful metadata from the chunk itself
        hash_m = re.search(r"Commit:\s*([a-f0-9]+)", chunk)
        if hash_m:
            meta["commit"] = hash_m.group(1)

        subj_m = re.search(r"Subject:\s*(.+)", chunk)
        if subj_m:
            meta["subject"] = subj_m.group(1).strip()

        file_m = re.search(r"diff --git a/(\S+) b/", chunk)
        if file_m:
            meta["file"] = file_m.group(1)

        metadatas.append(meta)

    # Add in batches
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        end = min(i + batch_size, len(chunks))
        collection.add(
            ids=ids[i:end],
            documents=chunks[i:end],
            metadatas=metadatas[i:end],
        )

    return jsonify({
        "success": True,
        "chunks_created": len(chunks),
        "total_chunks_in_db": collection.count(),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
    }), 201


@app.route("/api/query", methods=["POST"])
def query_rag():
    """
    Accept a prompt, retrieve relevant chunks from ChromaDB (RAG),
    and answer using Google Gemini.

    Request body (JSON):
        {
            "prompt": "What changes were made to the clustering code?",
            "top_k": 5,          # optional, number of chunks to retrieve
            "include_sources": true  # optional, include retrieved sources in response
        }
    """
    data = request.get_json(silent=True)
    if not data or not data.get("prompt"):
        return jsonify({"error": "Missing 'prompt' in request body"}), 400

    prompt = data["prompt"].strip()
    if not prompt:
        return jsonify({"error": "Prompt cannot be empty"}), 400

    top_k = int(data.get("top_k", 5))
    include_sources = bool(data.get("include_sources", True))

    # Retrieve from ChromaDB
    collection = get_collection()
    if collection.count() == 0:
        return jsonify({
            "error": "No chunks in the database. Call /api/ingest first to add text."
        }), 404

    results = collection.query(
        query_texts=[prompt],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for i, doc in enumerate(results["documents"][0]):
        chunks.append({
            "text": doc,
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],
        })

    # Build the RAG prompt and query Gemini
    rag_prompt = build_rag_prompt(prompt, chunks)

    try:
        answer = query_gemini(rag_prompt)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500

    response = {
        "answer": answer,
    }

    if include_sources:
        response["sources"] = []
        for i, chunk in enumerate(chunks):
            response["sources"].append({
                "chunk_index": i + 1,
                "commit": chunk["metadata"].get("commit", "unknown"),
                "subject": chunk["metadata"].get("subject", ""),
                "file": chunk["metadata"].get("file", ""),
                "distance": round(chunk["distance"], 4),
                "text_preview": chunk["text"][:200] + ("..." if len(chunk["text"]) > 200 else ""),
            })

    return jsonify(response)


@app.route("/api/retrieve", methods=["POST"])
def retrieve_rag():
    """
    Retrieval-only preview: return the top-k chunks ChromaDB would feed to the
    LLM for a given prompt, WITHOUT calling any LLM. Useful for inspecting and
    tuning the retrieval step of the RAG pipeline.

    Request body (JSON):
        {
            "prompt": "What changes were made to the clustering code?",
            "top_k": 5    # optional, number of chunks to retrieve
        }
    """
    data = request.get_json(silent=True)
    if not data or not data.get("prompt"):
        return jsonify({"error": "Missing 'prompt' in request body"}), 400

    prompt = data["prompt"].strip()
    if not prompt:
        return jsonify({"error": "Prompt cannot be empty"}), 400

    top_k = int(data.get("top_k", 5))

    collection = get_collection()
    if collection.count() == 0:
        return jsonify({
            "error": "No chunks in the database. Call /api/ingest first to add text."
        }), 404

    results = collection.query(
        query_texts=[prompt],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    distances_sum = 0.0
    for i, doc in enumerate(results["documents"][0]):
        distance = results["distances"][0][i]
        distances_sum += distance
        meta = results["metadatas"][0][i]
        chunks.append({
            "chunk_index": i + 1,
            "commit": meta.get("commit", "unknown"),
            "subject": meta.get("subject", ""),
            "file": meta.get("file", ""),
            "distance": round(distance, 4),
            "text": doc,
        })

    return jsonify({
        "prompt": prompt,
        "top_k": top_k,
        "retrieved_count": len(chunks),
        "total_chunks_in_db": collection.count(),
        "avg_distance": round(distances_sum / len(chunks), 4) if chunks else 0.0,
        "retrieved": chunks,
    })


@app.route("/api/stats", methods=["GET"])
def stats():
    """Get statistics about the ChromaDB collection."""
    collection = get_collection()
    return jsonify({
        "collection": COLLECTION_NAME,
        "total_chunks": collection.count(),
        "chroma_dir": CHROMA_DIR,
        "gemini_model": GEMINI_MODEL,
    })


@app.route("/ui")
@app.route("/ui/")
def ui():
    """Serve the RAG test console web UI."""
    return send_from_directory(WEBUI_DIR, "index.html")


@app.route("/", methods=["GET"])
def index():
    """API info page."""
    return jsonify({
        "name": "Git History RAG API",
        "endpoints": {
            "GET /ui": "Web console for testing prompts & answers on the RAG",
            "POST /api/ingest": "Send text to chunk and store in ChromaDB",
            "POST /api/query": "Ask a question — RAG retrieves context and Gemini answers",
            "POST /api/retrieve": "Retrieval-only preview of top-k chunks (no LLM)",
            "GET /api/stats": "View index statistics",
        },
    })


if __name__ == "__main__":
    print(f"Gemini model: {GEMINI_MODEL}")
    print(f"ChromaDB dir: {CHROMA_DIR}")
    print(f"Starting server on http://localhost:{config.PORT}")
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG)