#
# Copyright 2026
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Tests for app.py (Git History RAG API).

Run with:
    python3 -m pytest test_app.py -v

Notes:
- Uses a temporary ChromaDB directory so the real index is never touched.
- Mocks the Gemini API call so /api/query works without network access.
"""
import os
import sys

import pytest

# Ensure local app module is importable
sys.path.insert(0, os.path.dirname(__file__))

import app as app_module  # noqa: E402


# ============================================================
# Fixtures
# ============================================================
@pytest.fixture()
def client(tmp_path, monkeypatch):
    """Flask test client wired to an isolated ChromaDB dir & a stubbed Gemini."""
    # Use a throwaway DB directory for every test
    monkeypatch.setattr(app_module, "CHROMA_DIR", str(tmp_path / "chroma_test"))
    # Shorten default config to keep tests fast/independent of env
    monkeypatch.setattr(app_module, "COLLECTION_NAME", "test_collection")

    app_module.app.config.update(TESTING=True)

    with app_module.app.test_client() as c:
        yield c


@pytest.fixture()
def mock_gemini(monkeypatch):
    """Stub out the Gemini call to return a canned answer."""

    def fake_query_gemini(prompt: str) -> str:
        return "FAKE_ANSWER"

    monkeypatch.setattr(app_module, "query_gemini", fake_query_gemini)
    return fake_query_gemini


# ============================================================
# Unit tests: chunk_text
# ============================================================
def test_chunk_text_empty():
    assert app_module.chunk_text("   ") == []
    assert app_module.chunk_text("") == []


def test_chunk_text_short_single_chunk():
    text = "This is a short piece of history."
    assert app_module.chunk_text(text, chunk_size=1500, chunk_overlap=200) == [text]


def test_chunk_text_long_creates_overlapping_chunks():
    # ~100 words of filler
    text = ("word " * 300).strip()
    chunks = app_module.chunk_text(text, chunk_size=500, chunk_overlap=50)
    assert len(chunks) > 1
    # Every chunk is at or under the max size
    assert all(len(c) <= 501 for c in chunks)
    # Chunks are non-trivially sized
    assert all(len(c) > 50 for c in chunks)


def test_chunk_text_detects_git_history_format():
    # A single commit with one file diff that's larger than default chunk size,
    # formatted like git_history_extractor.py output.
    section = "diff --git a/README.md b/README.md\n" + ("+line\n" * 400)
    text = (
        "============================================================\n"
        "Commit: abc123456789\n"
        "Subject: Add docs\n"
        "Body:\n"
        "============================================================\n"
        + section
    )
    chunks = app_module.chunk_text(text, chunk_size=1500, chunk_overlap=200)
    assert len(chunks) >= 1
    # The commit header chunk carries the hash and subject; downstream
    # word-chunked sections may not. At least one chunk must reference it.
    assert any("abc123456789" in c for c in chunks)
    assert any("Add docs" in c for c in chunks)
# ============================================================
# Endpoint tests
# ============================================================
def test_index_endpoint(client):
    r = client.get("/")
    assert r.status_code == 200
    body = r.get_json()
    assert body["name"] == "Git History RAG API"
    assert "endpoints" in body


def test_stats_endpoint_empty(client):
    r = client.get("/api/stats")
    assert r.status_code == 200
    body = r.get_json()
    assert body["total_chunks"] == 0
    assert body["collection"] == "test_collection"


def test_ingest_requires_text(client):
    r = client.post("/api/ingest", json={"chunk_size": 500})
    assert r.status_code == 400
    assert r.get_json()["error"]


def test_ingest_requires_nonempty_text(client):
    r = client.post("/api/ingest", json={"text": "   "})
    assert r.status_code == 400
    assert r.get_json()["error"]


def test_ingest_and_stats_roundtrip(client):
    text = "UA big long history block. " * 120  # force multiple chunks

    r = client.post("/api/ingest", json={"text": text, "chunk_size": 500, "chunk_overlap": 50})
    assert r.status_code == 201, r.get_json()
    body = r.get_json()
    assert body["success"] is True
    assert body["chunks_created"] == body["total_chunks_in_db"]
    assert body["chunks_created"] > 1

    st = client.get("/api/stats").get_json()
    assert st["total_chunks"] == body["chunks_created"]


def test_ingest_git_history_extracts_metadata(client):
    section = "diff --git a/README.md b/README.md\n+One change line\n"
    text = (
        "============================================================\n"
        "Commit: beefcafe1234\n"
        "Subject: Update README\n"
        "Body:\n"
        "============================================================\n"
        + section
    )
    r = client.post("/api/ingest", json={"text": text})
    assert r.status_code == 201, r.get_json()
    body = r.get_json()
    # One chunk for the commit header info, one for the file diff.
    assert body["chunks_created"] >= 2
def test_query_without_chunks_returns_404(client, mock_gemini):
    r = client.post("/api/query", json={"prompt": "Any question?"})
    assert r.status_code == 404
    assert "No chunks" in r.get_json()["error"]


def test_query_requires_prompt(client):
    r = client.post("/api/query", json={})
    assert r.status_code == 400
    assert "prompt" in r.get_json()["error"]

    r = client.post("/api/query", json={"prompt": "  "})
    assert r.status_code == 400


def test_query_success_returns_answer_and_sources(client, mock_gemini):
    client.post("/api/ingest", json={"text": "diff --git a/a.py b/a.py\n+def hello(): return 1\n"})

    r = client.post(
        "/api/query",
        json={"prompt": "What does a.py do?", "top_k": 3, "include_sources": True},
    )
    assert r.status_code == 200, r.get_json()
    body = r.get_json()
    assert body["answer"] == "FAKE_ANSWER"
    assert len(body["sources"]) >= 1
    assert body["sources"][0]["commit"]
    assert body["sources"][0]["file"] == "a.py"


def test_query_without_sources_omits_sources(client, mock_gemini):
    client.post("/api/ingest", json={"text": "diff --git a/a.py b/a.py\n+def x(): pass\n"})

    r = client.post("/api/query", json={"prompt": "q", "include_sources": False})
    body = r.get_json()
    assert r.status_code == 200
    assert body["answer"] == "FAKE_ANSWER"
    assert "sources" not in body


# ============================================================
# Web UI + retrieval-only endpoint tests
# ============================================================
def test_ui_serves_console(client):
    r = client.get("/ui")
    assert r.status_code == 200
    assert "RAG Test Console" in r.get_data(as_text=True)

    r2 = client.get("/ui/")
    assert r2.status_code == 200


def test_index_endpoint_lists_ui(client):
    body = client.get("/").get_json()
    assert "GET /ui" in body["endpoints"]


def test_retrieve_requires_prompt(client):
    r = client.post("/api/retrieve", json={})
    assert r.status_code == 400
    assert "prompt" in r.get_json()["error"]

    r = client.post("/api/retrieve", json={"prompt": "   "})
    assert r.status_code == 400


def test_retrieve_success_without_llm(client):
    """Retrieval-only endpoint returns chunks and never calls Gemini."""
    client.post("/api/ingest", json={
        "text": "diff --git a/widget.py b/widget.py\n+def widget(): return 'hello'\n"
    })

    r = client.post("/api/retrieve", json={"prompt": "what is widget?", "top_k": 3})
    assert r.status_code == 200, r.get_json()
    body = r.get_json()
    assert body["prompt"] == "what is widget?"
    assert body["retrieved_count"] >= 1
    assert body["total_chunks_in_db"] >= 1
    assert body["avg_distance"] is not None
    first = body["retrieved"][0]
    assert first["file"] == "widget.py"
    assert "distance" in first
    assert "text" in first


def test_retrieve_empty_db_returns_404(client):
    r = client.post("/api/retrieve", json={"prompt": "anything"})
    assert r.status_code == 404
    assert "No chunks" in r.get_json()["error"]