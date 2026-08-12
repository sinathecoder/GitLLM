#!/usr/bin/env python3
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
Build RAG Index from Git History
--------------------------------
Reads a git history text file (as produced by git_history_extractor.py),
chunks it, embeds it, and stores it in a ChromaDB vector database so an
LLM can retrieve and answer questions about the code changes.

Dependencies:
    pip install chromadb sentence-transformers

Usage:
    python build_rag_index.py git_history.txt [--db ./git_history_db] [--chunk-size 1500] [--chunk-overlap 200]
"""

import argparse
import re
import sys
from pathlib import Path

from chromadb import PersistentClient
from chromadb.utils import embedding_functions

import config

# Use ChromaDB's built-in ONNX embedding model (no torch/pytorch required)
def get_embedding_function():
    """Return the embedding function used for vector search."""
    try:
        return embedding_functions.ONNXMiniLM_L6_V2()
    except Exception:
        # Fallback to default
        return embedding_functions.DefaultEmbeddingFunction()


def read_history_file(file_path: Path) -> str:
    """Read the git history text file."""
    if not file_path.exists():
        print(f"Error: File not found: {file_path}", file=sys.stderr)
        sys.exit(1)
    return file_path.read_text(encoding="utf-8")


def split_into_commits(content: str) -> list[str]:
    """
    Split the history file into individual commit blocks.
    The history format is:
        ==== (separator)
        Commit: <hash>
        ...
        ==== (end of header)
        diff --git ...
        ...
    We split on '====' lines that are followed by 'Commit:'.
    """
    # Split on separator lines that are followed by "Commit:" (start of a new commit)
    # Keep the separator with the following commit block
    parts = re.split(r"\n={10,}\n(?=Commit:)", content)
    commits = []
    for part in parts:
        part = part.strip()
        if part and ("Commit:" in part or "diff --git" in part):
            commits.append(part)
    return commits


def chunk_commit(commit_text: str, chunk_size: int, chunk_overlap: int) -> list[tuple[str, dict]]:
    """
    Chunk a single commit into smaller pieces.
    Returns a list of (chunk_text, metadata) tuples.
    """
    # Extract commit hash and subject for metadata
    commit_hash = ""
    subject = ""
    hash_match = re.search(r"Commit:\s*([a-f0-9]+)", commit_text)
    if hash_match:
        commit_hash = hash_match.group(1)

    subject_match = re.search(r"Subject:\s*(.+)", commit_text)
    if subject_match:
        subject = subject_match.group(1).strip()

    metadata_base = {"commit": commit_hash, "subject": subject}

    # Split the commit into per-file diff sections
    file_sections = re.split(r"(?=diff --git )", commit_text)
    chunks = []

    for section in file_sections:
        section = section.strip()
        if not section:
            continue

        # Extract the file path for metadata
        file_path = ""
        file_match = re.search(r"diff --git a/(\S+) b/", section)
        if file_match:
            file_path = file_match.group(1)

        # If this section doesn't have the commit header, prepend it
        # so every chunk carries the commit context
        if "Commit:" not in section and commit_hash:
            section = (
                f"Commit: {commit_hash}\n"
                f"Subject: {subject}\n"
                f"File: {file_path}\n"
                f"{section}"
            )

        meta = {**metadata_base, "file": file_path}

        # If the section is small enough, keep it as one chunk
        if len(section) <= chunk_size:
            chunks.append((section, meta))
            continue

        # Otherwise split into overlapping chunks
        words = section.split()
        current_chunk = []
        current_len = 0
        overlap_words: list[str] = []

        for word in words:
            current_chunk.append(word)
            current_len += len(word) + 1

            if current_len >= chunk_size:
                chunk_text = " ".join(current_chunk)
                chunks.append((chunk_text, meta))
                # Set up overlap
                overlap_words = current_chunk[-chunk_overlap // 10:]
                current_chunk = list(overlap_words)
                current_len = sum(len(w) + 1 for w in current_chunk)

        if current_chunk:
            chunks.append((" ".join(current_chunk), meta))

    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a RAG vector index from a git history text file."
    )
    parser.add_argument(
        "history_file",
        type=str,
        help="Path to the git history text file (from git_history_extractor.py)",
    )
    parser.add_argument(
        "--db",
        type=str,
        default=config.CHROMA_DIR,
        help=f"Directory for the ChromaDB vector database (default: {config.CHROMA_DIR})",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=config.DEFAULT_CHUNK_SIZE,
        help=f"Chunk size in characters (default: {config.DEFAULT_CHUNK_SIZE})",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=config.DEFAULT_CHUNK_OVERLAP,
        help=f"Overlap between chunks in characters (default: {config.DEFAULT_CHUNK_OVERLAP})",
    )
    args = parser.parse_args()

    history_file = Path(args.history_file).resolve()
    print(f"Reading history file: {history_file}")

    content = read_history_file(history_file)
    commits = split_into_commits(content)
    print(f"Found {len(commits)} commit blocks")

    # Chunk all commits
    all_chunks: list[tuple[str, dict]] = []
    for i, commit in enumerate(commits):
        commit_chunks = chunk_commit(commit, args.chunk_size, args.chunk_overlap)
        all_chunks.extend(commit_chunks)
        print(f"  Commit {i + 1}/{len(commits)}: {len(commit_chunks)} chunks")

    print(f"Total chunks: {len(all_chunks)}")

    # Set up ChromaDB
    db_dir = Path(args.db).resolve()
    db_dir.mkdir(parents=True, exist_ok=True)

    client = PersistentClient(path=str(db_dir))
    collection_name = config.COLLECTION_NAME

    # Delete existing collection if it exists
    try:
        client.delete_collection(collection_name)
        print(f"Removed existing '{collection_name}' collection")
    except Exception:
        pass

    # Use ONNX MiniLM embedding model
    print("Loading embedding model (ONNX MiniLM-L6-v2)...")
    embedding_fn = get_embedding_function()

    collection = client.create_collection(
        name=collection_name,
        embedding_function=embedding_fn,
        metadata={"description": config.COLLECTION_DESCRIPTION},
    )

    # Add chunks to the collection
    ids = [f"chunk_{i}" for i in range(len(all_chunks))]
    doc_texts = [chunk[0] for chunk in all_chunks]
    metadatas = [chunk[1] for chunk in all_chunks]

    print("Embedding and storing chunks...")
    batch_size = config.CHROMA_BATCH_SIZE
    for i in range(0, len(all_chunks), batch_size):
        batch_end = min(i + batch_size, len(all_chunks))
        collection.add(
            ids=ids[i:batch_end],
            documents=doc_texts[i:batch_end],
            metadatas=metadatas[i:batch_end],
        )
        print(f"  Stored chunks {i + 1}-{batch_end}")

    print(f"\n✅ RAG index built successfully!")
    print(f"   Collection:    {collection_name}")
    print(f"   Total chunks:  {collection.count()}")
    print(f"   Database dir:  {db_dir}")
    print(f"\nNext step: run query_git_rag.py to ask questions about the code changes.")


if __name__ == "__main__":
    main()