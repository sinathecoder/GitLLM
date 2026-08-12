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
Do a similarity search against a ChromaDB store.

Usage:
    python search_db.py "clustering code" [--db ./git_history_db] [--collection git_history] [--top-k 5]

Prints the top-k nearest chunks with their metadata and distance.
"""
import argparse
import sys

import chromadb

import config


def preview(text: str, width: int = 140) -> str:
    text = " ".join(text.split())
    return text[:width] + ("..." if len(text) > width else "")


def search(db_path: str, collection_name: str, query: str, top_k: int) -> int:
    client = chromadb.PersistentClient(path=db_path)
    col = client.get_collection(collection_name)
    total = col.count()

    if total == 0:
        print(f"Collection '{collection_name}' is empty.")
        return 0

    n = max(1, min(top_k, total))
    res = col.query(
        query_texts=[query],
        n_results=n,
        include=["documents", "metadatas", "distances"],
    )

    print(f"Query: {query!r}")
    print(f"Top {n} chunks from '{collection_name}':\n")

    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    distances = (res.get("distances") or [[]])[0]

    for i in range(len(docs)):
        meta = metas[i] or {}
        print(f"--- rank {i + 1} | distance={distances[i]:.4f} ---")
        if meta:
            print(f"    commit : {meta.get('commit', '')}")
            print(f"    subject: {meta.get('subject', '')}")
            print(f"    file   : {meta.get('file', '')}")
        print(f"    text   : {preview(docs[i])!r}\n")

    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Similarity search a ChromaDB store.")
    ap.add_argument("query", help="Search text / prompt")
    ap.add_argument("--db", default=config.CHROMA_DIR, help="ChromaDB directory")
    ap.add_argument("--collection", default=config.COLLECTION_NAME, help="Collection name")
    ap.add_argument("--top-k", type=int, default=config.DEFAULT_TOP_K, help="Number of results")
    args = ap.parse_args()

    try:
        return search(args.db, args.collection, args.query, args.top_k)
    except Exception as exc:  # noqa: BLE001 - surface user-friendly error
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())