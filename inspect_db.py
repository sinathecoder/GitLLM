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
Inspect a ChromaDB persistent store.

Usage:
    python inspect_db.py [--db ./git_history_db] [--collection name] [--limit 5]

Prints, for each collection (or the one requested):
  - total chunk count
  - metadata (commit / subject / file) and a text preview for each chunk

NOTE: Chroma locks the DB directory. Do NOT run this against a DB that is
already open by a running server (e.g. app.py). Stop the server first.
"""
import argparse
import sys

import chromadb

import config


def preview(text: str, width: int = 140) -> str:
    text = " ".join(text.split())
    return text[:width] + ("..." if len(text) > width else "")


def inspect(db_path: str, collection_name: str | None, limit: int) -> int:
    client = chromadb.PersistentClient(path=db_path)

    if collection_name:
        collections = [client.get_collection(collection_name)]
    else:
        collections = list(client.list_collections())

    if not collections:
        print(f"No collections found in {db_path}")
        return 0

    for coll in collections:
        col = client.get_collection(coll.name)
        total = col.count()
        print(f"\n=== collection '{coll.name}': {total} docs ===")

        if total == 0:
            print("  (empty)")
            continue

        take = min(limit, total)
        got = col.get(limit=take, include=["documents", "metadatas"])
        for i in range(take):
            meta = (got["metadatas"] or [{}] * take)[i] or {}
            print(f"  id={got['ids'][i]!r}")
            if meta:
                print(f"    meta: {meta}")
            doc = (got["documents"] or [""] * take)[i] or ""
            print(f"    text: {preview(doc)!r}")

    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Inspect a ChromaDB store.")
    ap.add_argument("--db", default=config.CHROMA_DIR, help="ChromaDB directory")
    ap.add_argument("--collection", default=None, help=f"Collection name (default: all, or '{config.COLLECTION_NAME}')")
    ap.add_argument("--limit", type=int, default=5, help="Max chunks to preview")
    args = ap.parse_args()

    try:
        return inspect(args.db, args.collection, args.limit)
    except Exception as exc:  # noqa: BLE001 - surface user-friendly error
        print(f"Error inspecting '{args.db}': {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())