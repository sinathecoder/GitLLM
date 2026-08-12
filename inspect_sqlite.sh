#!/usr/bin/env bash
#
# Inspect a ChromaDB store's underlying SQLite file directly (no Chroma API).
#
# Usage:
#   ./inspect_sqlite.sh [PATH_TO_DB_DIR]
#
# Defaults to ./git_history_db. Requires the `sqlite3` CLI.
#
set -euo pipefail

DB_DIR="${1:-./git_history_db}"
DB_FILE="$DB_DIR/chroma.sqlite3"

if [[ ! -f "$DB_FILE" ]]; then
  echo "No DB found at: $DB_FILE" >&2
  echo "Pass path to a ChromaDB directory, e.g.: ./inspect_sqlite.sh ./git_history_api_db" >&2
  exit 1
fi

echo "===== ChromaDB SQLite: $DB_FILE ====="
echo
echo "----- tables -----"
sqlite3 -header -column "$DB_FILE" ".tables"
echo

echo "----- collection counts (collections vs embedding segments) -----"
sqlite3 -header -column "$DB_FILE" \
  "SELECT * FROM collections;"
echo

echo "----- per-chunk: embedding id, text start, and metadata -----"
sqlite3 -header -column "$DB_FILE" \
  "SELECT e.embedding_id AS chunk_id,
          substr(COALESCE(ft.c0, ''), 1, 60) AS text_start,
          m.key AS meta_key,
          substr(COALESCE(m.string_value, ''), 1, 40) AS meta_value
   FROM embeddings e
   LEFT JOIN embedding_fulltext_search_content ft ON ft.id = e.id
   LEFT JOIN embedding_metadata m ON m.id = e.id
   ORDER BY e.embedding_id, m.key
   LIMIT 60;"
echo

echo "----- chunk count and how many have text -----"
sqlite3 -header -column "$DB_FILE" \
  "SELECT COUNT(*) AS total_embeddings,
          COUNT(ft.c0) AS with_text
   FROM embeddings e
   LEFT JOIN embedding_fulltext_search_content ft ON ft.id = e.id;"