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
Query Git History RAG
---------------------
Retrieves relevant git history chunks from the ChromaDB vector index
and uses an LLM to answer questions about the code changes.

Supports:
  - OpenAI API (default)
  - Any OpenAI-compatible API (e.g., Ollama, LM Studio, Azure OpenAI)
  - Local Ollama models

Dependencies:
    pip install chromadb sentence-transformers openai

Usage:
    # Interactive mode
    python query_git_rag.py

    # Single question
    python query_git_rag.py "What changes were made to the clustering code?"

    # With custom DB path
    python query_git_rag.py "What changed in pom.xml?" --db ./git_history_db

Environment variables:
    OPENAI_API_KEY     - Your OpenAI API key (or compatible provider key)
    OPENAI_BASE_URL    - Base URL for OpenAI-compatible API (default: https://api.openai.com/v1)
    OPENAI_MODEL       - Model name (default: gpt-4o-mini)
"""

import argparse
import sys
from pathlib import Path

from chromadb import PersistentClient
from chromadb.utils import embedding_functions

import config


def get_embedding_function():
    """Return the same embedding function used at index time."""
    try:
        return embedding_functions.ONNXMiniLM_L6_V2()
    except Exception:
        return embedding_functions.DefaultEmbeddingFunction()


def load_collection(db_dir: Path):
    """Load the ChromaDB collection."""
    if not db_dir.exists():
        print(f"Error: Database not found: {db_dir}", file=sys.stderr)
        print("Run build_rag_index.py first to create the index.", file=sys.stderr)
        sys.exit(1)

    client = PersistentClient(path=str(db_dir))
    try:
        collection = client.get_collection(
            name=config.COLLECTION_NAME,
            embedding_function=get_embedding_function(),
        )
    except Exception as e:
        print(f"Error loading collection: {e}", file=sys.stderr)
        print("Run build_rag_index.py first to create the index.", file=sys.stderr)
        sys.exit(1)

    return collection


def retrieve_context(collection, question: str, top_k: int = config.DEFAULT_TOP_K) -> list[dict]:
    """Retrieve the most relevant chunks for the question."""
    results = collection.query(
        query_texts=[question],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    if results and results["documents"] and results["documents"][0]:
        for i, doc in enumerate(results["documents"][0]):
            meta = results["metadatas"][0][i] if results["metadatas"] else {}
            distance = results["distances"][0][i] if results["distances"] else 0
            chunks.append({"text": doc, "metadata": meta, "distance": distance})

    return chunks


def build_prompt(question: str, chunks: list[dict]) -> str:
    """Build the prompt for the LLM with retrieved context."""
    context_parts = []
    for i, chunk in enumerate(chunks):
        meta = chunk["metadata"]
        commit = meta.get("commit", "unknown")
        subject = meta.get("subject", "")
        file = meta.get("file", "")

        header = f"--- Chunk {i + 1} | Commit: {commit} | Subject: {subject} | File: {file} ---"
        context_parts.append(f"{header}\n{chunk['text']}")

    context = "\n\n".join(context_parts)

    prompt = f"""You are an expert code analyst. You are given relevant sections of a Git repository's commit history (including code diffs). Answer the user's question based ONLY on the provided context. If the answer cannot be found in the context, say so clearly.

CONTEXT (Git history changes):
{context}

QUESTION:
{question}

ANSWER:
"""
    return prompt


def query_llm(prompt: str) -> str:
    """Send the prompt to an LLM via the OpenAI-compatible API."""
    try:
        from openai import OpenAI
    except ImportError:
        print("Error: openai package not installed. Run: pip install openai", file=sys.stderr)
        sys.exit(1)

    api_key = config.OPENAI_API_KEY
    base_url = config.OPENAI_BASE_URL
    model = config.OPENAI_MODEL

    if not api_key and "localhost" not in base_url and "127.0.0.1" not in base_url:
        print("Error: OPENAI_API_KEY environment variable not set.", file=sys.stderr)
        print("Set it in config.py (OPENAI_API_KEY) or with: export OPENAI_API_KEY='your-key'", file=sys.stderr)
        print("Or use a local model: set OPENAI_BASE_URL='http://localhost:11434/v1' in config.py", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=api_key or "ollama", base_url=base_url)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful code analysis assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=config.LLM_TEMPERATURE,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling LLM: {e}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Query the git history RAG index with an LLM.",
        epilog="""Examples:
  python query_git_rag.py "What changes were made to the clustering code?"
  python query_git_rag.py --interactive
  python query_git_rag.py "What changed in pom.xml?" --db ./git_history_db --top-k 8
""",
    )
    parser.add_argument(
        "question",
        type=str,
        nargs="?",
        help="Question to ask about the git history (omit for interactive mode)",
    )
    parser.add_argument(
        "--db",
        type=str,
        default=config.CHROMA_DIR,
        help=f"Path to the ChromaDB database (default: {config.CHROMA_DIR})",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=config.DEFAULT_TOP_K,
        help=f"Number of relevant chunks to retrieve (default: {config.DEFAULT_TOP_K})",
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Run in interactive chat mode",
    )
    args = parser.parse_args()

    db_dir = Path(args.db).resolve()
    collection = load_collection(db_dir)

    print(f"Loaded RAG index from: {db_dir}")
    print(f"Total chunks in index: {collection.count()}")
    print()

    def ask(question: str) -> None:
        print(f"🔍 Retrieving relevant context for: {question}")
        chunks = retrieve_context(collection, question, args.top_k)
        print(f"   Retrieved {len(chunks)} relevant chunks")

        print("🤖 Asking LLM...")
        prompt = build_prompt(question, chunks)
        answer = query_llm(prompt)

        print("\n" + "=" * 60)
        print("ANSWER:")
        print("=" * 60)
        print(answer)
        print("=" * 60)

        # Show sources
        print("\n📎 SOURCES:")
        for i, chunk in enumerate(chunks):
            meta = chunk["metadata"]
            print(f"  [{i + 1}] Commit: {meta.get('commit', '?')} | "
                  f"Subject: {meta.get('subject', '?')} | "
                  f"File: {meta.get('file', '?')} | "
                  f"Distance: {chunk['distance']:.4f}")

    if args.interactive or not args.question:
        print("Interactive mode — type your questions (or 'quit' to exit).")
        print("-" * 60)
        while True:
            try:
                question = input("\n❓ Question: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not question:
                continue
            if question.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break

            ask(question)
            print()
    else:
        ask(args.question)


if __name__ == "__main__":
    main()