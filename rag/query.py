from __future__ import annotations

import argparse
from pathlib import Path

from rag.config import DEFAULT_CONFIG
from rag.logging_config import get_logger
from rag.ollama_rag import query_index, resolve_index_settings

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local RAG query via Ollama")
    parser.add_argument("question", type=str, help="User question")
    parser.add_argument("--store-dir", type=Path, default=DEFAULT_CONFIG.store_dir)
    parser.add_argument("--top-k", type=int, default=DEFAULT_CONFIG.top_k)
    parser.add_argument("--ollama-model", type=str, default=DEFAULT_CONFIG.ollama_model)
    parser.add_argument("--ollama-base-url", type=str, default=DEFAULT_CONFIG.ollama_base_url)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger.info("CLI query gestartet mit args=%s", args)
    settings = resolve_index_settings(
        persist_dir=args.store_dir,
        default_embedding_model=DEFAULT_CONFIG.embedding_model,
        default_chunk_size=DEFAULT_CONFIG.chunk_size,
        default_chunk_overlap=DEFAULT_CONFIG.chunk_overlap,
    )

    answer, sources = query_index(
        question=args.question,
        persist_dir=args.store_dir,
        embedding_model=settings["embedding_model"],
        ollama_model=args.ollama_model,
        ollama_base_url=args.ollama_base_url,
        top_k=args.top_k,
        chunk_size=settings["chunk_size"],
        chunk_overlap=settings["chunk_overlap"],
    )
    logger.info("CLI query erfolgreich: %s Quellen", len(sources))

    print("=== Retrieved Context ===")
    for idx, src in enumerate(sources, start=1):
        print(f"[{idx}] score={src['score']:.4f} | source={src['source']}")
        print(src["text"])
        print("-" * 80)

    print("\\n=== Answer ===")
    print(answer)


if __name__ == "__main__":
    main()
