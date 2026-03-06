from __future__ import annotations

import argparse
from pathlib import Path

from rag.config import DEFAULT_CONFIG
from rag.logging_config import get_logger
from rag.ollama_rag import build_index

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build llama-index RAG index from text/markdown files")
    parser.add_argument("--input-dir", type=Path, default=Path("data/documents"))
    parser.add_argument("--store-dir", type=Path, default=DEFAULT_CONFIG.store_dir)
    parser.add_argument("--embedding-model", type=str, default=DEFAULT_CONFIG.embedding_model)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CONFIG.chunk_size)
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CONFIG.chunk_overlap)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger.info("CLI ingest gestartet mit args=%s", args)
    meta = build_index(
        input_dir=args.input_dir,
        persist_dir=args.store_dir,
        embedding_model=args.embedding_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    logger.info("CLI ingest erfolgreich: %s", meta)
    print(
        "Index erstellt: "
        f"{meta['num_documents']} Dokumente | "
        f"embedding={meta['embedding_model']} | "
        f"store={args.store_dir}"
    )


if __name__ == "__main__":
    main()
