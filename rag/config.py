from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RAGConfig:
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    ollama_model: str = "llama3"
    ollama_base_url: str = "http://127.0.0.1:11434"
    system_prompt: str = "Du bist ein hilfreicher Assistent. Antworte klar, präzise und gut strukturiert auf Deutsch."
    chunk_size: int = 700
    chunk_overlap: int = 120
    top_k: int = 4
    store_dir: Path = Path("rag_store")


DEFAULT_CONFIG = RAGConfig()
