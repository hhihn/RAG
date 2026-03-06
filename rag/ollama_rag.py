from __future__ import annotations

import json
import time
from functools import lru_cache
from pathlib import Path
from typing import Callable
from urllib.error import URLError
from urllib.request import urlopen

from llama_index.core import Settings, SimpleDirectoryReader, StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

from rag.logging_config import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=4)
def get_embed_model(model_name: str) -> HuggingFaceEmbedding:
    logger.info("Lade Embedding-Modell: %s", model_name)
    try:
        return HuggingFaceEmbedding(model_name=model_name)
    except Exception as exc:
        logger.exception("Fehler beim Laden des Embedding-Modells: %s", model_name)
        raise ValueError(
            "Embedding-Modell ungeeignet oder nicht kompatibel. "
            f"Verwendet: '{model_name}'. "
            "Empfohlen: 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2' "
            "(gut für Deutsch) oder 'intfloat/multilingual-e5-base'."
        ) from exc


@lru_cache(maxsize=4)
def get_ollama_llm(model_name: str, base_url: str) -> Ollama:
    logger.info("Initialisiere Ollama LLM: model=%s base_url=%s", model_name, base_url)
    return Ollama(model=model_name, base_url=base_url, request_timeout=120.0)


def _configure_settings(embedding_model: str, chunk_size: int, chunk_overlap: int) -> HuggingFaceEmbedding:
    embed_model = get_embed_model(embedding_model)
    Settings.embed_model = embed_model
    Settings.chunk_size = chunk_size
    Settings.chunk_overlap = chunk_overlap
    return embed_model


def _emit(progress_callback: Callable[[str], None] | None, message: str) -> None:
    logger.info(message)
    if progress_callback:
        progress_callback(message)


def build_index(
    input_dir: Path,
    persist_dir: Path,
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
    progress_callback: Callable[[str], None] | None = None,
) -> dict:
    in_dir = input_dir.expanduser()
    out_dir = persist_dir.expanduser()
    _emit(
        progress_callback,
        (
            f"Indexbau gestartet: input_dir={in_dir} persist_dir={out_dir} "
            f"embedding_model={embedding_model} chunk_size={chunk_size} chunk_overlap={chunk_overlap}"
        ),
    )
    logger.info(
        "Indexbau Parameter: input_dir=%s persist_dir=%s embedding_model=%s chunk_size=%s chunk_overlap=%s",
        in_dir, out_dir, embedding_model, chunk_size, chunk_overlap
    )

    if not in_dir.exists():
        raise FileNotFoundError(f"Input-Verzeichnis nicht gefunden: {in_dir}")

    _emit(progress_callback, "Initialisiere Embedding-Modell...")
    _configure_settings(embedding_model, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    _emit(progress_callback, "Lade Dokumente...")
    documents = SimpleDirectoryReader(
        input_dir=str(in_dir),
        recursive=True,
        required_exts=[".txt", ".md"],
    ).load_data()
    _emit(progress_callback, f"Dokumente geladen: {len(documents)}")
    if not documents:
        raise ValueError(f"Keine .txt/.md Dateien in {in_dir} gefunden.")

    _emit(progress_callback, "Erzeuge Vektorindex...")
    index = VectorStoreIndex.from_documents(documents, show_progress=True)
    _emit(progress_callback, "Speichere Index auf Disk...")
    index.storage_context.persist(persist_dir=str(out_dir))

    meta = {
        "embedding_model": embedding_model,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "num_documents": len(documents),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "rag_settings.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    _emit(progress_callback, "Indexbau abgeschlossen.")
    logger.info("Indexbau Meta: %s", meta)
    return meta


def load_index(persist_dir: Path, embedding_model: str, chunk_size: int, chunk_overlap: int) -> VectorStoreIndex:
    store = persist_dir.expanduser()
    logger.info("Lade Index: store=%s embedding_model=%s", store, embedding_model)
    if not store.exists():
        raise FileNotFoundError(f"Index-Verzeichnis nicht gefunden: {store}")

    _configure_settings(embedding_model, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    storage_context = StorageContext.from_defaults(persist_dir=str(store))
    return load_index_from_storage(storage_context)


def read_settings(persist_dir: Path) -> dict:
    settings_path = persist_dir.expanduser() / "rag_settings.json"
    if not settings_path.exists():
        logger.warning("Keine rag_settings.json gefunden: %s", settings_path)
        return {}
    logger.info("Lese Settings aus: %s", settings_path)
    return json.loads(settings_path.read_text(encoding="utf-8"))


def resolve_index_settings(
    persist_dir: Path,
    default_embedding_model: str,
    default_chunk_size: int,
    default_chunk_overlap: int,
) -> dict:
    saved = read_settings(persist_dir)
    return {
        "embedding_model": saved.get("embedding_model", default_embedding_model),
        "chunk_size": int(saved.get("chunk_size", default_chunk_size)),
        "chunk_overlap": int(saved.get("chunk_overlap", default_chunk_overlap)),
    }


def check_ollama_connection(base_url: str, timeout_s: float = 2.0) -> None:
    health_url = base_url.rstrip("/") + "/api/tags"
    logger.info("Prüfe Ollama-Verbindung: %s", health_url)
    try:
        with urlopen(health_url, timeout=timeout_s) as resp:
            status = getattr(resp, "status", 200)
            if status >= 400:
                raise ConnectionError(f"Ollama antwortet mit HTTP {status} auf {health_url}")
    except URLError as exc:
        logger.exception("Ollama-Verbindung fehlgeschlagen: %s", health_url)
        raise ConnectionError(
            f"Keine Verbindung zu Ollama unter {health_url}. "
            "Starte Ollama mit `ollama serve` und prüfe die URL."
        ) from exc


def query_index(
    question: str,
    persist_dir: Path,
    embedding_model: str,
    ollama_model: str,
    ollama_base_url: str,
    top_k: int,
    chunk_size: int,
    chunk_overlap: int,
    progress_callback: Callable[[str], None] | None = None,
) -> tuple[str, list[dict]]:
    details = run_rag_query(
        question=question,
        persist_dir=persist_dir,
        embedding_model=embedding_model,
        ollama_model=ollama_model,
        ollama_base_url=ollama_base_url,
        top_k=top_k,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        generate_answer=True,
        progress_callback=progress_callback,
    )
    return details["answer"], details["sources"]


def retrieve_contexts(
    question: str,
    persist_dir: Path,
    embedding_model: str,
    top_k: int,
    chunk_size: int,
    chunk_overlap: int,
    progress_callback: Callable[[str], None] | None = None,
) -> list[dict]:
    _emit(progress_callback, "Lade lokalen Index...")
    index = load_index(
        persist_dir=persist_dir,
        embedding_model=embedding_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    _emit(progress_callback, "Führe Retrieval aus...")
    retriever = index.as_retriever(similarity_top_k=top_k)
    nodes = retriever.retrieve(question)

    sources: list[dict] = []
    for node in nodes:
        meta = node.node.metadata or {}
        sources.append(
            {
                "score": float(node.score or 0.0),
                "source": str(meta.get("file_path", meta.get("filename", "unknown"))),
                "text": node.node.get_content(),
            }
        )
    _emit(progress_callback, f"Retrieval abgeschlossen, {len(sources)} Kontexte gefunden.")
    return sources


def build_teaching_prompt(question: str, sources: list[dict]) -> str:
    return build_teaching_prompt_with_system(question=question, sources=sources, system_prompt="")


def build_teaching_prompt_with_system(question: str, sources: list[dict], system_prompt: str) -> str:
    context_blocks: list[str] = []
    for idx, src in enumerate(sources, start=1):
        context_blocks.append(
            f"[Kontext {idx}] Quelle: {src['source']}\n{src['text']}"
        )

    joined_context = "\n\n".join(context_blocks)
    system_part = system_prompt.strip() or (
        "Du bist ein RAG-Assistent. Beantworte die Frage nur mit den bereitgestellten Kontexten. "
        "Wenn etwas nicht im Kontext steht, sage das klar."
    )
    return (
        f"System:\n{system_part}\n\n"
        f"Kontexte:\n{joined_context}\n\n"
        f"Frage: {question}\n"
        "Antwort:"
    )


def run_rag_query(
    question: str,
    persist_dir: Path,
    embedding_model: str,
    ollama_model: str,
    ollama_base_url: str,
    top_k: int,
    chunk_size: int,
    chunk_overlap: int,
    generate_answer: bool = True,
    system_prompt: str = "",
    progress_callback: Callable[[str], None] | None = None,
) -> dict:
    timings: list[dict] = []

    def measure(step_name: str, fn):
        start = time.perf_counter()
        result = fn()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        timings.append({"step": step_name, "ms": round(elapsed_ms, 1)})
        return result

    _emit(
        progress_callback,
        f"Query gestartet (store={persist_dir}, Modell={ollama_model}, top_k={top_k})",
    )
    measure("Ollama-Verbindung prüfen", lambda: check_ollama_connection(ollama_base_url))

    sources = measure(
        "Retrieval",
        lambda: retrieve_contexts(
            question=question,
            persist_dir=persist_dir,
            embedding_model=embedding_model,
            top_k=top_k,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            progress_callback=progress_callback,
        ),
    )
    prompt = measure(
        "Prompt erstellen",
        lambda: build_teaching_prompt_with_system(
            question=question,
            sources=sources,
            system_prompt=system_prompt,
        ),
    )

    answer = ""
    if generate_answer:
        _emit(progress_callback, "Initialisiere LLM...")
        llm = measure("LLM initialisieren", lambda: get_ollama_llm(ollama_model, ollama_base_url))
        _emit(progress_callback, "Generiere Antwort...")
        response = measure("Antwort generieren", lambda: llm.complete(prompt))
        answer = str(response).strip()
        _emit(progress_callback, "Antwortgenerierung abgeschlossen.")
    else:
        _emit(progress_callback, "Nur Retrieval-Modus: keine Antwortgenerierung.")

    _emit(progress_callback, f"Query abgeschlossen, {len(sources)} Quellen gefunden.")
    return {
        "sources": sources,
        "prompt": prompt,
        "answer": answer,
        "timings": timings,
    }


def run_llm_only_query(
    question: str,
    ollama_model: str,
    ollama_base_url: str,
    system_prompt: str = "",
    progress_callback: Callable[[str], None] | None = None,
) -> dict:
    timings: list[dict] = []

    def measure(step_name: str, fn):
        start = time.perf_counter()
        result = fn()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        timings.append({"step": step_name, "ms": round(elapsed_ms, 1)})
        return result

    _emit(progress_callback, f"LLM-only gestartet (Modell={ollama_model})")
    measure("Ollama-Verbindung prüfen", lambda: check_ollama_connection(ollama_base_url))
    prompt = measure(
        "Prompt erstellen",
        lambda: (
            f"System:\n{system_prompt.strip()}\n\nFrage:\n{question}"
            if system_prompt.strip()
            else question
        ),
    )
    _emit(progress_callback, "Initialisiere LLM...")
    llm = measure("LLM initialisieren", lambda: get_ollama_llm(ollama_model, ollama_base_url))
    _emit(progress_callback, "Generiere Antwort...")
    response = measure("Antwort generieren", lambda: llm.complete(prompt))
    _emit(progress_callback, "LLM-only abgeschlossen.")

    return {
        "sources": [],
        "prompt": prompt,
        "answer": str(response).strip(),
        "timings": timings,
    }


def preview_chunks(
    input_dir: Path,
    chunk_size: int,
    chunk_overlap: int,
    max_chunks: int = 8,
) -> list[dict]:
    in_dir = input_dir.expanduser()
    if not in_dir.exists():
        raise FileNotFoundError(f"Input-Verzeichnis nicht gefunden: {in_dir}")

    documents = SimpleDirectoryReader(
        input_dir=str(in_dir),
        recursive=True,
        required_exts=[".txt", ".md"],
    ).load_data()
    if not documents:
        raise ValueError(f"Keine .txt/.md Dateien in {in_dir} gefunden.")

    parser = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    nodes = parser.get_nodes_from_documents(documents)

    preview: list[dict] = []
    for idx, node in enumerate(nodes[:max_chunks], start=1):
        meta = node.metadata or {}
        preview.append(
            {
                "idx": idx,
                "source": str(meta.get("file_path", meta.get("filename", "unknown"))),
                "text": node.get_content(),
                "chars": len(node.get_content()),
            }
        )
    return preview
