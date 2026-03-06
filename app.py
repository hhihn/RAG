from __future__ import annotations

import html
import json
from datetime import datetime
from pathlib import Path

import gradio as gr

from rag.config import DEFAULT_CONFIG
from rag.logging_config import get_logger
from rag.ollama_rag import (
    build_index,
    preview_chunks,
    read_settings,
    resolve_index_settings,
    run_llm_only_query,
    run_rag_query,
)

CSS_PATH = Path(__file__).resolve().parent / "styles" / "gradio-modern.css"
CSS_TEXT = CSS_PATH.read_text(encoding="utf-8") if CSS_PATH.exists() else ""
LOGO_PATH = Path(__file__).resolve().parent / "data" / "haveltechlogo.png"
logger = get_logger(__name__)


def format_sources_html(sources: list[dict]) -> str:
    if not sources:
        return "<div class='context-empty'>Keine Kontexte gefunden.</div>"

    cards: list[str] = ["<div class='context-list'>"]
    for idx, src in enumerate(sources, start=1):
        score = float(src.get("score", 0.0))
        raw_source = str(src.get("source", "unknown"))
        marker = "data/documents"
        marker_idx = raw_source.find(marker)
        source_display = raw_source[marker_idx:] if marker_idx >= 0 else raw_source
        source = html.escape(source_display)
        text = html.escape(str(src.get("text", "")))
        cards.append(
            (
                "<details class='context-card'>"
                "<summary>"
                f"<span class='context-title'>Kontext {idx}</span>"
                f"<span class='context-meta'>Ähnlichkeit: {score:.4f}</span>"
                f"<span class='context-doc'>Dokument: {source}</span>"
                "</summary>"
                f"<div class='context-content'><pre>{text}</pre></div>"
                "</details>"
            )
        )
    cards.append("</div>")
    return "".join(cards)


def format_pipeline_html(timings: list[dict], mode: str) -> str:
    if mode == "LLM-Antwort ohne Retrieval":
        expected_steps = ["Ollama-Verbindung prüfen", "Prompt erstellen", "LLM initialisieren", "Antwort generieren"]
    else:
        expected_steps = [
            "Ollama-Verbindung prüfen",
            "Retrieval",
            "Prompt erstellen",
        ]
        if mode == "Retrieval + RAG-Antwort":
            expected_steps.extend(["LLM initialisieren", "Antwort generieren"])

    timing_map = {item["step"]: item["ms"] for item in timings}
    lines = ["<div class='pipeline-box'><h4>Pipeline-Ansicht</h4><ol>"]
    for step in expected_steps:
        ms = timing_map.get(step)
        if ms is None:
            lines.append(f"<li><span class='pipe-step'>{html.escape(step)}</span> <span class='pipe-miss'>–</span></li>")
        else:
            lines.append(
                f"<li><span class='pipe-step'>{html.escape(step)}</span> "
                f"<span class='pipe-time'>{ms:.1f} ms</span></li>"
            )
    lines.append("</ol></div>")
    return "".join(lines)


def format_chunk_preview_html(chunks: list[dict]) -> str:
    if not chunks:
        return "<div class='context-empty'>Keine Chunks in der Vorschau.</div>"
    rows = ["<div class='chunk-preview'><h4>Chunking-Vorschau</h4>"]
    for chunk in chunks:
        source = str(chunk["source"])
        marker = "data/documents"
        marker_idx = source.find(marker)
        source_display = source[marker_idx:] if marker_idx >= 0 else source
        text = html.escape(chunk["text"])
        rows.append(
            "<details class='context-card'>"
            "<summary>"
            f"<span class='context-title'>Chunk {chunk['idx']}</span>"
            f"<span class='context-meta'>Länge: {chunk['chars']} Zeichen (bei token-basiertem Split)</span>"
            f"<span class='context-doc'>Dokument: {html.escape(source_display)}</span>"
            "</summary>"
            f"<div class='context-content'><pre>{text}</pre></div>"
            "</details>"
        )
    rows.append("</div>")
    return "".join(rows)


def build_index_ui(
    input_dir: str,
    store_dir: str,
    embedding_model: str,
    chunk_size: float,
    chunk_overlap: float,
    progress=gr.Progress(track_tqdm=False),
) -> tuple[str, str, str]:
    events: list[str] = []

    def push(message: str, fraction: float | None = None) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{stamp}] {message}"
        events.append(line)
        if fraction is not None:
            progress(fraction, desc=message)

    def push_auto(message: str) -> None:
        lowered = message.lower()
        fraction = None
        if "initialisiere embedding" in lowered:
            fraction = 0.2
        elif "lade dokumente" in lowered:
            fraction = 0.35
        elif "erzeuge vektorindex" in lowered:
            fraction = 0.65
        elif "speichere index" in lowered:
            fraction = 0.85
        elif "abgeschlossen" in lowered:
            fraction = 1.0
        push(message, fraction)

    logger.info("UI Aktion: Indexbau angefordert (input_dir=%s, store_dir=%s)", input_dir, store_dir)
    push("Indexbau gestartet", 0.05)
    try:
        meta = build_index(
            input_dir=Path(input_dir),
            persist_dir=Path(store_dir),
            embedding_model=embedding_model,
            chunk_size=int(chunk_size),
            chunk_overlap=int(chunk_overlap),
            progress_callback=push_auto,
        )
    except Exception as exc:
        logger.exception("UI Indexbau fehlgeschlagen")
        push(f"Fehler: {exc}", 1.0)
        return f"Fehler beim Indexbau: {exc}", "\n".join(events), ""

    logger.info("UI Indexbau erfolgreich: %s", meta)
    push("Indexbau erfolgreich abgeschlossen", 1.0)
    return (
        f"Index erstellt: {meta['num_documents']} Dokumente\n"
        f"embedding={meta['embedding_model']}\n"
        f"chunk_size={meta['chunk_size']} | chunk_overlap={meta['chunk_overlap']}\n"
        f"store={Path(store_dir).expanduser()}",
        "\n".join(events),
        "",
    )


def preview_chunks_ui(
    input_dir: str,
    chunk_size: float,
    chunk_overlap: float,
) -> str:
    try:
        chunks = preview_chunks(
            input_dir=Path(input_dir),
            chunk_size=int(chunk_size),
            chunk_overlap=int(chunk_overlap),
            max_chunks=10,
        )
    except Exception as exc:
        logger.exception("Chunking-Vorschau fehlgeschlagen")
        return f"<div class='context-empty'>Fehler bei der Vorschau: {html.escape(str(exc))}</div>"
    return format_chunk_preview_html(chunks)


def ask_ui(
    question: str,
    store_dir: str,
    top_k: float,
    ollama_model: str,
    ollama_base_url: str,
    system_prompt: str,
    mode: str,
    progress=gr.Progress(track_tqdm=False),
) -> tuple[str, str, str, str, str]:
    events: list[str] = []

    def push(message: str, fraction: float | None = None) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{stamp}] {message}"
        events.append(line)
        if fraction is not None:
            progress(fraction, desc=message)

    def push_auto(message: str) -> None:
        lowered = message.lower()
        fraction = None
        if "prüfe ollama-verbindung" in lowered:
            fraction = 0.2
        elif "lade lokalen index" in lowered:
            fraction = 0.45
        elif "initialisiere query engine" in lowered:
            fraction = 0.6
        elif "initialisiere llm" in lowered:
            fraction = 0.7
        elif "retrieval + generierung" in lowered:
            fraction = 0.8
        elif "generiere antwort" in lowered:
            fraction = 0.9
        elif "abgeschlossen" in lowered:
            fraction = 1.0
        push(message, fraction)

    if not question.strip():
        logger.warning("UI Query ohne Frage erhalten")
        push("Abbruch: Frage fehlt", 1.0)
        return "Bitte eine Frage eingeben.", "", "\n".join(events), "", ""

    store = Path(store_dir).expanduser()
    push("Lese Index-Einstellungen", 0.05)
    settings = resolve_index_settings(
        persist_dir=store,
        default_embedding_model=DEFAULT_CONFIG.embedding_model,
        default_chunk_size=DEFAULT_CONFIG.chunk_size,
        default_chunk_overlap=DEFAULT_CONFIG.chunk_overlap,
    )

    try:
        logger.info(
            "UI Query angefordert (store=%s, top_k=%s, ollama_model=%s, base_url=%s, mode=%s)",
            store,
            int(top_k),
            ollama_model,
            ollama_base_url,
            mode,
        )
        push("Starte Query", 0.1)
        if mode == "LLM-Antwort ohne Retrieval":
            details = run_llm_only_query(
                question=question,
                ollama_model=ollama_model,
                ollama_base_url=ollama_base_url,
                system_prompt=system_prompt,
                progress_callback=push_auto,
            )
        else:
            details = run_rag_query(
                question=question,
                persist_dir=store,
                embedding_model=settings["embedding_model"],
                ollama_model=ollama_model,
                ollama_base_url=ollama_base_url,
                top_k=int(top_k),
                chunk_size=settings["chunk_size"],
                chunk_overlap=settings["chunk_overlap"],
                generate_answer=(mode == "Retrieval + RAG-Antwort"),
                system_prompt=system_prompt,
                progress_callback=push_auto,
            )
    except Exception as exc:
        logger.exception("UI Query fehlgeschlagen")
        push(f"Fehler: {exc}", 1.0)
        return f"Fehler bei Query: {exc}", "", "\n".join(events), "", ""

    sources = details["sources"]
    answer = details["answer"]
    prompt = details["prompt"]
    pipeline_html = format_pipeline_html(details["timings"], mode=mode)
    logger.info("UI Query erfolgreich: %s Quellen", len(sources))
    push(f"Fertig: {len(sources)} Quellen", 1.0)
    return format_sources_html(sources), answer, "\n".join(events), prompt, pipeline_html


def show_store_config(store_dir: str) -> str:
    logger.info("UI Aktion: Store-Config anzeigen (store_dir=%s)", store_dir)
    cfg = read_settings(Path(store_dir))
    if not cfg:
        return "Noch kein Index gefunden."
    return json.dumps(cfg, ensure_ascii=False, indent=2)


def create_app() -> gr.Blocks:
    with gr.Blocks(title="Lokales RAG mit Ollama", css=CSS_TEXT) as demo:
        with gr.Column(elem_classes=["app-shell"]):
            with gr.Column(elem_classes=["hero"]):
                with gr.Row(elem_classes=["hero-row"]):
                    with gr.Column(scale=5, elem_classes=["hero-copy"]):
                        gr.Markdown(
                            """
                            # Lokales RAG mit Ollama
                            Lokale Wissenssuche für deutsche Texte mit `llama-index` und `Ollama`.

                            Reihenfolge:
                            1. Ollama starten und Modell laden (`ollama serve`, `ollama pull llama3`)
                            2. Dokumente in `data/documents/` ablegen
                            3. Index bauen, dann Fragen stellen
                            """
                        )
                    if LOGO_PATH.exists():
                        with gr.Column(scale=2, min_width=220, elem_classes=["hero-logo-col"]):
                            gr.Image(
                                value=str(LOGO_PATH),
                                show_label=False,
                                interactive=False,
                                container=False,
                                elem_classes=["hero-logo"],
                            )

            with gr.Column(elem_classes=["panel"]):
                with gr.Tabs():
                    with gr.Tab("Index bauen"):
                        with gr.Column(elem_classes=["input-card"]):
                            gr.Markdown("### Index Einstellungen", elem_classes=["section-title"])

                            input_dir = gr.Textbox(
                                value="data/documents",
                                label="Input-Verzeichnis",
                                info="Pfad zu deinen .txt/.md Dateien. Unterordner werden rekursiv eingelesen.",
                            )
                            store_dir_ingest = gr.Textbox(
                                value=str(DEFAULT_CONFIG.store_dir),
                                label="Store-Verzeichnis",
                                info="Zielordner für den persistenten Index und rag_settings.json.",
                            )
                            embedding_model_ingest = gr.Textbox(
                                value=DEFAULT_CONFIG.embedding_model,
                                label="Embedding-Modell",
                                info="Empfohlen für Deutsch: sentence-transformers/paraphrase-multilingual-mpnet-base-v2.",
                            )
                            chunk_size = gr.Number(
                                value=DEFAULT_CONFIG.chunk_size,
                                label="Chunk Size",
                                precision=0,
                                info="Anzahl Tokens pro Segment (nicht Zeichen). Größer = mehr Kontext je Treffer, kleiner = präziseres Retrieval.",
                            )
                            chunk_overlap = gr.Number(
                                value=DEFAULT_CONFIG.chunk_overlap,
                                label="Chunk Overlap",
                                precision=0,
                                info="Token-Überlappung zwischen Segmenten. 80-160 ist meist ein guter Start für Fließtext.",
                            )

                            ingest_btn = gr.Button("Index erstellen", variant="primary")
                            ingest_status = gr.Textbox(label="Status", lines=5, elem_classes=["status-box"])
                            ingest_live_log = gr.Textbox(
                                label="Live-Log",
                                lines=8,
                                info="Zeigt den aktuellen Ablauf während des Indexbaus.",
                                elem_classes=["result-box"],
                            )
                            chunk_preview = gr.HTML(
                                label="Chunking-Vorschau",
                                value="<div class='context-empty'>Klicke auf „Chunking-Vorschau“, um Chunks vor dem Indexbau zu sehen.</div>",
                                elem_classes=["context-output"],
                            )
                            preview_btn = gr.Button("Chunking-Vorschau")

                            ingest_btn.click(
                                fn=build_index_ui,
                                inputs=[input_dir, store_dir_ingest, embedding_model_ingest, chunk_size, chunk_overlap],
                                outputs=[ingest_status, ingest_live_log, chunk_preview],
                            )
                            preview_btn.click(
                                fn=preview_chunks_ui,
                                inputs=[input_dir, chunk_size, chunk_overlap],
                                outputs=[chunk_preview],
                            )

                    with gr.Tab("Fragen"):
                        with gr.Column(elem_classes=["input-card"]):
                            gr.Markdown("### Query Einstellungen", elem_classes=["section-title"])

                            question = gr.Textbox(
                                label="Frage",
                                lines=3,
                                placeholder="Beispiel: Welche Kernaussagen zum Thema stehen in den Dokumenten?",
                                info="Formuliere konkrete Fragen für bessere Treffer.",
                            )
                            store_dir_query = gr.Textbox(
                                value=str(DEFAULT_CONFIG.store_dir),
                                label="Store-Verzeichnis",
                                info="Muss auf den zuvor gebauten Index zeigen.",
                            )
                            top_k = gr.Number(
                                value=DEFAULT_CONFIG.top_k,
                                label="Top-K Treffer",
                                precision=0,
                                minimum=1,
                                maximum=20,
                                info="Anzahl gefundener Kontexte für die Antwort. 3-6 ist ein guter Bereich.",
                            )
                            ollama_model = gr.Textbox(
                                value=DEFAULT_CONFIG.ollama_model,
                                label="Ollama Modell",
                                info="Lokales LLM in Ollama, z. B. llama3, mistral oder gemma.",
                            )
                            ollama_model_preset = gr.Dropdown(
                                choices=["llama3", "llama3:8b-instruct-q4_K_M"],
                                value=DEFAULT_CONFIG.ollama_model
                                if DEFAULT_CONFIG.ollama_model in {"llama3", "llama3:8b-instruct-q4_K_M"}
                                else "llama3",
                                label="Modell-Voreinstellung",
                                info="Schnell zwischen Standard und quantisierter Variante wechseln.",
                            )
                            ollama_base_url = gr.Textbox(
                                value=DEFAULT_CONFIG.ollama_base_url,
                                label="Ollama Base URL",
                                info="Adresse deines laufenden Ollama-Servers. Standard lokal: http://127.0.0.1:11434.",
                            )
                            system_prompt = gr.Textbox(
                                value=DEFAULT_CONFIG.system_prompt,
                                label="System-Prompt",
                                lines=4,
                                info="Bearbeitbarer System-Kontext. Wird bei der Anfrage an das Modell gesendet.",
                            )
                            mode = gr.Radio(
                                choices=["Retrieval-only", "Retrieval + RAG-Antwort", "LLM-Antwort ohne Retrieval"],
                                value="Retrieval + RAG-Antwort",
                                label="Modus",
                                info="Vergleiche reines Retrieval, RAG und reine LLM-Antwort ohne Retrieval.",
                            )

                            ask_btn = gr.Button("Frage senden", variant="primary")
                            pipeline_view = gr.HTML(
                                label="Pipeline-Ansicht",
                                value="<div class='context-empty'>Noch keine Query ausgeführt.</div>",
                                elem_classes=["context-output"],
                            )
                            retrieved = gr.HTML(label="Gefundene Kontexte", elem_classes=["context-output"])
                            answer = gr.Textbox(label="RAG-Antwort", lines=8, elem_classes=["answer-box"])
                            with gr.Accordion("Gesendeter Prompt an das LLM", open=False):
                                prompt_box = gr.Code(
                                    label="Prompt an das LLM",
                                    language="markdown",
                                    interactive=False,
                                    elem_classes=["code-box"],
                                )
                            query_live_log = gr.Textbox(
                                label="Live-Log",
                                lines=8,
                                info="Zeigt den aktuellen Ablauf während Retrieval und Antwortgenerierung.",
                                elem_classes=["result-box"],
                            )

                            ask_btn.click(
                                fn=ask_ui,
                                inputs=[
                                    question,
                                    store_dir_query,
                                    top_k,
                                    ollama_model,
                                    ollama_base_url,
                                    system_prompt,
                                    mode,
                                ],
                                outputs=[retrieved, answer, query_live_log, prompt_box, pipeline_view],
                            )
                            ollama_model_preset.change(
                                fn=lambda v: v,
                                inputs=[ollama_model_preset],
                                outputs=[ollama_model],
                            )

                            gr.Markdown("Kurzhilfe: Erst `Index bauen`, dann im selben Store fragen.", elem_classes=["settings-note"])
                            store_cfg = gr.Code(label="Store Config", language="json", interactive=False, elem_classes=["code-box"])
                            refresh_cfg_btn = gr.Button("Store-Config anzeigen")
                            refresh_cfg_btn.click(fn=show_store_config, inputs=[store_dir_query], outputs=[store_cfg])

    return demo


if __name__ == "__main__":
    app = create_app()
    app.launch(server_name="127.0.0.1", server_port=7860)
