# Einfaches lokales RAG mit Ollama

Dieses Projekt nutzt `llama-index` + `Ollama` für lokales RAG.
Embeddings laufen über HuggingFace (`sentence-transformers/paraphrase-multilingual-mpnet-base-v2` standardmäßig, gut für deutsche Texte), Antworten über ein lokales Ollama-Modell (`llama3` standardmäßig).

## Voraussetzungen
- Python 3.10+
- Ollama installiert
- Modell lokal verfügbar (z. B. `llama3`)

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Ollama vorbereiten
```bash
ollama serve
ollama pull llama3
ollama list
```
Bei bedarf auch das quantisierte Modell llama3:8b-instruct-q4_K_M pullen.
## Dokumente ablegen
Lege `.txt` und `.md` Dateien in `data/documents/`.

## Index bauen (CLI)
```bash
python -m rag.ingest --input-dir data/documents --store-dir rag_store
```
Das Embedding-Modell wird hier festgelegt und in `rag_store/rag_settings.json` gespeichert.

## Fragen stellen (CLI)
```bash
python -m rag.query "Worum geht es im Dokument?" --ollama-model llama3
```
Falls Ollama nicht auf Default-Port/Host läuft:
```bash
python -m rag.query "Worum geht es im Dokument?" --ollama-model llama3 --ollama-base-url http://127.0.0.1:11434
```

## Gradio UI starten
```bash
python app.py
```
Dann im Browser öffnen: `http://127.0.0.1:7860`

Tabs:
- `Index bauen`: Dateien einlesen und persistenten llama-index Store erstellen
- `Fragen`: Query gegen den Store, Antwort via Ollama
- Beide Tabs enthalten ein `Live-Log` Feld und Fortschrittsanzeige für den aktuellen Lauf.
- `Index bauen`: zusätzliche `Chunking-Vorschau` zur didaktischen Kontrolle von `chunk_size` und `chunk_overlap`.
- `Fragen`: `Pipeline-Ansicht`, `Retrieval-only Vergleich` und sichtbarer `Prompt an das LLM`.

## Logging
- Konsole: Laufende Logs während App/CLI Ausführung
- Datei: `logs/rag.log` (rotierend, max. 3 Backups)
