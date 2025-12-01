# PropertyRAG

RAG-System für Immobilien-Dokumentenanalyse. Extrahiert strukturierte Daten aus PDFs und ermöglicht natürlichsprachliche Abfragen.

## Features

- **PDF-Ingestion**: Upload von Mietverträgen, Gutachten, Grundbuchauszügen, Nebenkostenabrechnungen
- **Auto-Klassifikation**: Automatische Erkennung des Dokumenttyps
- **Strukturierte Extraktion**: Extrahiert domänenspezifische Felder (Miete, Parteien, Werte, etc.)
- **RAG-Queries**: Natürlichsprachliche Fragen mit Quellenangaben
- **Projekt-Organisation**: Dokumente in Projekten gruppieren

## Tech Stack

- Python 3.11+
- FastAPI
- PostgreSQL + pgvector
- OpenAI (Embeddings + GPT-4)
- SQLAlchemy (async)
- Pydantic v2

## Schnellstart

### Mit Docker (empfohlen)

```bash
# .env erstellen
cp .env.example .env
# OPENAI_API_KEY in .env setzen

# Starten
docker-compose up -d

# Migrations ausführen
docker-compose run migrate

# API verfügbar unter http://localhost:8000
```

### Lokale Entwicklung

```bash
# Venv erstellen
python -m venv venv
source venv/bin/activate  # Linux/Mac
# oder: venv\Scripts\activate  # Windows

# Dependencies installieren
pip install -e ".[dev]"

# PostgreSQL mit pgvector starten
docker-compose up -d db

# .env konfigurieren
cp .env.example .env
# OPENAI_API_KEY und DATABASE_URL setzen

# Migrations
alembic upgrade head

# Server starten
python -m propertyrag
# oder: uvicorn propertyrag.api.app:app --reload
```

## API Endpoints

### Dokumente

```bash
# PDF hochladen
curl -X POST http://localhost:8000/api/v1/documents/upload \
  -F "file=@mietvertrag.pdf" \
  -F "document_type=mietvertrag"

# Dokumente auflisten
curl http://localhost:8000/api/v1/documents

# Extrahierte Daten abrufen
curl http://localhost:8000/api/v1/documents/{id}/extracted
```

### Abfragen

```bash
# RAG-Query
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Wie hoch ist die monatliche Miete?"}'
```

### Projekte

```bash
# Projekt erstellen
curl -X POST http://localhost:8000/api/v1/projects \
  -H "Content-Type: application/json" \
  -d '{"name": "Immobilie Berlin", "description": "Dokumente für Objekt Berlin"}'
```

## Dokumenttypen

| Typ | Extrahierte Felder |
|-----|-------------------|
| `mietvertrag` | Parteien, Objekt, Miete, Nebenkosten, Laufzeit, Kündigungsfrist, Kaution |
| `gutachten` | Verkehrswert, Ertragswert, Sachwert, Gutachter, Baujahr, Flächen |
| `grundbuchauszug` | Eigentümer, Belastungen, Flurnummer, Grundstücksgröße |
| `nebenkostenabrechnung` | Zeitraum, Positionen, Umlageschlüssel, Nachzahlung/Guthaben |

## Konfiguration

Umgebungsvariablen (`.env`):

```env
# Pflicht
OPENAI_API_KEY=sk-...
DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/propertyrag

# Optional
DEBUG=false
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_CHAT_MODEL=gpt-4o
CHUNK_SIZE=512
CHUNK_OVERLAP=50
RETRIEVAL_TOP_K=5
```

## Projektstruktur

```
src/propertyrag/
├── api/                 # FastAPI Endpoints
│   ├── app.py
│   ├── routes/
│   └── schemas.py
├── core/                # Konfiguration, Modelle
│   ├── config.py
│   ├── models.py
│   └── logging.py
├── db/                  # Datenbank
│   ├── models.py
│   ├── repository.py
│   └── session.py
└── services/            # Business Logic
    ├── pdf_parser.py
    ├── chunker.py
    ├── embedder.py
    ├── classifier.py
    ├── extractor.py
    ├── retriever.py
    ├── rag.py
    └── ingestion.py
```

## Tests

```bash
# Tests ausführen
pytest

# Mit Coverage
pytest --cov=propertyrag

# Nur Unit Tests
pytest tests/test_chunker.py tests/test_embedder.py

# Nur API Tests
pytest tests/test_api.py
```

## Architektur-Entscheidungen

1. **pgvector statt ChromaDB**: Produktionsreif, ACID-konform, skalierbar
2. **pdfplumber statt PyPDF2**: Bessere Tabellenextraktion
3. **Tiktoken für Chunking**: Konsistent mit OpenAI-Tokenisierung
4. **OpenAI JSON Mode**: Strukturierte Outputs ohne Function Calling Overhead
5. **Repository Pattern**: Saubere Trennung DB ↔ Business Logic
6. **Async durchgängig**: asyncpg, AsyncSession, AsyncOpenAI

## Lizenz

MIT
