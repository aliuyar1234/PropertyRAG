"""Domain models for PropertyRAG."""

from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Annotated
from uuid import UUID

from pydantic import BaseModel, Field


class DocumentType(str, Enum):
    """Types of real estate documents."""

    MIETVERTRAG = "mietvertrag"
    GUTACHTEN = "gutachten"
    GRUNDBUCHAUSZUG = "grundbuchauszug"
    NEBENKOSTENABRECHNUNG = "nebenkostenabrechnung"
    UNKNOWN = "unknown"


class ProcessingStatus(str, Enum):
    """Document processing status."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# ============================================================================
# Base Models
# ============================================================================


class DocumentBase(BaseModel):
    """Base document model."""

    filename: str
    document_type: DocumentType = DocumentType.UNKNOWN
    project_id: UUID | None = None


class DocumentCreate(DocumentBase):
    """Model for creating a document."""

    pass


class Document(DocumentBase):
    """Full document model with all fields."""

    id: UUID
    status: ProcessingStatus = ProcessingStatus.PENDING
    page_count: int | None = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# ============================================================================
# Chunk Models
# ============================================================================


class ChunkBase(BaseModel):
    """Base chunk model."""

    content: str
    page_number: int | None = None
    chunk_index: int


class Chunk(ChunkBase):
    """Full chunk model."""

    id: UUID
    document_id: UUID
    token_count: int
    created_at: datetime

    class Config:
        from_attributes = True


class ChunkWithScore(Chunk):
    """Chunk with similarity score for retrieval results."""

    score: float


# ============================================================================
# Extracted Data Models - Domain Specific
# ============================================================================


class Partei(BaseModel):
    """Party in a contract (landlord/tenant)."""

    name: str
    adresse: str | None = None
    typ: str | None = None  # "Vermieter", "Mieter"


class MietvertragData(BaseModel):
    """Extracted data from a rental contract."""

    vermieter: Partei | None = None
    mieter: Partei | None = None
    objekt_adresse: str | None = None
    objekt_typ: str | None = None  # "Wohnung", "Gewerbe", etc.
    flaeche_qm: Decimal | None = None
    nettomiete_eur: Decimal | None = None
    nebenkosten_eur: Decimal | None = None
    bruttomiete_eur: Decimal | None = None
    mietbeginn: date | None = None
    mietende: date | None = None
    befristet: bool | None = None
    kuendigungsfrist_monate: int | None = None
    indexierung: str | None = None
    kaution_eur: Decimal | None = None
    sondervereinbarungen: list[str] = Field(default_factory=list)


class GutachtenData(BaseModel):
    """Extracted data from a property valuation report."""

    gutachter: str | None = None
    bewertungsstichtag: date | None = None
    verkehrswert_eur: Decimal | None = None
    ertragswert_eur: Decimal | None = None
    sachwert_eur: Decimal | None = None
    nutzungsart: str | None = None
    baujahr: int | None = None
    wohnflaeche_qm: Decimal | None = None
    grundstuecksflaeche_qm: Decimal | None = None
    adresse: str | None = None


class Belastung(BaseModel):
    """Encumbrance on a property."""

    typ: str  # "Hypothek", "Grundschuld", "Dienstbarkeit"
    betrag_eur: Decimal | None = None
    glaeubiger: str | None = None
    beschreibung: str | None = None


class GrundbuchauszugData(BaseModel):
    """Extracted data from a land register extract."""

    grundbuchamt: str | None = None
    blatt_nummer: str | None = None
    flurnummer: str | None = None
    gemarkung: str | None = None
    grundstuecksgroesse_qm: Decimal | None = None
    eigentuemer: list[str] = Field(default_factory=list)
    belastungen: list[Belastung] = Field(default_factory=list)
    stand_datum: date | None = None


class NebenkostenPosition(BaseModel):
    """Single position in utility bill."""

    bezeichnung: str
    betrag_eur: Decimal
    umlageschluessel: str | None = None


class NebenkostenabrechnungData(BaseModel):
    """Extracted data from a utility bill."""

    abrechnungszeitraum_von: date | None = None
    abrechnungszeitraum_bis: date | None = None
    objekt_adresse: str | None = None
    mieter: str | None = None
    gesamtkosten_eur: Decimal | None = None
    vorauszahlungen_eur: Decimal | None = None
    nachzahlung_eur: Decimal | None = None
    guthaben_eur: Decimal | None = None
    positionen: list[NebenkostenPosition] = Field(default_factory=list)


# Union type for extracted data
ExtractedData = (
    MietvertragData | GutachtenData | GrundbuchauszugData | NebenkostenabrechnungData
)


class ExtractedDocument(BaseModel):
    """Document with extracted structured data."""

    document_id: UUID
    document_type: DocumentType
    data: ExtractedData
    extraction_confidence: float | None = None
    extracted_at: datetime

    class Config:
        from_attributes = True


# ============================================================================
# Query Models
# ============================================================================


class QueryRequest(BaseModel):
    """RAG query request."""

    question: str
    project_id: UUID | None = None
    document_ids: list[UUID] | None = None
    top_k: int = 5


class Source(BaseModel):
    """Source reference for an answer."""

    document_id: UUID
    filename: str
    page_number: int | None = None
    chunk_content: str
    score: float


class QueryResponse(BaseModel):
    """RAG query response."""

    answer: str
    sources: list[Source]
    query: str


# ============================================================================
# Project Models
# ============================================================================


class ProjectBase(BaseModel):
    """Base project model."""

    name: str
    description: str | None = None


class ProjectCreate(ProjectBase):
    """Model for creating a project."""

    pass


class Project(ProjectBase):
    """Full project model."""

    id: UUID
    created_at: datetime
    updated_at: datetime
    document_count: int = 0

    class Config:
        from_attributes = True
