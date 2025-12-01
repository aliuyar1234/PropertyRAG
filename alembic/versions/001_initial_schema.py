"""Initial schema with pgvector.

Revision ID: 001
Revises:
Create Date: 2024-01-01
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Enable pgvector extension
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # Create document_type enum
    op.execute(
        """
        CREATE TYPE documenttype AS ENUM (
            'mietvertrag', 'gutachten', 'grundbuchauszug',
            'nebenkostenabrechnung', 'unknown'
        )
        """
    )

    # Create processing_status enum
    op.execute(
        """
        CREATE TYPE processingstatus AS ENUM (
            'pending', 'processing', 'completed', 'failed'
        )
        """
    )

    # Create projects table
    op.create_table(
        "projects",
        sa.Column("id", sa.UUID(), primary_key=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
    )

    # Create documents table
    op.create_table(
        "documents",
        sa.Column("id", sa.UUID(), primary_key=True),
        sa.Column("filename", sa.String(500), nullable=False),
        sa.Column(
            "document_type",
            sa.Enum(
                "mietvertrag",
                "gutachten",
                "grundbuchauszug",
                "nebenkostenabrechnung",
                "unknown",
                name="documenttype",
                create_type=False,
            ),
            default="unknown",
        ),
        sa.Column(
            "status",
            sa.Enum(
                "pending",
                "processing",
                "completed",
                "failed",
                name="processingstatus",
                create_type=False,
            ),
            default="pending",
        ),
        sa.Column("page_count", sa.Integer(), nullable=True),
        sa.Column(
            "project_id",
            sa.UUID(),
            sa.ForeignKey("projects.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
    )
    op.create_index("ix_documents_project_id", "documents", ["project_id"])
    op.create_index("ix_documents_document_type", "documents", ["document_type"])
    op.create_index("ix_documents_status", "documents", ["status"])

    # Create chunks table
    op.create_table(
        "chunks",
        sa.Column("id", sa.UUID(), primary_key=True),
        sa.Column(
            "document_id",
            sa.UUID(),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("page_number", sa.Integer(), nullable=True),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=False),
        sa.Column("embedding", Vector(1536), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
    )
    op.create_index("ix_chunks_document_id", "chunks", ["document_id"])

    # Create IVFFlat index for vector similarity search
    op.execute(
        """
        CREATE INDEX ix_chunks_embedding ON chunks
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100)
        """
    )

    # Create extracted_data table
    op.create_table(
        "extracted_data",
        sa.Column("id", sa.UUID(), primary_key=True),
        sa.Column(
            "document_id",
            sa.UUID(),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
            unique=True,
        ),
        sa.Column(
            "document_type",
            sa.Enum(
                "mietvertrag",
                "gutachten",
                "grundbuchauszug",
                "nebenkostenabrechnung",
                "unknown",
                name="documenttype",
                create_type=False,
            ),
            nullable=False,
        ),
        sa.Column("data", sa.dialects.postgresql.JSONB(), nullable=False),
        sa.Column("extraction_confidence", sa.Float(), nullable=True),
        sa.Column(
            "extracted_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
    )
    op.create_index(
        "ix_extracted_data_document_type", "extracted_data", ["document_type"]
    )
    op.execute(
        "CREATE INDEX ix_extracted_data_data ON extracted_data USING gin (data)"
    )


def downgrade() -> None:
    op.drop_table("extracted_data")
    op.drop_table("chunks")
    op.drop_table("documents")
    op.drop_table("projects")
    op.execute("DROP TYPE processingstatus")
    op.execute("DROP TYPE documenttype")
    op.execute("DROP EXTENSION IF EXISTS vector")
