"""create documents and document_versions

Revision ID: 0002_create_documents_versions
Revises: 0001_create_dialog_tables
Create Date: 2025-12-22
"""

from alembic import op
import sqlalchemy as sa

revision = "0002_create_documents_versions"
down_revision = "0001_create_dialog_tables"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "documents",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("slug", sa.String(length=64), nullable=False),
        sa.Column("title", sa.String(length=256), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
    )
    op.create_index("ix_documents_slug", "documents", ["slug"], unique=True)

    op.create_table(
        "document_versions",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("document_id", sa.Integer(), sa.ForeignKey("documents.id"), nullable=False),
        sa.Column("year", sa.Integer(), nullable=True),
        sa.Column("version_label", sa.String(length=128), nullable=True),
        sa.Column("file_path", sa.Text(), nullable=False),
        sa.Column("qdrant_collection", sa.String(length=128), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("status", sa.String(length=16), nullable=False, server_default=sa.text("'indexing'")),
        sa.Column("pages", sa.Integer(), nullable=True),
        sa.Column("chunks", sa.Integer(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
    )
    op.create_index("ix_document_versions_document_id", "document_versions", ["document_id"])
    op.create_index("ix_document_versions_is_active", "document_versions", ["is_active"])
    op.create_index("ix_document_versions_collection", "document_versions", ["qdrant_collection"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_document_versions_collection", table_name="document_versions")
    op.drop_index("ix_document_versions_is_active", table_name="document_versions")
    op.drop_index("ix_document_versions_document_id", table_name="document_versions")
    op.drop_table("document_versions")

    op.drop_index("ix_documents_slug", table_name="documents")
    op.drop_table("documents")
