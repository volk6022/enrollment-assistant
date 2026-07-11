"""create dialog tables

Revision ID: 0001_create_dialog_tables
Revises:
Create Date: 2025-12-22
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0001_create_dialog_tables"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "dialog_sessions",
        sa.Column("session_id", sa.String(length=64), primary_key=True),
        sa.Column("status", sa.String(length=32), nullable=True),
        sa.Column("level", sa.String(length=32), nullable=True),
        sa.Column("pending_clarification", sa.String(length=32), nullable=True),
        sa.Column("pending_question", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
    )

    op.create_table(
        "dialog_messages",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("session_id", sa.String(length=64), sa.ForeignKey("dialog_sessions.session_id"), nullable=False),
        sa.Column("role", sa.String(length=16), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("citations", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
    )
    op.create_index("ix_dialog_messages_session_id", "dialog_messages", ["session_id"])


def downgrade() -> None:
    op.drop_index("ix_dialog_messages_session_id", table_name="dialog_messages")
    op.drop_table("dialog_messages")
    op.drop_table("dialog_sessions")
