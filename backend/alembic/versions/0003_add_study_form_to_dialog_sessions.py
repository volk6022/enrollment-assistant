"""add study_form to dialog_sessions

Revision ID: 0003_add_study_form_to_dialog_sessions
Revises: 0002_create_documents_versions
Create Date: 2025-12-23
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect

revision = "0003_add_study_form_to_dialog_sessions"
down_revision = "0002_create_documents_versions"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    insp = inspect(bind)
    cols = {c["name"] for c in insp.get_columns("dialog_sessions")}
    if "study_form" not in cols:
        op.add_column(
            "dialog_sessions",
            sa.Column("study_form", sa.String(length=32), nullable=True),
        )


def downgrade() -> None:
    bind = op.get_bind()
    insp = inspect(bind)
    cols = {c["name"] for c in insp.get_columns("dialog_sessions")}
    if "study_form" in cols:
        op.drop_column("dialog_sessions", "study_form")
