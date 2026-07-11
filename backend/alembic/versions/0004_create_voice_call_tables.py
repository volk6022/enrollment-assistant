"""create voice call tables

Revision ID: 0004_create_voice_call_tables
Revises: 0003_add_study_form_to_dialog_sessions
Create Date: 2026-04-09
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from sqlalchemy import inspect

revision = "0004_create_voice_call_tables"
down_revision = "0003_add_study_form_to_dialog_sessions"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    insp = inspect(bind)
    tables = set(insp.get_table_names())

    if "voice_call_sessions" not in tables:
        op.create_table(
            "voice_call_sessions",
            sa.Column("call_id", sa.String(length=64), primary_key=True),
            sa.Column("session_id", sa.String(length=64), nullable=False),
            sa.Column("phone_number_masked", sa.String(length=64), nullable=True),
            sa.Column("transport", sa.String(length=32), nullable=False, server_default=sa.text("'demo'")),
            sa.Column("direction", sa.String(length=16), nullable=False, server_default=sa.text("'inbound'")),
            sa.Column("state", sa.String(length=32), nullable=False, server_default=sa.text("'active'")),
            sa.Column("handoff_requested", sa.Boolean(), nullable=False, server_default=sa.text("false")),
            sa.Column("barge_in_enabled", sa.Boolean(), nullable=False, server_default=sa.text("true")),
            sa.Column("metadata_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
            sa.Column("started_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
            sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        )
        op.create_index("ix_voice_call_sessions_session_id", "voice_call_sessions", ["session_id"])

    if "voice_call_turns" not in tables:
        op.create_table(
            "voice_call_turns",
            sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
            sa.Column("call_id", sa.String(length=64), sa.ForeignKey("voice_call_sessions.call_id"), nullable=False),
            sa.Column("session_id", sa.String(length=64), nullable=False),
            sa.Column("role", sa.String(length=16), nullable=False),
            sa.Column("transcript", sa.Text(), nullable=False),
            sa.Column("tts_text", sa.Text(), nullable=True),
            sa.Column("meta_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        )
        op.create_index("ix_voice_call_turns_call_id", "voice_call_turns", ["call_id"])
        op.create_index("ix_voice_call_turns_session_id", "voice_call_turns", ["session_id"])

    if "voice_call_events" not in tables:
        op.create_table(
            "voice_call_events",
            sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
            sa.Column("call_id", sa.String(length=64), sa.ForeignKey("voice_call_sessions.call_id"), nullable=False),
            sa.Column("event_type", sa.String(length=32), nullable=False),
            sa.Column("payload_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        )
        op.create_index("ix_voice_call_events_call_id", "voice_call_events", ["call_id"])
        op.create_index("ix_voice_call_events_event_type", "voice_call_events", ["event_type"])


def downgrade() -> None:
    bind = op.get_bind()
    insp = inspect(bind)
    tables = set(insp.get_table_names())

    if "voice_call_events" in tables:
        op.drop_index("ix_voice_call_events_event_type", table_name="voice_call_events")
        op.drop_index("ix_voice_call_events_call_id", table_name="voice_call_events")
        op.drop_table("voice_call_events")
    if "voice_call_turns" in tables:
        op.drop_index("ix_voice_call_turns_session_id", table_name="voice_call_turns")
        op.drop_index("ix_voice_call_turns_call_id", table_name="voice_call_turns")
        op.drop_table("voice_call_turns")
    if "voice_call_sessions" in tables:
        op.drop_index("ix_voice_call_sessions_session_id", table_name="voice_call_sessions")
        op.drop_table("voice_call_sessions")
