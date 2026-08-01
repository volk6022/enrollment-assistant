"""Unit tests for `backend.dialogue.memory.DialogueMemory` and the shapes in
`backend.dialogue.models`.

The load-bearing cases, each tied to a specific contract clause:

* `test_overlapping_turns_serialize_with_marker_and_unshifted_boundaries` --
  memory.md §4/§7: overlap is real dialogue, not a sorting bug, and the
  serializer must mark it without moving any boundary.
* `test_commit_*` (three cases) -- memory.md §5's table, and
  `inv_dropped_never_committed` from dialogue.qnt: a doigran reply and a
  yielded-mid-reply reply both commit (the second `is_partial=True`); a
  draft abandoned before any audio must never reach `turns` at all.
* `test_transcript_buffer_truncates_from_head` -- memory.md §6/§7: tail
  truncation would break llama-server's `cache_prompt` prefix match.
* `test_commit_agent_turn_voiced_fraction_reflects_partial_playback` --
  memory.md §7: fraction must come from delivered audio, never synthesized
  file length.
* `test_transcript_buffer_survives_completed_user_turns` /
  `test_transcript_buffer_head_truncation_spans_completed_turns` -- memory.md
  §6: `transcript_buffer` is a session-wide sliding window, not per-turn
  scratch space; a completed turn must NOT clear it, only `session.reset`
  (`reset_transcript()`) does.
"""
from __future__ import annotations

import pytest

from backend.dialogue.memory import DialogueMemory
from backend.dialogue.models import AgentState, DialogueState, DialogueTurn, Draft

# ---------------------------------------------------------------------------
# AgentState / Draft shape -- T-07/T-08 import these directly (tasks.md T-06).
# ---------------------------------------------------------------------------


def test_agent_state_has_exactly_the_eight_qnt_states() -> None:
    names = {member.value for member in AgentState}
    assert names == {
        "Greeting",
        "Listening",
        "DecidingInterject",
        "Formulating",
        "Speaking",
        "DecidingBargeIn",
        "Closing",
        "Ended",
    }


def test_draft_has_exactly_the_five_qnt_values() -> None:
    names = {member.value for member in Draft}
    assert names == {"NoDraft", "Building", "Voicing", "Dropped", "Committed"}


def test_dialogue_state_defaults_to_greeting_with_no_draft() -> None:
    state = DialogueState()
    assert state.agent is AgentState.GREETING
    assert state.draft is Draft.NO_DRAFT
    assert state.timers.turn_ms == 0
    assert state.timers.speech_left_ms == 0


# ---------------------------------------------------------------------------
# DialogueTurn field validation (memory.md §4: role-specific fields).
# ---------------------------------------------------------------------------


def test_user_turn_rejects_agent_only_fields() -> None:
    with pytest.raises(ValueError):
        DialogueTurn(role="user", text="привет", start_ms=0, end_ms=100, is_partial=True)


def test_agent_turn_rejects_user_only_field() -> None:
    with pytest.raises(ValueError):
        DialogueTurn(
            role="agent", text="здравствуйте", start_ms=0, end_ms=100, stt_confidence=0.9
        )


def test_turn_rejects_end_before_start() -> None:
    with pytest.raises(ValueError):
        DialogueTurn(role="user", text="x", start_ms=1000, end_ms=500)


# ---------------------------------------------------------------------------
# Overlap: the central case of memory.md §4.
# ---------------------------------------------------------------------------


def test_overlapping_turns_serialize_with_marker_and_unshifted_boundaries() -> None:
    memory = DialogueMemory()
    memory.commit_agent_turn(
        text="Для поступления понадобятся паспорт, аттестат и...",
        start_ms=12_300,
        delivered_ms=7_500,  # ends at 19_800
        planned_ms=7_500,
    )
    # Starts at 16_100, well before the agent turn's 19_800 end -- genuine
    # overlap, not a sequencing mistake.
    memory.add_user_turn(
        text="нет, подождите, я про медкомиссию",
        start_ms=16_100,
        end_ms=18_400,
    )

    assert [t.start_ms for t in memory.turns] == [12_300, 16_100]
    agent_turn, user_turn = memory.turns
    # Boundaries must be exactly what was passed in -- nothing "fixed" the
    # overlap by nudging either interval.
    assert (agent_turn.start_ms, agent_turn.end_ms) == (12_300, 19_800)
    assert (user_turn.start_ms, user_turn.end_ms) == (16_100, 18_400)

    rendered = memory.serialize_history()
    lines = rendered.splitlines()
    assert lines == [
        "[00:12.3–00:19.8] агент: Для поступления понадобятся паспорт, аттестат и...",
        "[00:16.1–00:18.4] собеседник: (поверх) нет, подождите, я про медкомиссию",
    ]


def test_non_overlapping_turns_get_no_marker() -> None:
    memory = DialogueMemory()
    memory.add_user_turn(text="привет", start_ms=0, end_ms=1_000)
    memory.commit_agent_turn(text="здравствуйте", start_ms=1_000, delivered_ms=2_000, planned_ms=2_000)

    rendered = memory.serialize_history()
    assert "(поверх)" not in rendered


def test_turn_starting_inside_an_earlier_but_not_immediately_preceding_turn_is_marked() -> None:
    # A[0, 10000] fully contains the gap before B; C starts inside A's span
    # but after B ends -- only comparing against the immediately preceding
    # line would miss this, a running-max-of-end_ms check must catch it.
    memory = DialogueMemory()
    memory.commit_agent_turn(text="A", start_ms=0, delivered_ms=10_000, planned_ms=10_000)
    memory.add_user_turn(text="B", start_ms=1_000, end_ms=2_000)
    memory.add_user_turn(text="C", start_ms=3_000, end_ms=4_000)

    rendered = memory.serialize_history().splitlines()
    assert "(поверх)" not in rendered[0]  # A: nothing precedes it
    assert "(поверх)" in rendered[1]  # B: starts inside A
    assert "(поверх)" in rendered[2]  # C: starts inside A too (running max), not just after B


def test_turns_stay_sorted_by_start_ms_regardless_of_commit_order() -> None:
    # The agent turn conceptually starts earlier (its first audio chunk went
    # out first) but is only committed later, once playback is known to have
    # finished -- collecting history in commit order would put it after a
    # user turn that both started and was recorded later in real time.
    memory = DialogueMemory()
    memory.add_user_turn(text="later user turn", start_ms=5_000, end_ms=6_000)
    memory.commit_agent_turn(text="earlier agent turn", start_ms=1_000, delivered_ms=500, planned_ms=500)

    assert [t.text for t in memory.turns] == ["earlier agent turn", "later user turn"]


# ---------------------------------------------------------------------------
# Commit vs abandon -- memory.md §5's three rows.
# ---------------------------------------------------------------------------


def test_commit_fully_played_reply_is_committed_with_full_fraction() -> None:
    memory = DialogueMemory()
    turn = memory.commit_agent_turn(
        text="полный ответ", start_ms=0, delivered_ms=9_700, planned_ms=9_700
    )
    assert turn.is_partial is False
    assert turn.voiced_fraction == 1.0
    assert memory.turns == [turn]


def test_commit_yielded_mid_reply_is_committed_as_partial() -> None:
    memory = DialogueMemory()
    turn = memory.commit_agent_turn(
        text="Для поступления понадобятся паспорт, аттестат",
        start_ms=0,
        delivered_ms=4_100,
        planned_ms=9_700,
    )
    assert turn.is_partial is True
    assert turn.voiced_fraction == pytest.approx(4_100 / 9_700)
    assert memory.turns == [turn]  # heard-in-part still enters memory (FR-16)


def test_draft_abandoned_before_any_audio_never_enters_memory() -> None:
    # Per memory.md §5, "отменить" IS simply never calling commit -- there
    # is no drop() to call. The invariant under test is that the turns list
    # stays empty when a draft never gets committed.
    memory = DialogueMemory()
    memory.append_transcript("черновик, который никто не услышал", 0, 500)

    assert memory.turns == []


def test_commit_agent_turn_refuses_zero_delivered_audio() -> None:
    # Defends inv_dropped_never_committed at the API boundary: a draft with
    # nothing actually voiced cannot be committed even by mistake.
    memory = DialogueMemory()
    with pytest.raises(ValueError):
        memory.commit_agent_turn(text="never voiced", start_ms=0, delivered_ms=0, planned_ms=9_700)
    assert memory.turns == []


# ---------------------------------------------------------------------------
# Transcript buffer -- memory.md §6.
# ---------------------------------------------------------------------------


def test_transcript_buffer_truncates_from_head() -> None:
    memory = DialogueMemory(max_transcript_chars=10)
    memory.append_transcript("0123456789", 0, 1_000)
    assert memory.transcript_buffer == "0123456789"

    memory.append_transcript("ABC", 1_000, 1_500)
    # Head truncated: the oldest characters ("012") are dropped, the tail
    # (newest content, "ABC") is preserved -- never the reverse.
    assert memory.transcript_buffer == "3456789ABC"
    assert len(memory.transcript_buffer) == 10


def test_transcript_buffer_default_cap_matches_fr20() -> None:
    memory = DialogueMemory()
    memory.append_transcript("x" * 6_000, 0, 1_000)
    assert len(memory.transcript_buffer) == 5_000
    assert memory.transcript_buffer == "x" * 5_000


def test_append_transcript_records_chunk_timings() -> None:
    memory = DialogueMemory()
    memory.append_transcript("привет", 0, 300)
    memory.append_transcript(", как дела", 300, 900)
    assert memory.chunk_timings == [(0, 300), (300, 900)]
    assert memory.transcript_buffer == "привет, как дела"


def test_transcript_buffer_survives_completed_user_turns() -> None:
    # memory.md §6: transcript_buffer is a session-wide sliding window, not
    # per-utterance scratch space. Finishing a turn (add_user_turn) must NOT
    # clear it -- only session.reset (reset_transcript()) does.
    memory = DialogueMemory()

    memory.append_transcript("привет", 0, 300)
    memory.add_user_turn(text="привет", start_ms=0, end_ms=300)
    assert memory.transcript_buffer == "привет"  # turn completion didn't clear it

    memory.append_transcript(", как дела", 1_000, 1_500)
    memory.add_user_turn(text=", как дела", start_ms=1_000, end_ms=1_500)

    # Text from BOTH turns is still present -- it accumulates across turn
    # boundaries, it doesn't restart per turn.
    assert memory.transcript_buffer == "привет, как дела"
    assert memory.chunk_timings == [(0, 300), (1_000, 1_500)]
    assert len(memory.turns) == 2


def test_transcript_buffer_head_truncation_spans_completed_turns() -> None:
    # Sanity check straight from memory.md §6: 5000 chars is ~40s of dense
    # speech, while a turn ends after 0.5s of silence -- a single turn's
    # text is nowhere near the cap, so truncation only makes sense once the
    # buffer has accumulated across several turns, not within one.
    memory = DialogueMemory(max_transcript_chars=15)

    memory.append_transcript("0123456789", 0, 1_000)  # first turn, 10 chars
    memory.add_user_turn(text="0123456789", start_ms=0, end_ms=1_000)
    assert memory.transcript_buffer == "0123456789"  # under the cap, untouched

    memory.append_transcript("ABCDEFGHIJ", 1_000, 2_000)  # second turn, 10 chars
    memory.add_user_turn(text="ABCDEFGHIJ", start_ms=1_000, end_ms=2_000)
    # 20 chars total now exceeds the 15-char cap: truncated from the head
    # (the tail of turn 1), the second turn's text survives in full.
    assert memory.transcript_buffer == "56789ABCDEFGHIJ"
    assert len(memory.transcript_buffer) == 15


def test_reset_transcript_is_only_for_session_reset() -> None:
    # reset_transcript() exists for session.reset, not for turn completion
    # (memory.md §6). This test exercises exactly that call, distinct from
    # ordinary turn completion above, which must NOT trigger it.
    memory = DialogueMemory()
    memory.append_transcript("привет", 0, 300)
    memory.add_user_turn(text="привет", start_ms=0, end_ms=300)

    memory.reset_transcript()  # simulates session.reset

    assert memory.transcript_buffer == ""
    assert memory.chunk_timings == []
    assert len(memory.turns) == 1  # committed history is untouched by session.reset too


# ---------------------------------------------------------------------------
# serialize_history: max_turns window.
# ---------------------------------------------------------------------------


def test_serialize_history_max_turns_keeps_only_the_most_recent() -> None:
    memory = DialogueMemory()
    memory.add_user_turn(text="один", start_ms=0, end_ms=100)
    memory.add_user_turn(text="два", start_ms=200, end_ms=300)
    memory.add_user_turn(text="три", start_ms=400, end_ms=500)

    rendered = memory.serialize_history(max_turns=2)
    assert "один" not in rendered
    assert "два" in rendered
    assert "три" in rendered


def test_serialize_history_empty_memory_is_empty_string() -> None:
    assert DialogueMemory().serialize_history() == ""


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
