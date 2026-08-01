"""`DialogueMemory` -- the interleaved dialogue history plus the session-wide
transcript buffer, assembled from whisper timings and our own playback
offsets (`contracts/memory.md` §4-§6, FR-18/FR-19/FR-20).

`turns` and `transcript_buffer` are two DIFFERENT representations of the
same conversation (memory.md §6's table), not one derived from the other:

  `turns`             structured history: roles, boundaries, overlaps --
                       what the LLM reasons about order and overlap with
                       (FR-18, FR-19). Changes once per finished turn.
  `transcript_buffer` raw speech text, concatenated, session-wide, from
                       `session.ready` to `session.reset` -- cheap
                       incremental prefill via tail-append (FR-10, FR-20).
                       Changes on every partial transcript, mid-turn.

Everything here runs on the event loop only (plan.md §2: "История диалога,
тайминги, буфер транскрипта трогаются только на loop"). No lock, no thread
safety -- that is not an oversight, it is the contract: STT/TTS worker
threads hand back plain values (transcribed text, delivered-audio
milliseconds), and only loop code ever calls into this class.

Four rules this module exists to get right, each one a `contracts/
memory.md` §7 "что нельзя делать" entry that fails silently if broken:

1. Turns are kept sorted by `start_ms`, insertion order be damned, and nver
   nudged apart when two intervals overlap. Collecting history in event-
   arrival order is exactly the mistake §7 calls out -- an agent turn
   commits only once fully played out or yielded, which can happen well
   after a later user turn already landed in `turns`.
2. `commit_agent_turn()` is the ONLY way an agent reply enters `turns`, and
   it refuses a reply with zero delivered audio (`inv_dropped_never_committed`
   in dialogue.qnt). A dropped draft is not represented by any call into
   this class at all -- per memory.md §5, "отменить" means simply never
   calling commit.
3. `append_transcript()` only ever grows the buffer at the tail and only
   ever truncates from the head. Truncating the tail would change the
   shared prefix llama-server's `cache_prompt` matches against, turning
   every future update into a full prompt re-evaluation (memory.md §6/§7,
   llm.md §6's ~61-token cache floor).
4. `transcript_buffer` spans the WHOLE session, not one turn -- it is a
   sliding window over everything said since `session.ready`, and a turn
   ending does not clear it. `reset_transcript()` exists only for
   `session.reset`; calling it at turn boundaries would defeat the point of
   the 5000-char cap (memory.md §6: at ~40s of dense speech per cap and a
   0.5s end-of-turn threshold, a per-turn buffer would never come close to
   filling, and head-truncation would never trigger).
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass, field

from backend.dialogue.models import DialogueTurn

_ROLE_LABELS: dict[str, str] = {"agent": "агент", "user": "собеседник"}
_OVERLAP_MARKER = "(поверх) "

_DEFAULT_MAX_TRANSCRIPT_CHARS = 5000
"""Matches FR-20's literal number. Production wiring passes
`settings.dialogue.transcript_buffer_chars` explicitly (FR-32: config.py is
the only place that reads `.env`) -- this default exists only so the class
is usable standalone in tests without wiring up `Settings`, the same
pattern `AudioRing`/`VadGate` already use for their threshold defaults.
"""


def _format_timestamp(ms: int) -> str:
    """`12300` -> `"00:12.3"` -- minutes:seconds.decisecond, matching the
    example rendering in `contracts/memory.md` §4 exactly.
    """
    total_deciseconds = round(ms / 100)
    minutes, remainder_deciseconds = divmod(total_deciseconds, 600)
    seconds, decisecond = divmod(remainder_deciseconds, 10)
    return f"{minutes:02d}:{seconds:02d}.{decisecond}"


@dataclass
class DialogueMemory:
    """`contracts/memory.md` §4's three fields, plus the constructor-level
    cap `append_transcript()` enforces. `turns` is maintained sorted by
    `start_ms` as an invariant of this class -- never append to it directly;
    every mutator below goes through `_insert_turn()`.

    `transcript_buffer`/`chunk_timings` are session-scoped, not turn-scoped
    (memory.md §6) -- they accumulate from `session.ready` and only
    `reset_transcript()` (called on `session.reset`) clears them. A
    completed user or agent turn does NOT reset them; `add_user_turn()` /
    `commit_agent_turn()` and `append_transcript()` are independent, both
    fed by the caller from the same underlying speech but for different
    purposes (structured reasoning vs. cheap incremental prefill).
    """

    turns: list[DialogueTurn] = field(default_factory=list)
    transcript_buffer: str = ""
    chunk_timings: list[tuple[int, int]] = field(default_factory=list)
    max_transcript_chars: int = _DEFAULT_MAX_TRANSCRIPT_CHARS

    def _insert_turn(self, turn: DialogueTurn) -> DialogueTurn:
        """Keeps `turns` sorted by `start_ms` without touching any other
        turn's boundaries. `bisect.insort` with `key=` finds the correct
        slot in O(log n) comparisons / O(n) shift, the same way it would for
        a non-overlapping list -- overlap is a property of the *values*
        being sorted (two turns can share time), not of the sort itself,
        which only ever cares about `start_ms` as a key.
        """
        bisect.insort(self.turns, turn, key=lambda t: t.start_ms)
        return turn

    def add_user_turn(
        self,
        *,
        text: str,
        start_ms: int,
        end_ms: int,
        stt_confidence: float | None = None,
    ) -> DialogueTurn:
        """`start_ms` is the VAD's prerolled `speech_start_ms`, `end_ms` is
        the `SpeechEnded` boundary (memory.md §4's table, "реплика
        собеседника" row) -- both are the caller's (T-09's) responsibility
        to supply from `VadGate` events, this method only stores them.
        """
        turn = DialogueTurn(
            role="user",
            text=text,
            start_ms=start_ms,
            end_ms=end_ms,
            stt_confidence=stt_confidence,
        )
        return self._insert_turn(turn)

    def commit_agent_turn(
        self,
        *,
        text: str,
        start_ms: int,
        delivered_ms: int,
        planned_ms: int,
    ) -> DialogueTurn:
        """Commit a reply that voiced at least one chunk -- covers BOTH
        rows of memory.md §5 that reach `turns`: doigran (`delivered_ms ==
        planned_ms`) and yielded mid-reply (`delivered_ms < planned_ms`,
        `is_partial=True`). The third row (abandoned before any audio) is
        not a call into this method at all; `delivered_ms <= 0` raises
        instead of silently producing a zero-fraction "committed" turn,
        which would violate `inv_dropped_never_committed` by construction.

        `delivered_ms` MUST be actually-emitted audio duration, never the
        synthesized file length (memory.md §7) -- those two numbers differ
        by exactly the amount that got yielded away mid-sentence, and that
        difference is `voiced_fraction`'s entire reason to exist. Computing
        it here from caller-supplied durations (rather than accepting a
        pre-computed `voiced_fraction`/`is_partial` pair) means there is
        only one source of truth for whether a turn is partial -- no call
        site can pass an `is_partial` that disagrees with its own
        `delivered_ms`/`planned_ms`.
        """
        if delivered_ms <= 0:
            raise ValueError(
                "commit_agent_turn() requires delivered_ms > 0 -- a draft "
                "with no audio delivered yet must be abandoned by simply "
                "never calling commit (memory.md §5), not committed with a "
                "zero voiced_fraction"
            )
        planned_ms = max(planned_ms, delivered_ms)
        voiced_fraction = min(1.0, delivered_ms / planned_ms)
        turn = DialogueTurn(
            role="agent",
            text=text,
            start_ms=start_ms,
            end_ms=start_ms + delivered_ms,
            voiced_fraction=voiced_fraction,
            is_partial=delivered_ms < planned_ms,
        )
        return self._insert_turn(turn)

    def append_transcript(self, text: str, chunk_start_ms: int, chunk_end_ms: int) -> None:
        """Grows `transcript_buffer` by `text` -- the caller (STT worker /
        session orchestration) is responsible for diffing consecutive
        whisper partials down to just the new suffix; this method only ever
        appends, matching FR-10/llm.md §6's incremental-prefill design. Caps
        at `max_transcript_chars` by dropping from the HEAD, never the tail
        (memory.md §6/§7).

        Called across turn boundaries without interruption -- a user turn
        ending does not stop or reset this accumulation, it is a sliding
        window over the WHOLE session (memory.md §6). Only `reset_transcript()`
        clears it, and that only happens on `session.reset`.
        """
        self.transcript_buffer += text
        self.chunk_timings.append((chunk_start_ms, chunk_end_ms))
        if len(self.transcript_buffer) > self.max_transcript_chars:
            self.transcript_buffer = self.transcript_buffer[-self.max_transcript_chars :]

    def reset_transcript(self) -> None:
        """Clears `transcript_buffer`/`chunk_timings` for `session.reset`
        ONLY -- never for a finished turn.

        `transcript_buffer` is a session-wide sliding window (memory.md §6),
        not per-utterance scratch space: it accumulates from `session.ready`
        onward and a user turn ending (`add_user_turn()`) does NOT call
        this. Calling it at turn boundaries would be a bug -- the whole
        point of the 5000-char cap and head-truncation is that the buffer
        keeps growing *across* turns until it actually needs trimming
        (memory.md §6: ~40s of dense speech fits under the cap, and a turn
        ends after 0.5s of silence, so a per-turn buffer would never fill
        it and the truncation rule would never fire). The ONLY caller of
        this method should be session-reset handling in T-09's
        orchestration; T-07's automaton nodes and turn-completion code must
        NOT call it.

        Does not touch `turns` -- that is a separate, independent
        representation of the same conversation (module docstring, memory.md
        §6's table) and has its own lifecycle.
        """
        self.transcript_buffer = ""
        self.chunk_timings = []

    def serialize_history(self, *, max_turns: int | None = None) -> str:
        """Renders `turns` as the "история диалога" prompt block (llm.md
        §6), one line per turn, oldest first. Overlap is marked explicitly
        (memory.md §4's worked example) rather than left for the LLM to
        infer from timestamps it would have to parse itself:

            [00:12.3–00:19.8] агент: Для поступления понадобятся паспорт...
            [00:16.1–00:18.4] собеседник: (поверх) нет, подождите...

        A turn is marked overlapping when it starts before the latest
        `end_ms` seen so far among the turns already rendered -- a running
        maximum, not just a comparison against the immediately preceding
        line, so a turn nested inside an earlier, longer-running turn is
        still caught correctly (interval-merge logic, not adjacent-pair
        logic). Boundaries are never adjusted to remove the overlap; only
        the marker changes.

        `max_turns`, when given, keeps only the most recent turns -- this is
        `DIALOGUE_HISTORY_MAX_TURNS` (`config.py`'s `DialogueSettings`),
        supplied by the caller, never read from `.env` here (FR-32).
        """
        turns = self.turns[-max_turns:] if max_turns is not None else self.turns
        lines: list[str] = []
        latest_end_ms = float("-inf")
        for turn in turns:
            marker = _OVERLAP_MARKER if turn.start_ms < latest_end_ms else ""
            lines.append(
                f"[{_format_timestamp(turn.start_ms)}–{_format_timestamp(turn.end_ms)}] "
                f"{_ROLE_LABELS[turn.role]}: {marker}{turn.text}"
            )
            latest_end_ms = max(latest_end_ms, turn.end_ms)
        return "\n".join(lines)


__all__ = ["DialogueMemory"]
