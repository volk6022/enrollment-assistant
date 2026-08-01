"""The 25 `dialogue.qnt` actions, translated to Python 1:1 (T-07; tasks.md,
plan.md §9 "Соответствие модели и кода": "action greetingPlays и остальные
23 -> ребро в DialogueMachine, имя метода = имя действия" -- plan.md's count
predates `formulatingWaits`, added to the model after T-09's implementation
found `Formulating` had no self-loop for "still generating, nothing
happened" (`dialogue.qnt`'s own comment on the action); 24 -> 25 total).

Every function below is named EXACTLY like its `action` in
`specs/formal/dialogue.qnt` (camelCase, not `snake_case` -- deliberate
deviation from the project's usual naming so the correspondence in
`tests/unit/test_state_machine.py` and the final cross-check
(tasks.md "Финальная проверка" §2) can compare the two by name, not by a
second hand-kept table). Each is a pure function:

    (DialogueState, StepContext) -> DialogueState | None

Returns the updated `DialogueState` if the action's guard holds this step,
`None` otherwise -- the same shape as Quint's `all { guard, ... effects }`
block, just phrased as "did this fire" instead of "is this enabled". Callers
(`machine.py`) try a state's candidates in order and apply the first that
fires; `dialogue.qnt`'s `inv_no_deadlock` guarantees at least one always does
for a reachable state, given a well-formed `StepContext`.

Two things the model has that `DialogueState` (T-06, `models.py`) does not,
and why:

* `userSpeaking` -- deliberately excluded from `DialogueState` ("already
  owned by audio.vad.VadGate", models.py). The model needs it as internal
  state so `userStartsSpeaking` (false -> true) and every level-guarded
  action (userSpeaking already true) can be told apart; production code
  gets the CURRENT level from the caller every step (`AutomatonInput.
  user_speaking`) and the EDGE from `StepContext.speech_started`, computed
  by `machine.py` from one extra bit it tracks itself (`AutomatonState.
  user_was_speaking`) without touching T-06's file at all.
* `turnLen`/`idleLen`/`overlapLen`/`speechLeft` are small bounded ordinals
  in the model (so Apalache can check exhaustively) but real millisecond
  counters in `DialogueState.timers` (`DialogueTimers`, T-06). Every model
  effect of the shape `X' = X + 1` becomes `X_ms += ctx.event.elapsed_ms`
  here -- one abstract tick equals "however much wall-clock time this step
  represents". Effects of the shape `X' = 1` (`userInterruptsGreeting`,
  `closingInterrupted` -- starting a new turn at its first tick, not
  incrementing an existing one) become `X_ms = ctx.event.elapsed_ms` for the
  same reason, not a hardcoded `1`.

Thresholds (`TALK_LIMIT`/`IDLE_LIMIT`/`OVERLAP_LIMIT`/`TAIL_MIN` in the
model, ordinal) arrive as `StepContext.thresholds`
(`DialogueThresholds`, real milliseconds) -- built by the caller from
`backend.config.settings.dialogue.dialogue_interject_after_s` etc. (plan.md
§9's correspondence table) multiplied by 1000. This module does NOT import
`backend.config`: that module's `settings` singleton is constructed at
import time and calls `sys.exit()` if `.env` is missing (`config.py`'s own
docstring), which would make importing this module for a unit test
dependent on a `.env` file existing on disk. `dialogue/__init__.py` and
`memory.py` already avoid this for the same reason ("Nothing in this
package reads `.env` directly (FR-32) -- callers pass thresholds
explicitly") -- this module keeps to that pattern.

One faithfully-preserved oddity, called out so it does not look like a
bug later: `overlapGrows` does NOT decrement `speech_left_ms` while overlap
accumulates, even though the agent's audio is (in reality) still playing
during that time. This is not an omission -- the model's own `overlapGrows`
action sets `speechLeft' = speechLeft` unchanged, on purpose (module-level
comment in `dialogue.qnt`: "Timers are abstracted to small bounded counters
... the thresholds here are ORDINAL, not literal seconds" -- ordinal ticks
for DIFFERENT quantities don't have to advance together). Model is primary;
this is reproduced exactly, not "fixed".
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

from backend.dialogue.models import AgentState, DialogueState, DialogueTimers, Draft
from backend.llm.schemas import BargeInDecision, InterjectDecision

_GREETING = AgentState.GREETING
_LISTENING = AgentState.LISTENING
_DECIDING_INTERJECT = AgentState.DECIDING_INTERJECT
_FORMULATING = AgentState.FORMULATING
_SPEAKING = AgentState.SPEAKING
_DECIDING_BARGE_IN = AgentState.DECIDING_BARGE_IN
_CLOSING = AgentState.CLOSING
_ENDED = AgentState.ENDED

_NO_DRAFT = Draft.NO_DRAFT
_BUILDING = Draft.BUILDING
_VOICING = Draft.VOICING
_DROPPED = Draft.DROPPED
_COMMITTED = Draft.COMMITTED


class DeadlockError(RuntimeError):
    """No candidate action's guard matched for the current state.

    `dialogue.qnt`'s `inv_no_deadlock` (verified exhaustively by Apalache,
    `specs/formal/verify.ps1`) says every reachable state has at least one
    enabled action -- reaching this means either `machine.py`'s candidate
    list for a state is incomplete/misordered, or the caller built a
    `StepContext` the model does not allow (e.g. a decision-less step while
    `agent == DecidingInterject`). Either way this is a bug to fix, not a
    condition to swallow.
    """

    def __init__(self, agent: AgentState) -> None:
        super().__init__(
            f"no dialogue.qnt action fired for state {agent.value!r} -- "
            "inv_no_deadlock says this is unreachable for a well-formed "
            "StepContext; this is a bug in the caller or in machine.py's "
            "candidate list, not a runtime condition to handle silently"
        )
        self.agent = agent


@dataclass(frozen=True, slots=True)
class DialogueThresholds:
    """Real-millisecond stand-ins for `dialogue.qnt`'s ordinal
    `TALK_LIMIT`/`IDLE_LIMIT`/`OVERLAP_LIMIT`/`TAIL_MIN` (plan.md §9).

    Built by the caller (T-09's session orchestration) from
    `settings.dialogue.dialogue_interject_after_s` etc., multiplied by 1000
    -- deliberately not a `classmethod` reading `backend.config` here, see
    the module docstring.
    """

    turn_limit_ms: int
    idle_limit_ms: int
    overlap_limit_ms: int
    tail_min_ms: int


@dataclass(frozen=True, slots=True)
class AutomatonInput:
    """External signals for one step (tasks.md T-07 scope: the automaton
    itself, not audio/VAD/STT/LLM plumbing -- those are T-02/T-05/T-09).

    Two speech signals, not one, because the model needs both a LEVEL and
    an EDGE to tell `userStartsSpeaking` (fires once, on the false->true
    transition) apart from every action that requires speech to ALREADY
    have been ongoing (`userInterruptsGreeting`, `userKeepsTalking`,
    `closingInterrupted`, `draftAbandoned`, the three `Speaking`-state
    overlap actions). `user_speaking` is the current level; `machine.py`
    derives the edge itself (`StepContext.speech_started`) by remembering
    the previous level in `AutomatonState`, so callers only ever report
    "is the user talking right now" -- one less thing for T-09 to get
    wrong.

    `turn_ended` is a third, independently-driven signal (FR-06's 0.5 s
    VAD silence-timeout) -- NOT derived from `user_speaking` turning False,
    because `dialogue.qnt`'s `userTurnEnds` guard requires `userSpeaking`
    (the pre-step level) to still read `true`: per `STATE_MACHINE.md`'s
    "Область моделирования", VAD's own silence detector is explicitly
    outside the automaton, and `userTurnEnds` is simply "the action that
    fires when VAD reports the turn ended", not something this module
    infers from a level going quiet on its own.
    """

    user_speaking: bool = False
    elapsed_ms: int = 0
    turn_ended: bool = False

    draft_ready: bool = False
    answer_duration_ms: int | None = None
    """Required exactly when `draft_ready` is True -- becomes the new
    `speech_left_ms` (draftReady's `speechLeft' = SPEECH_MAX`)."""

    farewell_duration_ms: int | None = None
    """Required exactly when Listening -> Closing fires (idleHangup) --
    becomes the new `speech_left_ms` for playing the farewell."""

    interject_decision: InterjectDecision | None = None
    barge_in_decision: BargeInDecision | None = None

    dialogue_history: str = ""
    transcript_so_far: str = ""
    draft_answer: str = ""
    interlocutor_transcript: str = ""
    total_answer_ms: int | None = None
    """Needed alongside `speech_left_ms` to compute "доля озвученного"
    (FR-14) for the barge-in prompt -- `machine.py`'s DecidingBargeIn node
    reads this, not the pure `bargeInAccepted`/`bargeInDeclined` actions."""


@dataclass(frozen=True, slots=True)
class StepContext:
    """Bundles one step's `AutomatonInput` with the derived edge flag and
    the session's thresholds -- what every action function needs besides
    the current `DialogueState`."""

    event: AutomatonInput
    speech_started: bool
    thresholds: DialogueThresholds


ActionFn = Callable[[DialogueState, StepContext], "DialogueState | None"]


def _continuing_speech(ctx: StepContext) -> bool:
    """`userSpeaking` already true before this step -- the level-guarded
    half of every action that reacts to ongoing speech, as opposed to the
    instant it starts (`userStartsSpeaking`)."""
    return ctx.event.user_speaking and not ctx.speech_started


# ---------------------------------------------------------------------------
# greeting: interruptible by the user at any point
# ---------------------------------------------------------------------------


def greetingPlays(state: DialogueState, ctx: StepContext) -> DialogueState | None:
    if not (
        state.agent is _GREETING
        and state.timers.speech_left_ms > 0
        and not ctx.event.user_speaking
    ):
        return None
    remaining = max(0, state.timers.speech_left_ms - ctx.event.elapsed_ms)
    return replace(state, timers=replace(state.timers, speech_left_ms=remaining))


def greetingFinishes(state: DialogueState, ctx: StepContext) -> DialogueState | None:
    if not (state.agent is _GREETING and state.timers.speech_left_ms == 0):
        return None
    return replace(state, agent=_LISTENING, draft=_NO_DRAFT, timers=DialogueTimers())


def userInterruptsGreeting(state: DialogueState, ctx: StepContext) -> DialogueState | None:
    if not (state.agent is _GREETING and _continuing_speech(ctx)):
        return None
    return replace(
        state,
        agent=_LISTENING,
        draft=_NO_DRAFT,
        timers=DialogueTimers(turn_ms=ctx.event.elapsed_ms),
    )


# ---------------------------------------------------------------------------
# user speech start (any state but Ended)
# ---------------------------------------------------------------------------


def userStartsSpeaking(state: DialogueState, ctx: StepContext) -> DialogueState | None:
    if not (ctx.speech_started and state.agent is not _ENDED):
        return None
    return replace(state, timers=replace(state.timers, idle_ms=0, overlap_ms=0))


# ---------------------------------------------------------------------------
# listening
# ---------------------------------------------------------------------------


def userKeepsTalking(state: DialogueState, ctx: StepContext) -> DialogueState | None:
    if not (
        state.agent is _LISTENING
        and _continuing_speech(ctx)
        and state.timers.turn_ms < ctx.thresholds.turn_limit_ms
    ):
        return None
    new_turn_ms = state.timers.turn_ms + ctx.event.elapsed_ms
    return replace(
        state, timers=replace(state.timers, turn_ms=new_turn_ms, idle_ms=0, overlap_ms=0)
    )


def userTurnEnds(state: DialogueState, ctx: StepContext) -> DialogueState | None:
    if not (state.agent is _LISTENING and ctx.event.turn_ended):
        return None
    return replace(state, agent=_FORMULATING, draft=_BUILDING, timers=DialogueTimers())


def reachTalkLimit(state: DialogueState, ctx: StepContext) -> DialogueState | None:
    if not (
        state.agent is _LISTENING
        and _continuing_speech(ctx)
        and state.timers.turn_ms >= ctx.thresholds.turn_limit_ms
    ):
        return None
    return replace(
        state,
        agent=_DECIDING_INTERJECT,
        timers=replace(state.timers, idle_ms=0, overlap_ms=0),
    )


def idleTicks(state: DialogueState, ctx: StepContext) -> DialogueState | None:
    if not (
        state.agent is _LISTENING
        and not ctx.event.user_speaking
        and state.timers.idle_ms < ctx.thresholds.idle_limit_ms
    ):
        return None
    new_idle_ms = state.timers.idle_ms + ctx.event.elapsed_ms
    return replace(
        state, timers=replace(state.timers, idle_ms=new_idle_ms, turn_ms=0, overlap_ms=0)
    )


def idleHangup(state: DialogueState, ctx: StepContext) -> DialogueState | None:
    if not (
        state.agent is _LISTENING
        and not ctx.event.user_speaking
        and state.timers.idle_ms >= ctx.thresholds.idle_limit_ms
    ):
        return None
    if ctx.event.farewell_duration_ms is None:
        raise ValueError("idleHangup requires AutomatonInput.farewell_duration_ms")
    return replace(
        state,
        agent=_CLOSING,
        draft=_NO_DRAFT,
        timers=DialogueTimers(
            idle_ms=state.timers.idle_ms,  # idleLen' = idleLen -- preserved, not reset
            speech_left_ms=ctx.event.farewell_duration_ms,
        ),
    )


# ---------------------------------------------------------------------------
# interject decision
# ---------------------------------------------------------------------------


def interjectDeclined(state: DialogueState, ctx: StepContext) -> DialogueState | None:
    decision = ctx.event.interject_decision
    if not (state.agent is _DECIDING_INTERJECT and decision is not None and not decision.interject):
        return None
    # CRUCIAL (FR-08): turn_ms is NOT reset here.
    return replace(state, agent=_LISTENING, timers=replace(state.timers, idle_ms=0, overlap_ms=0))


def interjectAccepted(state: DialogueState, ctx: StepContext) -> DialogueState | None:
    decision = ctx.event.interject_decision
    if not (state.agent is _DECIDING_INTERJECT and decision is not None and decision.interject):
        return None
    return replace(
        state,
        agent=_FORMULATING,
        draft=_BUILDING,
        timers=replace(state.timers, idle_ms=0, overlap_ms=0, speech_left_ms=0),
    )


# ---------------------------------------------------------------------------
# formulating: no audio out yet, so abandoning is free
# ---------------------------------------------------------------------------


def draftReady(state: DialogueState, ctx: StepContext) -> DialogueState | None:
    if not (state.agent is _FORMULATING and state.draft is _BUILDING and ctx.event.draft_ready):
        return None
    if ctx.event.answer_duration_ms is None:
        raise ValueError("draftReady requires AutomatonInput.answer_duration_ms")
    return replace(
        state,
        agent=_SPEAKING,
        draft=_VOICING,
        timers=replace(
            state.timers,
            speech_left_ms=ctx.event.answer_duration_ms,
            idle_ms=0,
            overlap_ms=0,
        ),
    )


def draftAbandoned(state: DialogueState, ctx: StepContext) -> DialogueState | None:
    if not (state.agent is _FORMULATING and state.draft is _BUILDING and _continuing_speech(ctx)):
        return None
    return replace(
        state,
        agent=_LISTENING,
        draft=_DROPPED,
        timers=replace(state.timers, idle_ms=0, overlap_ms=0, speech_left_ms=0),
    )


def formulatingWaits(state: DialogueState, ctx: StepContext) -> DialogueState | None:
    """`Formulating`'s "still generating, nothing happened" self-loop
    (defect #2). Added to `dialogue.qnt` after T-09's implementation found
    the state had no legal action for a quiet tick -- every other state has
    one (`greetingPlays`, `idleTicks`, `userKeepsTalking`, `speechPlays`,
    `closingPlays`), `Formulating` originally only had `draftReady` and
    `draftAbandoned`, both of which end the state. Real generation takes
    real time and an orchestration that steps the machine on every incoming
    audio chunk (T-09's `backend/ws/session.py`) needs somewhere to land
    when neither of those guards fires yet. Mirrors `dialogue.qnt`'s own
    effect exactly: only `idle_ms`/`overlap_ms` reset, `turn_ms` and
    `speech_left_ms` are left untouched.
    """
    if not (state.agent is _FORMULATING and state.draft is _BUILDING):
        return None
    return replace(state, timers=replace(state.timers, idle_ms=0, overlap_ms=0))


# ---------------------------------------------------------------------------
# speaking
# ---------------------------------------------------------------------------


def speechPlays(state: DialogueState, ctx: StepContext) -> DialogueState | None:
    if not (
        state.agent is _SPEAKING and state.timers.speech_left_ms > 0 and not ctx.event.user_speaking
    ):
        return None
    remaining = max(0, state.timers.speech_left_ms - ctx.event.elapsed_ms)
    return replace(
        state,
        timers=replace(state.timers, speech_left_ms=remaining, turn_ms=0, idle_ms=0, overlap_ms=0),
    )


def speechCompletes(state: DialogueState, ctx: StepContext) -> DialogueState | None:
    if not (state.agent is _SPEAKING and state.timers.speech_left_ms == 0):
        return None
    return replace(state, agent=_LISTENING, draft=_COMMITTED, timers=DialogueTimers())


def overlapGrows(state: DialogueState, ctx: StepContext) -> DialogueState | None:
    if not (
        state.agent is _SPEAKING
        and _continuing_speech(ctx)
        and state.timers.speech_left_ms > 0
        and state.timers.overlap_ms < ctx.thresholds.overlap_limit_ms
    ):
        return None
    new_overlap_ms = state.timers.overlap_ms + ctx.event.elapsed_ms
    # speech_left_ms deliberately UNCHANGED -- see module docstring.
    return replace(state, timers=replace(state.timers, overlap_ms=new_overlap_ms, idle_ms=0))


def overlapTriggersDecision(state: DialogueState, ctx: StepContext) -> DialogueState | None:
    if not (
        state.agent is _SPEAKING
        and _continuing_speech(ctx)
        and state.timers.overlap_ms >= ctx.thresholds.overlap_limit_ms
        and state.timers.speech_left_ms > ctx.thresholds.tail_min_ms
    ):
        return None
    return replace(state, agent=_DECIDING_BARGE_IN, timers=replace(state.timers, idle_ms=0))


def finishTailThroughOverlap(state: DialogueState, ctx: StepContext) -> DialogueState | None:
    """FR-15: overlap arrives over a tail too short to bother the LLM with
    (`speech_left_ms <= tail_min_ms`) -- play the tail out, no `decide()`
    call. This is `dialogue.qnt`'s patched deadlock (module comment there:
    "FOUND BY THE MODEL CHECKER, not by design review")."""
    if not (
        state.agent is _SPEAKING
        and _continuing_speech(ctx)
        and 0 < state.timers.speech_left_ms <= ctx.thresholds.tail_min_ms
        and state.timers.overlap_ms >= ctx.thresholds.overlap_limit_ms
    ):
        return None
    remaining = max(0, state.timers.speech_left_ms - ctx.event.elapsed_ms)
    return replace(state, timers=replace(state.timers, speech_left_ms=remaining, idle_ms=0))


# ---------------------------------------------------------------------------
# barge-in decision
# ---------------------------------------------------------------------------


def bargeInAccepted(state: DialogueState, ctx: StepContext) -> DialogueState | None:
    decision = ctx.event.barge_in_decision
    if not (state.agent is _DECIDING_BARGE_IN and decision is not None and decision.interrupt):
        return None
    return replace(
        state,
        agent=_LISTENING,
        draft=_COMMITTED,
        timers=replace(state.timers, speech_left_ms=0, overlap_ms=0, idle_ms=0),
    )


def bargeInDeclined(state: DialogueState, ctx: StepContext) -> DialogueState | None:
    decision = ctx.event.barge_in_decision
    if not (state.agent is _DECIDING_BARGE_IN and decision is not None and not decision.interrupt):
        return None
    return replace(state, agent=_SPEAKING, timers=replace(state.timers, overlap_ms=0, idle_ms=0))


# ---------------------------------------------------------------------------
# closing
# ---------------------------------------------------------------------------


def closingPlays(state: DialogueState, ctx: StepContext) -> DialogueState | None:
    if not (state.agent is _CLOSING and state.timers.speech_left_ms > 0):
        return None
    remaining = max(0, state.timers.speech_left_ms - ctx.event.elapsed_ms)
    return replace(state, timers=replace(state.timers, speech_left_ms=remaining, overlap_ms=0))


def closingInterrupted(state: DialogueState, ctx: StepContext) -> DialogueState | None:
    if not (state.agent is _CLOSING and _continuing_speech(ctx)):
        return None
    return replace(
        state,
        agent=_LISTENING,
        draft=_NO_DRAFT,
        timers=DialogueTimers(turn_ms=ctx.event.elapsed_ms),
    )


def closingCompletes(state: DialogueState, ctx: StepContext) -> DialogueState | None:
    if not (
        state.agent is _CLOSING
        and state.timers.speech_left_ms == 0
        and not ctx.event.user_speaking
    ):
        return None
    return replace(
        state,
        agent=_ENDED,
        timers=replace(state.timers, turn_ms=0, overlap_ms=0, speech_left_ms=0),
    )


# ---------------------------------------------------------------------------
# ended
# ---------------------------------------------------------------------------


def ended(state: DialogueState, ctx: StepContext) -> DialogueState | None:
    if state.agent is not _ENDED:
        return None
    return state


# ---------------------------------------------------------------------------
# registries -- single source of truth for machine.py's graph wiring AND
# for the "24 actions <-> graph edges, both directions" test/cross-check
# (tasks.md T-07 "Проверка", "Финальная проверка" §2).
# ---------------------------------------------------------------------------

#: Every `dialogue.qnt` `step` action, in the exact order the model lists
#: them (`action step = any { ... }`) -- 25 entries, verified by
#: `test_state_machine.py`'s registry-vs-`.qnt` cross-check (parses
#: `specs/formal/dialogue.qnt` directly rather than comparing against a
#: second hand-kept copy -- see that test module's docstring, defect #3).
ACTIONS: dict[str, ActionFn] = {
    "greetingPlays": greetingPlays,
    "greetingFinishes": greetingFinishes,
    "userInterruptsGreeting": userInterruptsGreeting,
    "userStartsSpeaking": userStartsSpeaking,
    "userKeepsTalking": userKeepsTalking,
    "userTurnEnds": userTurnEnds,
    "reachTalkLimit": reachTalkLimit,
    "idleTicks": idleTicks,
    "idleHangup": idleHangup,
    "interjectDeclined": interjectDeclined,
    "interjectAccepted": interjectAccepted,
    "draftReady": draftReady,
    "draftAbandoned": draftAbandoned,
    "formulatingWaits": formulatingWaits,
    "speechPlays": speechPlays,
    "speechCompletes": speechCompletes,
    "overlapGrows": overlapGrows,
    "overlapTriggersDecision": overlapTriggersDecision,
    "finishTailThroughOverlap": finishTailThroughOverlap,
    "bargeInAccepted": bargeInAccepted,
    "bargeInDeclined": bargeInDeclined,
    "closingPlays": closingPlays,
    "closingInterrupted": closingInterrupted,
    "closingCompletes": closingCompletes,
    "ended": ended,
}

#: Destination `AgentState` for each action -- the source is implied by
#: `CANDIDATES_BY_STATE` below (an action may be a candidate of more than
#: one source state only for `userStartsSpeaking`, which self-loops from
#: every live state; its destination is always "unchanged", encoded here
#: as `None` and resolved to the current state by `machine.py`).
ACTION_DESTINATION: dict[str, AgentState | None] = {
    "greetingPlays": _GREETING,
    "greetingFinishes": _LISTENING,
    "userInterruptsGreeting": _LISTENING,
    "userStartsSpeaking": None,
    "userKeepsTalking": _LISTENING,
    "userTurnEnds": _FORMULATING,
    "reachTalkLimit": _DECIDING_INTERJECT,
    "idleTicks": _LISTENING,
    "idleHangup": _CLOSING,
    "interjectDeclined": _LISTENING,
    "interjectAccepted": _FORMULATING,
    "draftReady": _SPEAKING,
    "draftAbandoned": _LISTENING,
    "formulatingWaits": _FORMULATING,
    "speechPlays": _SPEAKING,
    "speechCompletes": _LISTENING,
    "overlapGrows": _SPEAKING,
    "overlapTriggersDecision": _DECIDING_BARGE_IN,
    "finishTailThroughOverlap": _SPEAKING,
    "bargeInAccepted": _LISTENING,
    "bargeInDeclined": _SPEAKING,
    "closingPlays": _CLOSING,
    "closingInterrupted": _LISTENING,
    "closingCompletes": _ENDED,
    "ended": _ENDED,
}

#: Ordered candidates per source state -- `machine.py`'s node functions try
#: these in order and apply the first whose guard fires. Order only matters
#: where the model's `any { }` leaves a real choice open (documented at the
#: call site in machine.py); everywhere else the guards are mutually
#: exclusive by construction and order is cosmetic.
CANDIDATES_BY_STATE: dict[AgentState, tuple[str, ...]] = {
    _GREETING: ("userInterruptsGreeting", "greetingFinishes", "greetingPlays", "userStartsSpeaking"),
    _LISTENING: (
        "userTurnEnds",
        "reachTalkLimit",
        "userKeepsTalking",
        "idleHangup",
        "idleTicks",
        "userStartsSpeaking",
    ),
    _DECIDING_INTERJECT: ("interjectDeclined", "interjectAccepted", "userStartsSpeaking"),
    _FORMULATING: ("draftAbandoned", "draftReady", "formulatingWaits", "userStartsSpeaking"),
    _SPEAKING: (
        "overlapTriggersDecision",
        "finishTailThroughOverlap",
        "overlapGrows",
        "speechCompletes",
        "speechPlays",
        "userStartsSpeaking",
    ),
    _DECIDING_BARGE_IN: ("bargeInAccepted", "bargeInDeclined", "userStartsSpeaking"),
    _CLOSING: ("closingInterrupted", "closingCompletes", "closingPlays", "userStartsSpeaking"),
    _ENDED: ("ended",),
}


__all__ = [
    "DeadlockError",
    "DialogueThresholds",
    "AutomatonInput",
    "StepContext",
    "ActionFn",
    "ACTIONS",
    "ACTION_DESTINATION",
    "CANDIDATES_BY_STATE",
    # the 24 actions, exported individually so tests/machine.py can
    # reference them by name instead of only through the registries
    "greetingPlays",
    "greetingFinishes",
    "userInterruptsGreeting",
    "userStartsSpeaking",
    "userKeepsTalking",
    "userTurnEnds",
    "reachTalkLimit",
    "idleTicks",
    "idleHangup",
    "interjectDeclined",
    "interjectAccepted",
    "draftReady",
    "draftAbandoned",
    "formulatingWaits",
    "speechPlays",
    "speechCompletes",
    "overlapGrows",
    "overlapTriggersDecision",
    "finishTailThroughOverlap",
    "bargeInAccepted",
    "bargeInDeclined",
    "closingPlays",
    "closingInterrupted",
    "closingCompletes",
    "ended",
]
