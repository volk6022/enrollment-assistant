"""Unit tests for `backend.dialogue.{nodes,machine}` (T-07).

Structure mirrors tasks.md T-07's "Проверка" list plus the "Финальная
проверка" §2 cross-check:

* `test_exactly_24_actions_...` / `test_every_candidate_...` /
  `test_every_action_has_a_declared_destination` -- the "24 действия
  модели, 24 ребра графа, соответствие в обе стороны" table, built from
  `nodes.ACTIONS`/`CANDIDATES_BY_STATE`/`ACTION_DESTINATION` (the actual
  data `machine.py` uses to build `add_conditional_edges` path_maps) rather
  than by inspecting the compiled graph's `get_graph().edges`: LangGraph
  deduplicates conditional edges that share a (source, target) pair for
  drawing purposes (verified empirically -- `userKeepsTalking`,
  `overlapGrows`, `speechPlays`, `userInterruptsGreeting` all vanish from
  `get_graph().edges` because another action from the same source already
  targets the same destination), so it is not a faithful source for this
  check. The registries are what's actually passed to `add_conditional_edges`,
  so they are the correct thing to assert against.
* `test_fr08_...` -- FR-08: declining to interject must not reset `turn_ms`.
* `test_fr15_...` -- FR-15: overlap over a tail `<= TAIL_MIN` never calls
  the decision client.
* `test_fr03_...` -- FR-03: cutting the greeting never calls the decision
  client either (nothing to decide -- the greeting is pre-recorded).
* `test_commit_*` (three cases) -- memory.md §5's table at the automaton
  level: draft abandoned before any audio never reaches `Draft.COMMITTED`;
  a fully-played reply does; a yielded-mid-reply reply also does (T-06's
  `DialogueMemory.commit_agent_turn` is what actually marks it partial from
  `delivered_ms < planned_ms` -- this module only has to get `Draft` right,
  which `inv_dropped_never_committed` in `dialogue.qnt` hinges on).

`FakeDecisionClient` stands in for `LlamaClient` (contracts/llm.md §1's
`decide()` shape) -- no real HTTP, per the project's "mock only external
APIs" rule; recording every call lets tests assert decide() was (or was
not) invoked, which is the observable half of FR-17 ("классификатор
перебивания не опрашивается в цикле").
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from backend.dialogue.machine import AutomatonState, DialogueMachine
from backend.dialogue.models import AgentState, DialogueState, DialogueTimers, Draft
from backend.dialogue.nodes import (
    ACTION_DESTINATION,
    ACTIONS,
    CANDIDATES_BY_STATE,
    AutomatonInput,
    DialogueThresholds,
)

# ---------------------------------------------------------------------------
# ground truth: the 24 action names in `specs/formal/dialogue.qnt`'s
# `action step = any { ... }` block, in the order they're listed there.
# ---------------------------------------------------------------------------

QNT_STEP_ACTIONS: frozenset[str] = frozenset(
    {
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
    }
)

# `dialogue.qnt`'s STATE_MACHINE.md diagram, expanded into (source, action,
# dest) triples -- the graph edges `machine.py._build_graph()` wires via
# `CANDIDATES_BY_STATE`/`ACTION_DESTINATION`. `userStartsSpeaking` appears
# once per live (non-Ended) state as a self-loop, exactly as
# `dialogue.qnt`'s guard (`agent != Ended`, no state-specific restriction)
# allows.
EXPECTED_EDGES: frozenset[tuple[str, str, str]] = frozenset(
    {
        ("Greeting", "userInterruptsGreeting", "Listening"),
        ("Greeting", "greetingFinishes", "Listening"),
        ("Greeting", "greetingPlays", "Greeting"),
        ("Greeting", "userStartsSpeaking", "Greeting"),
        ("Listening", "userTurnEnds", "Formulating"),
        ("Listening", "reachTalkLimit", "DecidingInterject"),
        ("Listening", "userKeepsTalking", "Listening"),
        ("Listening", "idleHangup", "Closing"),
        ("Listening", "idleTicks", "Listening"),
        ("Listening", "userStartsSpeaking", "Listening"),
        ("DecidingInterject", "interjectDeclined", "Listening"),
        ("DecidingInterject", "interjectAccepted", "Formulating"),
        ("DecidingInterject", "userStartsSpeaking", "DecidingInterject"),
        ("Formulating", "draftAbandoned", "Listening"),
        ("Formulating", "draftReady", "Speaking"),
        ("Formulating", "userStartsSpeaking", "Formulating"),
        ("Speaking", "overlapTriggersDecision", "DecidingBargeIn"),
        ("Speaking", "finishTailThroughOverlap", "Speaking"),
        ("Speaking", "overlapGrows", "Speaking"),
        ("Speaking", "speechCompletes", "Listening"),
        ("Speaking", "speechPlays", "Speaking"),
        ("Speaking", "userStartsSpeaking", "Speaking"),
        ("DecidingBargeIn", "bargeInAccepted", "Listening"),
        ("DecidingBargeIn", "bargeInDeclined", "Speaking"),
        ("DecidingBargeIn", "userStartsSpeaking", "DecidingBargeIn"),
        ("Closing", "closingInterrupted", "Listening"),
        ("Closing", "closingCompletes", "Ended"),
        ("Closing", "closingPlays", "Closing"),
        ("Closing", "userStartsSpeaking", "Closing"),
        ("Ended", "ended", "Ended"),
    }
)


def _actual_edges() -> set[tuple[str, str, str]]:
    """Re-derives the (source, action, dest) triples from the exact data
    `machine.py._build_graph()` reads (`CANDIDATES_BY_STATE`,
    `ACTION_DESTINATION`) -- the real source of truth for what
    `add_conditional_edges` is told, not a guess re-typed by hand."""
    edges: set[tuple[str, str, str]] = set()
    for agent, action_names in CANDIDATES_BY_STATE.items():
        for name in action_names:
            destination = ACTION_DESTINATION[name]
            dest_name = destination.value if destination is not None else agent.value
            edges.add((agent.value, name, dest_name))
    return edges


def test_exactly_24_actions_match_the_qnt_model() -> None:
    assert set(ACTIONS) == QNT_STEP_ACTIONS
    assert len(ACTIONS) == 24


def test_every_qnt_action_is_a_candidate_somewhere() -> None:
    """No action without an edge."""
    referenced = {name for names in CANDIDATES_BY_STATE.values() for name in names}
    assert referenced == QNT_STEP_ACTIONS


def test_every_candidate_is_a_known_qnt_action() -> None:
    """No edge without an action."""
    for agent, names in CANDIDATES_BY_STATE.items():
        for name in names:
            assert name in ACTIONS, f"{agent}: {name!r} is not one of the 24 dialogue.qnt actions"


def test_action_destination_is_declared_for_every_action() -> None:
    assert set(ACTION_DESTINATION) == QNT_STEP_ACTIONS


def test_graph_edges_match_the_state_machine_diagram() -> None:
    """The full (source, action, dest) correspondence, both directions,
    against `specs/formal/STATE_MACHINE.md`'s mermaid diagram."""
    assert _actual_edges() == EXPECTED_EDGES


def test_ended_state_has_only_the_self_loop() -> None:
    assert CANDIDATES_BY_STATE[AgentState.ENDED] == ("ended",)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@dataclass
class FakeDecisionClient:
    """Stands in for `LlamaClient.decide()` (contracts/llm.md §1) -- records
    every call instead of making one, and returns whatever canned decision
    the test configured."""

    interject_response: dict | None = None
    barge_in_response: dict | None = None
    calls: list[tuple[str, dict]] = field(default_factory=list)

    async def decide(self, prompt: str, schema: dict, *, max_tokens: int = 256) -> dict:
        self.calls.append((prompt, schema))
        if "interject" in schema["properties"]:
            assert self.interject_response is not None, "test did not configure an interject response"
            return self.interject_response
        assert self.barge_in_response is not None, "test did not configure a barge-in response"
        return self.barge_in_response


_THRESHOLDS = DialogueThresholds(
    turn_limit_ms=20_000, idle_limit_ms=2_000, overlap_limit_ms=1_000, tail_min_ms=2_000
)


def _machine(client: FakeDecisionClient) -> DialogueMachine:
    return DialogueMachine(decision_client=client, thresholds=_THRESHOLDS)


# ---------------------------------------------------------------------------
# FR-03: greeting is cut immediately, no LLM call -- nothing to decide.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fr03_greeting_is_cut_without_calling_the_decision_client() -> None:
    client = FakeDecisionClient()
    machine = _machine(client)
    automaton = AutomatonState(
        dialogue=DialogueState(agent=AgentState.GREETING, timers=DialogueTimers(speech_left_ms=1500))
    )

    # First tick establishes the speech-start edge (dialogue.qnt's
    # `userStartsSpeaking`) -- agent unchanged, matches the model's
    # two-step decomposition of "silence -> speech".
    automaton = await machine.step(automaton, AutomatonInput(user_speaking=True, elapsed_ms=0))
    assert automaton.dialogue.agent is AgentState.GREETING

    # Second tick: speech is now ongoing -> userInterruptsGreeting cuts it.
    automaton = await machine.step(automaton, AutomatonInput(user_speaking=True, elapsed_ms=40))
    assert automaton.dialogue.agent is AgentState.LISTENING
    assert automaton.dialogue.draft is Draft.NO_DRAFT
    assert client.calls == []


@pytest.mark.asyncio
async def test_fr03_uninterrupted_greeting_finishes_on_its_own_and_calls_nothing() -> None:
    client = FakeDecisionClient()
    machine = _machine(client)
    automaton = AutomatonState(
        dialogue=DialogueState(agent=AgentState.GREETING, timers=DialogueTimers(speech_left_ms=300))
    )
    automaton = await machine.step(automaton, AutomatonInput(user_speaking=False, elapsed_ms=300))
    automaton = await machine.step(automaton, AutomatonInput(user_speaking=False, elapsed_ms=0))
    assert automaton.dialogue.agent is AgentState.LISTENING
    assert client.calls == []


# ---------------------------------------------------------------------------
# FR-07/FR-08: interject decision, and the turn timer NOT resetting on
# "keep listening".
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fr08_declining_to_interject_does_not_reset_the_turn_timer() -> None:
    client = FakeDecisionClient(
        interject_response={"interject": False, "understood": "still on topic", "reason": "coherent"}
    )
    machine = _machine(client)
    automaton = AutomatonState(
        dialogue=DialogueState(agent=AgentState.LISTENING, timers=DialogueTimers(turn_ms=20_000)),
        user_was_speaking=True,
    )

    automaton = await machine.step(automaton, AutomatonInput(user_speaking=True, elapsed_ms=50))
    assert automaton.dialogue.agent is AgentState.DECIDING_INTERJECT

    automaton = await machine.step(
        automaton,
        AutomatonInput(
            user_speaking=True, elapsed_ms=50, dialogue_history="...", transcript_so_far="..."
        ),
    )
    assert automaton.dialogue.agent is AgentState.LISTENING
    assert automaton.dialogue.timers.turn_ms == 20_000, (
        "FR-08: turn_ms must survive 'keep listening' unchanged, or a brief "
        "slow-down buys the speaker another full TALK_LIMIT"
    )
    assert len(client.calls) == 1

    # And because turn_ms was never reset, the VERY NEXT tick immediately
    # re-triggers a decision instead of quietly resuming userKeepsTalking --
    # this is what FR-08 is actually for.
    automaton = await machine.step(automaton, AutomatonInput(user_speaking=True, elapsed_ms=10))
    assert automaton.dialogue.agent is AgentState.DECIDING_INTERJECT


@pytest.mark.asyncio
async def test_interject_accepted_moves_to_formulating_with_a_fresh_draft() -> None:
    client = FakeDecisionClient(
        interject_response={"interject": True, "understood": "wants directions", "reason": "rambling"}
    )
    machine = _machine(client)
    automaton = AutomatonState(
        dialogue=DialogueState(agent=AgentState.DECIDING_INTERJECT, timers=DialogueTimers(turn_ms=20_000)),
        user_was_speaking=True,
    )
    automaton = await machine.step(
        automaton, AutomatonInput(user_speaking=True, dialogue_history="h", transcript_so_far="t")
    )
    assert automaton.dialogue.agent is AgentState.FORMULATING
    assert automaton.dialogue.draft is Draft.BUILDING


# ---------------------------------------------------------------------------
# FR-13/FR-15: barge-in decision, and the short-tail patch that needs none.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fr15_overlap_over_a_short_tail_never_calls_the_decision_client() -> None:
    client = FakeDecisionClient()
    machine = _machine(client)
    automaton = AutomatonState(
        dialogue=DialogueState(
            agent=AgentState.SPEAKING,
            draft=Draft.VOICING,
            timers=DialogueTimers(speech_left_ms=1_500, overlap_ms=1_000),  # <= tail_min_ms
        ),
        user_was_speaking=True,
    )

    automaton = await machine.step(automaton, AutomatonInput(user_speaking=True, elapsed_ms=200))

    assert automaton.dialogue.agent is AgentState.SPEAKING, "should finish the tail, not hand off"
    assert automaton.dialogue.timers.speech_left_ms == 1_300
    assert client.calls == [], "FR-15: nothing left worth interrupting, no decide() call"


@pytest.mark.asyncio
async def test_overlap_over_a_long_tail_does_trigger_a_decision() -> None:
    client = FakeDecisionClient(barge_in_response={"interrupt": False, "reason": "just acknowledging"})
    machine = _machine(client)
    automaton = AutomatonState(
        dialogue=DialogueState(
            agent=AgentState.SPEAKING,
            draft=Draft.VOICING,
            timers=DialogueTimers(speech_left_ms=5_000, overlap_ms=1_000),  # > tail_min_ms
        ),
        user_was_speaking=True,
    )

    automaton = await machine.step(automaton, AutomatonInput(user_speaking=True, elapsed_ms=200))
    assert automaton.dialogue.agent is AgentState.DECIDING_BARGE_IN

    automaton = await machine.step(
        automaton,
        AutomatonInput(
            user_speaking=True,
            dialogue_history="h",
            draft_answer="d",
            interlocutor_transcript="ага понятно",
            total_answer_ms=8_000,
        ),
    )
    assert automaton.dialogue.agent is AgentState.SPEAKING, "declined -- agent keeps talking"
    assert automaton.dialogue.timers.overlap_ms == 0
    assert len(client.calls) == 1


# ---------------------------------------------------------------------------
# commit / drop: the three cases `inv_dropped_never_committed` cares about.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_commit_case_1_draft_abandoned_before_any_audio_is_dropped_not_committed() -> None:
    client = FakeDecisionClient()
    machine = _machine(client)
    automaton = AutomatonState(
        dialogue=DialogueState(agent=AgentState.FORMULATING, draft=Draft.BUILDING),
        user_was_speaking=True,
    )
    automaton = await machine.step(automaton, AutomatonInput(user_speaking=True, elapsed_ms=100))
    assert automaton.dialogue.agent is AgentState.LISTENING
    assert automaton.dialogue.draft is Draft.DROPPED, (
        "FR-12: nothing was voiced, so this must never look like a committed reply"
    )


@pytest.mark.asyncio
async def test_commit_case_2_fully_played_reply_is_committed() -> None:
    client = FakeDecisionClient()
    machine = _machine(client)
    automaton = AutomatonState(
        dialogue=DialogueState(agent=AgentState.SPEAKING, draft=Draft.VOICING, timers=DialogueTimers(speech_left_ms=200)),
    )
    automaton = await machine.step(automaton, AutomatonInput(user_speaking=False, elapsed_ms=200))
    assert automaton.dialogue.timers.speech_left_ms == 0
    automaton = await machine.step(automaton, AutomatonInput(user_speaking=False, elapsed_ms=0))
    assert automaton.dialogue.agent is AgentState.LISTENING
    assert automaton.dialogue.draft is Draft.COMMITTED


@pytest.mark.asyncio
async def test_commit_case_3_yielded_mid_reply_is_committed_as_partial() -> None:
    """FR-16: yielding still commits -- the user already heard part of it."""
    client = FakeDecisionClient(barge_in_response={"interrupt": True, "reason": "asked something else"})
    machine = _machine(client)
    automaton = AutomatonState(
        dialogue=DialogueState(
            agent=AgentState.DECIDING_BARGE_IN,
            draft=Draft.VOICING,
            timers=DialogueTimers(speech_left_ms=4_000, overlap_ms=1_000),
        ),
        user_was_speaking=True,
    )
    automaton = await machine.step(
        automaton,
        AutomatonInput(
            user_speaking=True,
            dialogue_history="h",
            draft_answer="d",
            interlocutor_transcript="нет подождите, я про другое",
            total_answer_ms=8_000,
        ),
    )
    assert automaton.dialogue.agent is AgentState.LISTENING
    assert automaton.dialogue.draft is Draft.COMMITTED, (
        "FR-16: the user heard part of this reply -- it must reach memory, "
        "marked partial by DialogueMemory.commit_agent_turn (T-06) from "
        "delivered_ms < planned_ms, which this automaton makes true by "
        "leaving speech_left_ms > 0 at commit time"
    )
    assert automaton.dialogue.timers.speech_left_ms == 0  # audio is cut immediately on yield


# ---------------------------------------------------------------------------
# idle hangup / closing cancel (FR-25/FR-26) -- quick sanity, not required
# by tasks.md but cheap given the fixtures above.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_idle_after_answer_leads_to_closing_and_closing_is_cancellable() -> None:
    client = FakeDecisionClient()
    machine = _machine(client)
    automaton = AutomatonState(
        dialogue=DialogueState(agent=AgentState.LISTENING, timers=DialogueTimers(idle_ms=2_000)),
    )
    automaton = await machine.step(
        automaton, AutomatonInput(user_speaking=False, elapsed_ms=0, farewell_duration_ms=1_000)
    )
    assert automaton.dialogue.agent is AgentState.CLOSING
    assert automaton.dialogue.timers.speech_left_ms == 1_000

    automaton = await machine.step(automaton, AutomatonInput(user_speaking=True, elapsed_ms=0))
    assert automaton.dialogue.agent is AgentState.CLOSING  # speech-start edge only
    automaton = await machine.step(automaton, AutomatonInput(user_speaking=True, elapsed_ms=30))
    assert automaton.dialogue.agent is AgentState.LISTENING, "FR-26: late speech cancels closing"
    assert client.calls == []
