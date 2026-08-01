"""LangGraph automaton: one node per `dialogue.qnt` `AgentState`, including
`Ended` (tasks.md T-07; plan.md §7 "Реализация — LangGraph"). Transitions go
through `add_conditional_edges` only -- `add_edge` has no `condition`
parameter, so any code that reaches for one is wrong on its face.

How one `.qnt` step maps onto one LangGraph node visit
--------------------------------------------------------------------------
LangGraph keeps following conditional edges within a single `ainvoke()`
call until a route resolves to `END` (verified experimentally: two chained
nodes without a stop condition hit `GraphRecursionError`, not "one hop and
return"). But `dialogue.qnt`'s `step` action fires exactly ONE of its 24
actions per model step -- so the destination node reached by a real
transition must NOT itself re-evaluate its own candidates against the same
external event, or a single VAD tick could fire two actions
(e.g. `reachTalkLimit` immediately followed by `userKeepsTalking` on the
`Listening` node it just cascaded into).

The fix is a `dispatched` flag on the internal graph-working state
(`_GraphState`, distinct from `AutomatonState`/`DialogueState`, invisible
to callers): every node checks it first and, if already set, does nothing
but route straight to `END`. Only the node that is the actual entry point
for this call (chosen by `state.dialogue.agent`, via a conditional edge out
of `START`) ever evaluates candidates; any node reached afterwards by a
real (non-self-loop) transition sees `dispatched=True` and passes through.
This is implementation plumbing, not a 25th action -- `DialogueMachine.step`
is the one qnt-step-per-call boundary; `tests/unit/test_state_machine.py`
asserts the "no double-fire" property directly.

Two of the eight nodes do real I/O -- `DecidingInterject` and
`DecidingBargeIn` call `LlamaClient.decide()` (via the narrow
`DecisionClient` protocol below, so tests need no real HTTP) using the
prompts from `backend/llm/prompts.py` and the schemas from
`backend/llm/schemas.py` (contracts/llm.md §4-5), exactly once per node
visit that doesn't already have a decision supplied on `AutomatonInput`.
The other six nodes are pure and synchronous; every node function is
declared `async def` anyway so the compiled graph can be driven uniformly
with `.ainvoke()`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Protocol

from langgraph.graph import END, START, StateGraph

from backend.dialogue.models import AgentState, DialogueState
from backend.dialogue.nodes import (
    ACTION_DESTINATION,
    ACTIONS,
    CANDIDATES_BY_STATE,
    AutomatonInput,
    DeadlockError,
    DialogueThresholds,
    StepContext,
)
from backend.llm.prompts import build_barge_in_prompt, build_interject_prompt
from backend.llm.schemas import BargeInDecision, InterjectDecision


class DecisionClient(Protocol):
    """Structural match for `backend.llm.client.LlamaClient.decide` (llm.md
    §1) -- narrow on purpose so tests can pass a stub that never touches
    HTTP instead of a real `LlamaClient`, per the project's "mock only
    external APIs" rule."""

    async def decide(
        self, prompt: str, schema: dict[str, Any], *, max_tokens: int = 256
    ) -> dict[str, Any]: ...


@dataclass
class AutomatonState:
    """The one extra bit `dialogue.qnt`'s `userSpeaking` needs that
    `DialogueState` (T-06) deliberately excludes: the PREVIOUS speech
    level, so `machine.py` can derive `userStartsSpeaking`'s edge itself
    (`speech_started = event.user_speaking and not user_was_speaking`)
    instead of requiring every caller to compute that edge correctly on
    every call. This type belongs to T-07, not T-06 -- it wraps
    `DialogueState`, it does not extend or replace it.
    """

    dialogue: DialogueState = field(default_factory=DialogueState)
    user_was_speaking: bool = False


@dataclass
class _GraphState:
    """LangGraph's working schema for one `DialogueMachine.step()` call.
    `dispatched`/`route_key`/`last_action` are call-scoped plumbing, reset
    fresh by `step()` every time -- never part of `AutomatonState`, which
    is the only thing that survives between calls."""

    automaton: AutomatonState
    event: AutomatonInput
    thresholds: DialogueThresholds
    dispatched: bool = False
    route_key: str = ""
    last_action: str | None = None


def _route_by_key(gs: _GraphState) -> str:
    return gs.route_key


def _entry_route(gs: _GraphState) -> str:
    return gs.automaton.dialogue.agent.value


def _speech_started(gs: _GraphState) -> bool:
    return gs.event.user_speaking and not gs.automaton.user_was_speaking


def _dispatch(
    agent: AgentState,
    dialogue: DialogueState,
    event: AutomatonInput,
    speech_started: bool,
    thresholds: DialogueThresholds,
) -> dict[str, Any]:
    """Shared candidate-evaluation loop: try `agent`'s candidates in order,
    apply the first whose guard fires. Used by every node -- the six pure
    ones directly, the two LLM-backed ones after they've resolved a
    decision (or found they don't need one this step)."""
    ctx = StepContext(event=event, speech_started=speech_started, thresholds=thresholds)
    for name in CANDIDATES_BY_STATE[agent]:
        result = ACTIONS[name](dialogue, ctx)
        if result is not None:
            new_automaton = AutomatonState(dialogue=result, user_was_speaking=event.user_speaking)
            return {
                "automaton": new_automaton,
                "dispatched": True,
                "route_key": name,
                "last_action": name,
            }
    raise DeadlockError(agent)


def _build_simple_node(agent: AgentState):
    async def node(gs: _GraphState) -> dict[str, Any]:
        if gs.dispatched:
            return {"route_key": "__done__"}
        return _dispatch(
            agent, gs.automaton.dialogue, gs.event, _speech_started(gs), gs.thresholds
        )

    return node


class DialogueMachine:
    """Compiles the `dialogue.qnt`-equivalent LangGraph once, then drives it
    one event at a time via `step()`. Owns the two LLM decision calls
    (FR-07/FR-13); everything else is pure state transition."""

    def __init__(
        self,
        *,
        decision_client: DecisionClient,
        thresholds: DialogueThresholds,
        decision_max_tokens: int = 256,
    ) -> None:
        self._decision_client = decision_client
        self._thresholds = thresholds
        self._decision_max_tokens = decision_max_tokens
        self._graph = self._build_graph()

    async def step(self, automaton: AutomatonState, event: AutomatonInput) -> AutomatonState:
        """Applies exactly one `dialogue.qnt` action -- whichever one of
        the current state's candidates fires for `event` -- and returns the
        resulting `AutomatonState`. Call again with the next event; there
        is no separate "advance until quiescent" mode, by design (module
        docstring)."""
        initial = _GraphState(automaton=automaton, event=event, thresholds=self._thresholds)
        result = await self._graph.ainvoke(initial)
        return result["automaton"]

    async def _node_deciding_interject(self, gs: _GraphState) -> dict[str, Any]:
        if gs.dispatched:
            return {"route_key": "__done__"}
        event = gs.event
        speech_started = _speech_started(gs)
        if event.interject_decision is None and not speech_started:
            prompt = build_interject_prompt(
                dialogue_history=event.dialogue_history,
                transcript_so_far=event.transcript_so_far,
            )
            raw = await self._decision_client.decide(
                prompt, InterjectDecision.SCHEMA, max_tokens=self._decision_max_tokens
            )
            event = replace(event, interject_decision=InterjectDecision.model_validate(raw))
        return _dispatch(
            AgentState.DECIDING_INTERJECT,
            gs.automaton.dialogue,
            event,
            speech_started,
            gs.thresholds,
        )

    async def _node_deciding_barge_in(self, gs: _GraphState) -> dict[str, Any]:
        if gs.dispatched:
            return {"route_key": "__done__"}
        event = gs.event
        speech_started = _speech_started(gs)
        if event.barge_in_decision is None and not speech_started:
            total_ms = event.total_answer_ms or 0
            speech_left_ms = gs.automaton.dialogue.timers.speech_left_ms
            voiced_seconds = max(0.0, (total_ms - speech_left_ms) / 1000)
            total_seconds = total_ms / 1000
            prompt = build_barge_in_prompt(
                dialogue_history=event.dialogue_history,
                draft_answer=event.draft_answer,
                voiced_seconds=voiced_seconds,
                total_seconds=total_seconds,
                interlocutor_transcript=event.interlocutor_transcript,
            )
            raw = await self._decision_client.decide(
                prompt, BargeInDecision.SCHEMA, max_tokens=self._decision_max_tokens
            )
            event = replace(event, barge_in_decision=BargeInDecision.model_validate(raw))
        return _dispatch(
            AgentState.DECIDING_BARGE_IN,
            gs.automaton.dialogue,
            event,
            speech_started,
            gs.thresholds,
        )

    def _build_graph(self):
        graph: StateGraph = StateGraph(_GraphState)

        for agent in AgentState:
            node_name = agent.value
            if agent is AgentState.DECIDING_INTERJECT:
                graph.add_node(node_name, self._node_deciding_interject)
            elif agent is AgentState.DECIDING_BARGE_IN:
                graph.add_node(node_name, self._node_deciding_barge_in)
            else:
                graph.add_node(node_name, _build_simple_node(agent))

        graph.add_conditional_edges(
            START, _entry_route, {a.value: a.value for a in AgentState}
        )

        for agent in AgentState:
            node_name = agent.value
            path_map: dict[str, str] = {"__done__": END}
            for action_name in CANDIDATES_BY_STATE[agent]:
                destination = ACTION_DESTINATION[action_name]
                # `userStartsSpeaking` has no fixed destination in
                # ACTION_DESTINATION (it never changes `agent`) -- route it
                # back to the node it fired from.
                path_map[action_name] = destination.value if destination is not None else node_name
            graph.add_conditional_edges(node_name, _route_by_key, path_map)

        return graph.compile()


__all__ = ["DecisionClient", "AutomatonState", "DialogueMachine"]
