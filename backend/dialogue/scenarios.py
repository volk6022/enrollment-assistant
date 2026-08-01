"""`ScenarioRegistry` -- loads, validates and matches `dialogue/scenarios.yaml`
(T-08, FR-21..FR-24).

`dialogue/scenarios.yaml` is content a non-programmer edits (FR-24: "Сценарии
редактируются без правки кода"). That only stays safe if the vocabulary of
conditions and actions it can use is *closed* and *checked at load time* --
this module is the closed vocabulary's implementation. There is no `eval()`
anywhere here and there must never be one: every condition string and every
action name is matched against an explicit, finite set of patterns (the
`_*_RE` regexes below, `ActionKind`'s six members), and
anything that doesn't match raises `ScenarioValidationError` naming the file
and line, right at `ScenarioRegistry.load()` -- which callers (T-09/app.py)
are expected to call once at process startup, so a typo in the YAML fails
the same way a missing `.env` variable fails in `backend/config.py`: loudly,
before a single session runs, not silently on some live caller's third
question.

`ScenarioValidationError` is the one exception type this module raises, and
it is a load-time-only concern: the YAML is wrong (bad syntax, unknown
condition/action/state, duplicate keys, a param of the wrong shape). Raised
inside `ScenarioRegistry.load()`, i.e. at startup, before any session
exists.

At *match* time, by contrast, a `ScenarioContext` field being `None` is not
an error -- it is the normal way "this decision hasn't happened yet" is
represented. `ScenarioRegistry.match(state, context)` evaluates every
scenario whose `entry_states` contains `state`, not just the one that ends
up winning: in `Listening`, `stt_no_speech`'s `transcript.empty` and
`user_says_goodbye`'s `intent == "goodbye"` are both entry_states-Listening
candidates evaluated on the same call, and when the transcript actually is
empty there was no transcript to run `IntentDecision` on, so `intent`/
`confidence` are legitimately unset. Every leaf condition here treats an
unset field it needs as simply "not satisfied" (`None == "goodbye"` is
`False`; ordering comparisons guard explicitly since `None < 0.7` raises
`TypeError`) rather than raising -- the file's own `priority` ordering
(`user_says_goodbye: 90` vs. `stt_no_speech: 80`) is what keeps the right
scenario winning when more than one candidate's `when` happens to be true,
not an exception unwinding the losing candidates' evaluation.

`intent`/`confidence` in `ScenarioContext` are meant to be populated from
nothing but `IntentDecision` (`contracts/llm.md` §4.3) -- this module does
not compute or guess either value itself, it only compares whatever the
caller put in `ScenarioContext` against the literals written in
`scenarios.yaml`.

Threshold literals written directly into `when:` clauses (`confidence >=
0.7`, `rag.max_score < 0.3`, ...) are scenario-authored business rules --
exactly the kind of thing FR-24 lets a non-programmer tune without touching
code, the same way `answer_question`'s `priority: 50` is scenario-authored.
They are a *different* category from the automaton's own timing thresholds
(`DIALOGUE_INTERJECT_AFTER_S` and friends), which live only in `.env`
(`backend/config.py`'s `DialogueSettings`) and never appear as a literal
inside any `when:` clause in the file as shipped -- this loader does not
read `.env` at all (FR-32: `backend/config.py` is the only module that
does), it only parses whatever literal is actually written in the YAML.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml

from backend.dialogue.models import AgentState

__all__ = [
    "ActionKind",
    "Scenario",
    "ScenarioAction",
    "ScenarioContext",
    "ScenarioRegistry",
    "ScenarioValidationError",
]


class ScenarioValidationError(RuntimeError):
    """`dialogue/scenarios.yaml` is malformed -- bad YAML syntax, an
    unknown condition/action/state, a duplicate key, or a param of the
    wrong shape. Always raised with the file path and the offending line
    number baked into the message (`str(err)` is printed as-is by the
    caller, same convention as `backend.config.ConfigurationError`), so
    whoever sees it in a startup log can go straight to the bad line
    instead of bisecting a 280-line YAML file by hand.
    """


# ---------------------------------------------------------------------------
# Runtime input to match(): the only thing intent/confidence ever come from
# is IntentDecision (llm.md §4.3) -- this dataclass has no logic of its own,
# it is just a typed home for whatever the caller already decided.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ScenarioContext:
    """One decision point's worth of runtime facts, handed to
    `ScenarioRegistry.match()`. Field names mirror the closed condition
    vocabulary's dotted names (`rag.count` -> `rag_count`, `intent.query` ->
    `intent_query`) so reading a condition next to the field it reads from
    is a one-to-one lookup, not a guess.

    Fields default to `None`/`""` rather than requiring the caller to
    populate all ten for every state -- a `Listening`-state match only ever
    needs `intent`/`confidence`/`transcript`; supplying `interject_accepted`
    there would be meaningless (that field only exists once
    `DecidingInterject` has run). What a *condition* requires and doesn't
    find populated raises `ScenarioContextError` at evaluation time instead
    (see module docstring) -- that is where "missing" is actually an error;
    here it is just "not applicable to this decision point yet".
    """

    intent: str | None = None
    confidence: float | None = None
    intent_query: str = ""
    rag_count: int | None = None
    rag_max_score: float | None = None
    rag_answer: str = ""
    interject_accepted: bool | None = None
    bargein_accepted: bool | None = None
    transcript: str = ""
    error_type: str | None = None


# ---------------------------------------------------------------------------
# Condition AST -- the closed vocabulary's runtime half. Every node type
# here has a literal, one-to-one counterpart in scenarios.yaml's "СЛОВАРЬ
# УСЛОВИЙ" comment block; there is no node type here without one there and
# vice versa. Compiled once at load time by `_parse_leaf_condition`/
# `_parse_condition`, evaluated many times per session by `match()`.
# ---------------------------------------------------------------------------


class _Condition(ABC):
    """Closed set of condition node types. Never subclassed outside this
    module -- adding a new condition kind means updating both this class
    hierarchy and scenarios.yaml's vocabulary comment together, in the same
    change, so the documented vocabulary and the implemented one cannot
    drift apart.
    """

    @abstractmethod
    def evaluate(self, context: ScenarioContext) -> bool: ...


@dataclass(frozen=True, slots=True)
class IntentEquals(_Condition):
    value: str

    def evaluate(self, context: ScenarioContext) -> bool:
        # `None == "goodbye"` is simply False -- no guard needed. `intent`
        # being unset means "no IntentDecision yet", which correctly makes
        # every intent-conditioned scenario inapplicable without raising.
        return context.intent == self.value


@dataclass(frozen=True, slots=True)
class ConfidenceLessThan(_Condition):
    threshold: float

    def evaluate(self, context: ScenarioContext) -> bool:
        # Ordering comparisons raise TypeError on None (unlike ==), so this
        # guard is load-bearing, not defensive style.
        return context.confidence is not None and context.confidence < self.threshold


@dataclass(frozen=True, slots=True)
class ConfidenceAtLeast(_Condition):
    threshold: float

    def evaluate(self, context: ScenarioContext) -> bool:
        return context.confidence is not None and context.confidence >= self.threshold


@dataclass(frozen=True, slots=True)
class RagCountEquals(_Condition):
    value: int

    def evaluate(self, context: ScenarioContext) -> bool:
        return context.rag_count == self.value


@dataclass(frozen=True, slots=True)
class RagMaxScoreLessThan(_Condition):
    threshold: float

    def evaluate(self, context: ScenarioContext) -> bool:
        return context.rag_max_score is not None and context.rag_max_score < self.threshold


@dataclass(frozen=True, slots=True)
class InterjectAccepted(_Condition):
    def evaluate(self, context: ScenarioContext) -> bool:
        # Undecided (None) reads as "not accepted" -- the same conservative
        # default `not interject_accepted` (interject_keep_listening) relies
        # on to win when nothing has decided otherwise yet.
        return context.interject_accepted is True


@dataclass(frozen=True, slots=True)
class BargeinAccepted(_Condition):
    def evaluate(self, context: ScenarioContext) -> bool:
        return context.bargein_accepted is True


@dataclass(frozen=True, slots=True)
class TranscriptEmpty(_Condition):
    def evaluate(self, context: ScenarioContext) -> bool:
        return context.transcript.strip() == ""


@dataclass(frozen=True, slots=True)
class ErrorTypeEquals(_Condition):
    value: str

    def evaluate(self, context: ScenarioContext) -> bool:
        # No error IS the majority state -- error_type unset simply means
        # this condition is false, same as every other leaf here.
        return context.error_type == self.value


@dataclass(frozen=True, slots=True)
class AllOf(_Condition):
    children: tuple[_Condition, ...]

    def evaluate(self, context: ScenarioContext) -> bool:
        return all(child.evaluate(context) for child in self.children)


@dataclass(frozen=True, slots=True)
class AnyOf(_Condition):
    children: tuple[_Condition, ...]

    def evaluate(self, context: ScenarioContext) -> bool:
        return any(child.evaluate(context) for child in self.children)


@dataclass(frozen=True, slots=True)
class Not(_Condition):
    child: _Condition

    def evaluate(self, context: ScenarioContext) -> bool:
        return not self.child.evaluate(context)


@dataclass(frozen=True, slots=True)
class AlwaysTrue(_Condition):
    """The condition tree of a scenario with no `when:` block at all --
    `late_question` (Closing state) matches on entry_states alone, on
    purpose: any speech during the farewell cancels it (FR-26), there is
    nothing further to decide.
    """

    def evaluate(self, context: ScenarioContext) -> bool:
        return True


_ALWAYS_TRUE = AlwaysTrue()


# ---------------------------------------------------------------------------
# Actions -- the closed vocabulary's other half.
# ---------------------------------------------------------------------------


class ActionKind(StrEnum):
    RAG_QUERY = "rag_query"
    SAY = "say"
    SAY_GENERATED = "say_generated"
    COMMIT_PARTIAL = "commit_partial"
    RESUME_SPEAKING = "resume_speaking"
    NOOP = "noop"


_ACTION_PARAM_FIELDS: dict[ActionKind, frozenset[str]] = {
    ActionKind.RAG_QUERY: frozenset({"query"}),
    ActionKind.SAY: frozenset({"text"}),
    ActionKind.SAY_GENERATED: frozenset({"instruction", "use_rag"}),
    ActionKind.COMMIT_PARTIAL: frozenset(),
    ActionKind.RESUME_SPEAKING: frozenset(),
    ActionKind.NOOP: frozenset(),
}


@dataclass(frozen=True, slots=True)
class ScenarioAction:
    """One step of a scenario's `actions:` list. Flat rather than one
    subclass per `ActionKind`: the set of kinds is closed at exactly six
    (`_ACTION_PARAM_FIELDS`), so a discriminated dataclass with
    kind-conditional fields -- the same pattern `DialogueTurn` already uses
    for its role-conditional fields -- says everything a class hierarchy
    would, with one fewer indirection for T-09's dispatch code (`match
    action.kind: case ActionKind.SAY: ...`).

    `__post_init__` re-checks that exactly the fields `_ACTION_PARAM_FIELDS`
    lists for `kind` are populated. `_parse_action` below already only ever
    constructs one of these with the right fields, so this can only fire on
    a future bug in this module itself -- kept anyway because catching that
    kind of bug at construction time, loudly, is cheap and exactly the
    posture the rest of this module takes everywhere else.
    """

    kind: ActionKind
    query: str | None = None
    text: str | None = None
    instruction: str | None = None
    use_rag: bool | None = None

    def __post_init__(self) -> None:
        populated = {
            name
            for name, value in (
                ("query", self.query),
                ("text", self.text),
                ("instruction", self.instruction),
                ("use_rag", self.use_rag),
            )
            if value is not None
        }
        expected = _ACTION_PARAM_FIELDS[self.kind]
        if populated != expected:
            raise ValueError(
                f"ScenarioAction(kind={self.kind.value}) ожидает поля {sorted(expected)}, "
                f"получены {sorted(populated)}"
            )


@dataclass(frozen=True, slots=True)
class Scenario:
    """One entry of `scenarios.yaml`'s `scenarios:` mapping, fully parsed
    and validated. `order_index` is this scenario's position in the file
    (0-based, insertion order of the YAML mapping) -- `ScenarioRegistry.
    match()`'s tie-break ("при равенстве — тот, что выше в файле") needs it
    because Python dict/YAML-mapping order is the only record of "higher in
    the file" once the mapping is parsed.
    """

    name: str
    description: str
    entry_states: frozenset[AgentState]
    priority: int
    condition: _Condition
    actions: tuple[ScenarioAction, ...]
    next_state: AgentState
    enabled: bool
    order_index: int


# ---------------------------------------------------------------------------
# YAML loading with line numbers. PyYAML's SafeLoader is used deliberately
# (never yaml.load with the full/unsafe Loader, never eval()) -- the
# closed-vocabulary promise this module exists to keep would be worthless
# if the YAML itself could construct arbitrary Python objects.
# ---------------------------------------------------------------------------


class _LineDict(dict):
    """A `dict` that also remembers the 1-based source line its YAML
    mapping started on, as an *attribute* (`__line__`), not a key --
    subclassing rather than stashing a sentinel key means every existing
    dict operation (`.get`, `.items()`, `in`, equality) behaves exactly as
    it would on a plain dict, no filtering required anywhere else in this
    module.
    """

    __line__: int


class _LineLoader(yaml.SafeLoader):
    """`yaml.SafeLoader` plus two things a hand-edited config file needs
    that plain SafeLoader doesn't give you: line numbers on every mapping
    (so `ScenarioValidationError` can name a line), and a hard failure on
    duplicate keys (PyYAML's default is to silently keep the *last*
    occurrence -- exactly the kind of quietly-swallowed typo this whole
    module exists to prevent, e.g. two scenarios accidentally named
    `answer_question` where the second one silently wins and the first
    never matches anything again).
    """


def _construct_mapping(loader: yaml.SafeLoader, node: yaml.MappingNode, deep: bool = False) -> _LineDict:
    seen: set[object] = set()
    for key_node, _value_node in node.value:
        key = loader.construct_object(key_node, deep=True)
        if key in seen:
            raise yaml.constructor.ConstructorError(
                None,
                None,
                f"повторяющийся ключ {key!r} в mapping'е",
                node.start_mark,
            )
        seen.add(key)
    mapping = _LineDict(yaml.SafeLoader.construct_mapping(loader, node, deep=deep))
    mapping.__line__ = node.start_mark.line + 1
    return mapping


_LineLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping)


def _line_of(node: Any, fallback: int) -> int:
    return node.__line__ if isinstance(node, _LineDict) else fallback


def _fail(path: Path, line: int | str, message: str) -> ScenarioValidationError:
    return ScenarioValidationError(f"{path}:{line}: {message}")


def _load_yaml_document(path: Path) -> Any:
    try:
        raw_text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ScenarioValidationError(
            f"{path}: не удалось прочитать файл сценариев: {exc}"
        ) from exc
    try:
        return yaml.load(raw_text, Loader=_LineLoader)
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        line: int | str = mark.line + 1 if mark is not None else "?"
        raise _fail(path, line, f"битый YAML: {exc}") from exc


# ---------------------------------------------------------------------------
# Leaf condition grammar -- the exact regexes below are the executable form
# of scenarios.yaml's "СЛОВАРЬ УСЛОВИЙ" comment block. Anything that parses
# as a string but matches none of them is an unknown condition and fails
# load, by construction: there is no fallback branch that lets an
# unrecognized string through.
# ---------------------------------------------------------------------------

_INTENT_VALUES = frozenset({"question", "clarification_needed", "goodbye", "smalltalk", "unclear"})
_ERROR_TYPE_VALUES = frozenset({"timeout", "connection_refused", "server_error"})

_NOT_PREFIX_RE = re.compile(r"^not\s+(.+)$")
_INTENT_RE = re.compile(r'^intent\s*==\s*"([a-z_]+)"$')
_CONFIDENCE_RE = re.compile(r"^confidence\s*(<|>=)\s*(\d+(?:\.\d+)?)$")
_RAG_COUNT_RE = re.compile(r"^rag\.count\s*==\s*(\d+)$")
_RAG_MAX_SCORE_RE = re.compile(r"^rag\.max_score\s*<\s*(\d+(?:\.\d+)?)$")
_ERROR_TYPE_RE = re.compile(r'^error\.type\s*==\s*"([a-z_]+)"$')


def _parse_leaf_condition(
    path: Path, scenario_name: str, raw: str, line: int, entry_states: frozenset[AgentState]
) -> _Condition:
    text = raw.strip()

    not_match = _NOT_PREFIX_RE.match(text)
    if not_match:
        inner = _parse_leaf_condition(path, scenario_name, not_match.group(1), line, entry_states)
        return Not(inner)

    if text == "interject_accepted":
        if AgentState.DECIDING_INTERJECT not in entry_states:
            raise _fail(
                path,
                line,
                f"сценарий '{scenario_name}': условие 'interject_accepted' допустимо только "
                f"в entry_states содержащем DecidingInterject",
            )
        return InterjectAccepted()

    if text == "bargein_accepted":
        if AgentState.DECIDING_BARGE_IN not in entry_states:
            raise _fail(
                path,
                line,
                f"сценарий '{scenario_name}': условие 'bargein_accepted' допустимо только "
                f"в entry_states содержащем DecidingBargeIn",
            )
        return BargeinAccepted()

    if text == "transcript.empty":
        return TranscriptEmpty()

    match = _INTENT_RE.match(text)
    if match:
        value = match.group(1)
        if value not in _INTENT_VALUES:
            raise _fail(
                path,
                line,
                f"сценарий '{scenario_name}': неизвестное значение intent '{value}' "
                f"(допустимые: {sorted(_INTENT_VALUES)})",
            )
        return IntentEquals(value)

    match = _CONFIDENCE_RE.match(text)
    if match:
        operator, number = match.group(1), float(match.group(2))
        return ConfidenceLessThan(number) if operator == "<" else ConfidenceAtLeast(number)

    match = _RAG_COUNT_RE.match(text)
    if match:
        return RagCountEquals(int(match.group(1)))

    match = _RAG_MAX_SCORE_RE.match(text)
    if match:
        return RagMaxScoreLessThan(float(match.group(1)))

    match = _ERROR_TYPE_RE.match(text)
    if match:
        value = match.group(1)
        if value not in _ERROR_TYPE_VALUES:
            raise _fail(
                path,
                line,
                f"сценарий '{scenario_name}': неизвестное значение error.type '{value}' "
                f"(допустимые: {sorted(_ERROR_TYPE_VALUES)})",
            )
        return ErrorTypeEquals(value)

    raise _fail(path, line, f"сценарий '{scenario_name}': неизвестное условие '{raw}'")


def _parse_condition(
    path: Path,
    scenario_name: str,
    raw: Any,
    fallback_line: int,
    entry_states: frozenset[AgentState],
) -> _Condition:
    line = _line_of(raw, fallback_line)

    if isinstance(raw, str):
        return _parse_leaf_condition(path, scenario_name, raw, line, entry_states)

    if isinstance(raw, dict):
        keys = set(raw)
        if keys == {"all"}:
            items = raw["all"]
            if not isinstance(items, list) or not items:
                raise _fail(path, line, f"сценарий '{scenario_name}': 'all:' обязан быть непустым списком")
            return AllOf(
                tuple(_parse_condition(path, scenario_name, item, line, entry_states) for item in items)
            )
        if keys == {"any"}:
            items = raw["any"]
            if not isinstance(items, list) or not items:
                raise _fail(path, line, f"сценарий '{scenario_name}': 'any:' обязан быть непустым списком")
            return AnyOf(
                tuple(_parse_condition(path, scenario_name, item, line, entry_states) for item in items)
            )
        if keys == {"not"}:
            return Not(_parse_condition(path, scenario_name, raw["not"], line, entry_states))
        raise _fail(
            path,
            line,
            f"сценарий '{scenario_name}': неизвестная структура условия {sorted(keys)} "
            "(допустимы только all/any/not)",
        )

    raise _fail(
        path,
        line,
        f"сценарий '{scenario_name}': условие должно быть строкой или all:/any:/not:, получено {raw!r}",
    )


# ---------------------------------------------------------------------------
# Placeholder substitutions -- {transcript} {intent.query} {rag.answer}
# {phone}, closed the same way conditions/actions are.
# ---------------------------------------------------------------------------

_PLACEHOLDER_RE = re.compile(r"\{([a-zA-Z_.]+)\}")
_ALLOWED_PLACEHOLDERS = frozenset({"transcript", "intent.query", "rag.answer", "phone"})


def _check_placeholders(
    path: Path, scenario_name: str, field_name: str, text: str, line: int, substitutions: dict[str, str]
) -> None:
    for token in _PLACEHOLDER_RE.findall(text):
        if token not in _ALLOWED_PLACEHOLDERS:
            raise _fail(
                path,
                line,
                f"сценарий '{scenario_name}': неизвестная подстановка '{{{token}}}' в params.{field_name} "
                f"(допустимые: {sorted(_ALLOWED_PLACEHOLDERS)})",
            )
        if token == "phone" and "phone" not in substitutions:
            raise _fail(
                path,
                line,
                f"сценарий '{scenario_name}': params.{field_name} использует подстановку {{phone}}, "
                "но в substitutions.phone нет значения",
            )


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

_ACTION_KIND_VALUES = {kind.value for kind in ActionKind}
_PARAM_TYPES: dict[str, type] = {"query": str, "text": str, "instruction": str, "use_rag": bool}


def _parse_action(
    path: Path, scenario_name: str, raw: Any, fallback_line: int, substitutions: dict[str, str]
) -> ScenarioAction:
    if not isinstance(raw, dict):
        raise _fail(
            path, fallback_line, f"сценарий '{scenario_name}': каждое действие обязано быть mapping'ом"
        )
    line = _line_of(raw, fallback_line)

    unknown_keys = set(raw) - {"action", "params"}
    if unknown_keys:
        raise _fail(
            path,
            line,
            f"сценарий '{scenario_name}': неизвестные ключи в действии: {sorted(unknown_keys)}",
        )

    kind_raw = raw.get("action")
    if kind_raw not in _ACTION_KIND_VALUES:
        raise _fail(
            path,
            line,
            f"сценарий '{scenario_name}': неизвестное действие '{kind_raw}' "
            f"(допустимые: {sorted(_ACTION_KIND_VALUES)})",
        )
    kind = ActionKind(kind_raw)

    params = raw.get("params") or {}
    if not isinstance(params, dict):
        raise _fail(
            path, line, f"сценарий '{scenario_name}': params действия '{kind_raw}' обязан быть mapping'ом"
        )

    expected_fields = _ACTION_PARAM_FIELDS[kind]
    actual_fields = set(params)
    if actual_fields != expected_fields:
        missing = expected_fields - actual_fields
        extra = actual_fields - expected_fields
        details = []
        if missing:
            details.append(f"не хватает {sorted(missing)}")
        if extra:
            details.append(f"лишние {sorted(extra)}")
        raise _fail(
            path,
            line,
            f"сценарий '{scenario_name}': действие '{kind_raw}' — " + ", ".join(details),
        )

    for param_name in expected_fields:
        value = params[param_name]
        expected_type = _PARAM_TYPES[param_name]
        if expected_type is bool:
            if not isinstance(value, bool):
                raise _fail(
                    path,
                    line,
                    f"сценарий '{scenario_name}': params.{param_name} действия '{kind_raw}' "
                    "обязан быть true/false",
                )
        elif not isinstance(value, str):
            raise _fail(
                path,
                line,
                f"сценарий '{scenario_name}': params.{param_name} действия '{kind_raw}' "
                "обязан быть строкой",
            )
        if expected_type is str:
            _check_placeholders(path, scenario_name, param_name, value, line, substitutions)

    if kind is ActionKind.RAG_QUERY:
        return ScenarioAction(kind=kind, query=params["query"])
    if kind is ActionKind.SAY:
        return ScenarioAction(kind=kind, text=params["text"])
    if kind is ActionKind.SAY_GENERATED:
        return ScenarioAction(kind=kind, instruction=params["instruction"], use_rag=params["use_rag"])
    return ScenarioAction(kind=kind)


# ---------------------------------------------------------------------------
# Scenario / document parsing
# ---------------------------------------------------------------------------

_VALID_STATE_NAMES = {state.value for state in AgentState}
_SCENARIO_ALLOWED_KEYS = {"enabled", "description", "entry_states", "priority", "when", "actions", "next_state"}
_TOP_LEVEL_ALLOWED_KEYS = {"version", "last_modified", "scenarios", "substitutions"}


def _parse_state_name(path: Path, scenario_name: str, raw: Any, line: int, field_label: str) -> AgentState:
    if raw not in _VALID_STATE_NAMES:
        raise _fail(
            path,
            line,
            f"сценарий '{scenario_name}': неизвестное состояние {raw!r} в {field_label} "
            f"(допустимые: {sorted(_VALID_STATE_NAMES)})",
        )
    return AgentState(raw)


def _parse_scenario(
    path: Path,
    name: str,
    body: Any,
    order_index: int,
    fallback_line: int,
    substitutions: dict[str, str],
) -> Scenario:
    if not isinstance(body, dict):
        raise _fail(path, fallback_line, f"сценарий '{name}': должен быть mapping'ом")
    line = _line_of(body, fallback_line)

    unknown_keys = set(body) - _SCENARIO_ALLOWED_KEYS
    if unknown_keys:
        raise _fail(path, line, f"сценарий '{name}': неизвестные ключи {sorted(unknown_keys)}")

    enabled = body.get("enabled", True)
    if not isinstance(enabled, bool):
        raise _fail(path, line, f"сценарий '{name}': enabled обязан быть true/false")

    description = body.get("description", "")
    if not isinstance(description, str):
        raise _fail(path, line, f"сценарий '{name}': description обязан быть строкой")

    entry_states_raw = body.get("entry_states")
    if not isinstance(entry_states_raw, list) or not entry_states_raw:
        raise _fail(path, line, f"сценарий '{name}': entry_states обязан быть непустым списком")
    entry_states = frozenset(
        _parse_state_name(path, name, item, line, "entry_states") for item in entry_states_raw
    )

    priority = body.get("priority")
    if not isinstance(priority, int) or isinstance(priority, bool):
        raise _fail(path, line, f"сценарий '{name}': priority обязан быть целым числом")

    when_raw = body.get("when")
    condition = (
        _ALWAYS_TRUE
        if when_raw is None
        else _parse_condition(path, name, when_raw, line, entry_states)
    )

    actions_raw = body.get("actions")
    if not isinstance(actions_raw, list) or not actions_raw:
        raise _fail(path, line, f"сценарий '{name}': actions обязан быть непустым списком")
    actions = tuple(
        _parse_action(path, name, item, line, substitutions) for item in actions_raw
    )

    next_state_raw = body.get("next_state")
    next_state = _parse_state_name(path, name, next_state_raw, line, "next_state")

    return Scenario(
        name=name,
        description=description,
        entry_states=entry_states,
        priority=priority,
        condition=condition,
        actions=actions,
        next_state=next_state,
        enabled=enabled,
        order_index=order_index,
    )


def _parse_substitutions(path: Path, raw: Any, fallback_line: int) -> dict[str, str]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise _fail(path, fallback_line, "substitutions обязан быть mapping'ом строка->строка")
    line = _line_of(raw, fallback_line)
    substitutions: dict[str, str] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise _fail(path, line, f"substitutions.{key!r} обязан отображать строку в строку")
        substitutions[key] = value
    return substitutions


def _parse_document(path: Path, doc: Any) -> tuple[tuple[Scenario, ...], dict[str, str]]:
    if doc is None:
        raise _fail(path, 1, "файл сценариев пуст")
    if not isinstance(doc, dict):
        raise _fail(path, 1, "верхний уровень scenarios.yaml обязан быть mapping'ом")
    top_line = _line_of(doc, 1)

    unknown_keys = set(doc) - _TOP_LEVEL_ALLOWED_KEYS
    if unknown_keys:
        raise _fail(path, top_line, f"неизвестные ключи верхнего уровня: {sorted(unknown_keys)}")

    substitutions = _parse_substitutions(path, doc.get("substitutions"), top_line)

    scenarios_raw = doc.get("scenarios")
    if not isinstance(scenarios_raw, dict) or not scenarios_raw:
        raise _fail(path, top_line, "секция 'scenarios' обязана быть непустым mapping'ом")
    scenarios_line = _line_of(scenarios_raw, top_line)

    scenarios = tuple(
        _parse_scenario(path, name, body, order_index, scenarios_line, substitutions)
        for order_index, (name, body) in enumerate(scenarios_raw.items())
    )
    return scenarios, substitutions


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ScenarioRegistry:
    """Loaded, validated `dialogue/scenarios.yaml`. Construct only via
    `load()` -- the constructor itself does no validation, `load()` is
    where every guarantee this module makes is actually enforced.
    """

    scenarios: tuple[Scenario, ...]
    substitutions: dict[str, str]
    by_name: dict[str, Scenario] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "by_name", {scenario.name: scenario for scenario in self.scenarios})

    @classmethod
    def load(cls, path: Path) -> ScenarioRegistry:
        """Parse and fully validate `path` (`dialogue/scenarios.yaml`).
        Raises `ScenarioValidationError` naming the file and line on any
        problem -- bad YAML syntax, an unknown condition/action/state, a
        duplicate key, a param of the wrong shape, an unresolved `{phone}`
        substitution. Callers (T-09/app.py) are expected to call this once
        at process startup and let the exception propagate: a scenario
        that can never match because of a typo is a production incident
        waiting to happen, not something to discover mid-session.
        """
        doc = _load_yaml_document(path)
        scenarios, substitutions = _parse_document(path, doc)
        return cls(scenarios=scenarios, substitutions=substitutions)

    def match(self, state: AgentState, context: ScenarioContext) -> Scenario | None:
        """The single scenario that applies to `state` under `context`, or
        `None` if none do. Among scenarios whose `entry_states` contains
        `state`, whose `enabled` is true, and whose `when` evaluates true
        against `context`, the winner has the largest `priority`; ties
        break by file order (`order_index`, ascending -- "выше в файле").

        Every scenario sharing `state` is evaluated, not just the eventual
        winner -- a field a given scenario's `when` needs but `context`
        never populated (module docstring's `stt_no_speech` /
        `user_says_goodbye` example) simply makes that one scenario's
        condition false; it never raises.
        """
        candidates = [
            scenario
            for scenario in self.scenarios
            if scenario.enabled and state in scenario.entry_states and scenario.condition.evaluate(context)
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda scenario: (-scenario.priority, scenario.order_index))

    def render(self, template: str, context: ScenarioContext) -> str:
        """Applies the closed substitution set (`{transcript}`
        `{intent.query}` `{rag.answer}` `{phone}`) to `template`. Plain
        `str.replace()` calls, not `str.format(**mapping)` -- the set of
        substitutions is closed and validated at load time
        (`_check_placeholders`), so there is nothing here that needs
        `format()`'s ability to raise on an unexpected key or silently
        accept an unvalidated one.
        """
        return (
            template.replace("{transcript}", context.transcript)
            .replace("{intent.query}", context.intent_query)
            .replace("{rag.answer}", context.rag_answer)
            .replace("{phone}", self.substitutions.get("phone", ""))
        )
