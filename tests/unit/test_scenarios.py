"""Unit tests for `backend.dialogue.scenarios.ScenarioRegistry` (T-08).

The load-bearing cases, each tied to a specific contract clause:

* `test_broken_yaml_syntax_fails_and_names_the_line` / `test_unknown_*` --
  spec.md FR-24 / plan.md §7: the loader must fail at startup, not at
  runtime, whenever `dialogue/scenarios.yaml` references anything outside
  the closed vocabulary (condition, action, state), and the exception must
  point at the offending line so the fix doesn't require bisecting the
  file by hand.
* `test_priority_resolves_the_higher_number` /
  `test_priority_tie_breaks_by_file_order` -- scenarios.yaml's own rule:
  "среди подходящих побеждает наибольший priority; при равенстве — тот,
  что выше в файле".
* `test_nested_all_any_not_*` -- the vocabulary comment's "Условия
  соединяются через `all:` (И) и `any:` (ИЛИ). Вложенность допустима",
  plus the bare `not <condition>` string form actually used in the file
  (`interject_keep_listening`, `bargein_continue`).
* `test_render_*` -- the four closed substitutions
  (`{transcript}` `{intent.query}` `{rag.answer}` `{phone}`).
* `test_real_scenarios_file_*` -- every one of the 12 scenarios shipped in
  `dialogue/scenarios.yaml` matches in the situation its own
  `description:` says it should, using the real, unmodified file (not a
  synthetic fixture) so a change to the real file that breaks a scenario's
  applicability is caught here, not in production.
* `test_match_does_not_raise_when_sibling_scenario_needs_unset_field` --
  the bug this module's design went through one iteration to avoid:
  `ScenarioRegistry.match()` evaluates every scenario sharing the queried
  `entry_states`, not just the eventual winner, so an empty transcript
  (no `IntentDecision` ever run, `intent`/`confidence` still `None`) must
  not crash evaluating `user_says_goodbye`'s `intent == "goodbye"` just
  because `stt_no_speech` is also a Listening-state candidate.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from backend.dialogue.models import AgentState
from backend.dialogue.scenarios import (
    ActionKind,
    ScenarioAction,
    ScenarioContext,
    ScenarioRegistry,
    ScenarioValidationError,
)

_REAL_SCENARIOS_PATH = Path(__file__).resolve().parents[2] / "dialogue" / "scenarios.yaml"


def _write(tmp_path: Path, text: str, name: str = "scenarios.yaml") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


_MINIMAL_VALID = """
scenarios:
  greet_back:
    entry_states: [Listening]
    priority: 10
    actions:
      - action: noop
    next_state: Listening
"""


# ---------------------------------------------------------------------------
# Load-time failures: bad YAML, unknown condition/action/state.
# ---------------------------------------------------------------------------


def test_broken_yaml_syntax_fails_and_names_the_line(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
scenarios:
  bad:
    entry_states: [Listening]
    priority: 10
    actions:
      - action: say
        params:
          text: "unterminated
    next_state: Listening
""",
    )
    with pytest.raises(ScenarioValidationError) as excinfo:
        ScenarioRegistry.load(path)
    message = str(excinfo.value)
    assert str(path) in message
    assert any(ch.isdigit() for ch in message)  # a line number is present


def test_unknown_condition_fails_and_names_the_scenario_and_line(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
scenarios:
  typo_condition:
    entry_states: [Listening]
    priority: 10
    when:
      all:
        - intnet == "question"
    actions:
      - action: noop
    next_state: Listening
""",
    )
    with pytest.raises(ScenarioValidationError) as excinfo:
        ScenarioRegistry.load(path)
    message = str(excinfo.value)
    assert "typo_condition" in message
    assert "intnet" in message
    assert f"{path}:" in message


@pytest.mark.parametrize(
    "condition",
    [
        'intent == "not_a_real_intent"',
        'error.type == "not_a_real_error"',
    ],
)
def test_unknown_enum_value_in_condition_fails_load(tmp_path: Path, condition: str) -> None:
    path = _write(
        tmp_path,
        f"""
scenarios:
  bad_enum:
    entry_states: [Listening]
    priority: 10
    when:
      all:
        - {condition}
    actions:
      - action: noop
    next_state: Listening
""",
    )
    with pytest.raises(ScenarioValidationError):
        ScenarioRegistry.load(path)


def test_unknown_action_fails_load(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
scenarios:
  bad_action:
    entry_states: [Listening]
    priority: 10
    actions:
      - action: sing_a_song
    next_state: Listening
""",
    )
    with pytest.raises(ScenarioValidationError) as excinfo:
        ScenarioRegistry.load(path)
    assert "sing_a_song" in str(excinfo.value)


def test_unknown_entry_state_fails_load(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
scenarios:
  bad_state:
    entry_states: [Listenning]
    priority: 10
    actions:
      - action: noop
    next_state: Listening
""",
    )
    with pytest.raises(ScenarioValidationError) as excinfo:
        ScenarioRegistry.load(path)
    assert "Listenning" in str(excinfo.value)


def test_unknown_next_state_fails_load(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
scenarios:
  bad_next_state:
    entry_states: [Listening]
    priority: 10
    actions:
      - action: noop
    next_state: Closingg
""",
    )
    with pytest.raises(ScenarioValidationError) as excinfo:
        ScenarioRegistry.load(path)
    assert "Closingg" in str(excinfo.value)


def test_action_missing_required_param_fails_load(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
scenarios:
  missing_param:
    entry_states: [Listening]
    priority: 10
    actions:
      - action: say
    next_state: Listening
""",
    )
    with pytest.raises(ScenarioValidationError, match="say"):
        ScenarioRegistry.load(path)


def test_action_with_extra_param_fails_load(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
scenarios:
  extra_param:
    entry_states: [Listening]
    priority: 10
    actions:
      - action: noop
        params:
          unexpected: 1
    next_state: Listening
""",
    )
    with pytest.raises(ScenarioValidationError):
        ScenarioRegistry.load(path)


def test_action_param_wrong_type_fails_load(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
scenarios:
  wrong_type:
    entry_states: [Listening]
    priority: 10
    actions:
      - action: say_generated
        params:
          instruction: "do it"
          use_rag: "yes"
    next_state: Listening
""",
    )
    with pytest.raises(ScenarioValidationError, match="use_rag"):
        ScenarioRegistry.load(path)


def test_duplicate_scenario_name_fails_load(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
scenarios:
  dup:
    entry_states: [Listening]
    priority: 10
    actions:
      - action: noop
    next_state: Listening
  dup:
    entry_states: [Closing]
    priority: 20
    actions:
      - action: noop
    next_state: Listening
""",
    )
    with pytest.raises(ScenarioValidationError):
        ScenarioRegistry.load(path)


def test_unknown_top_level_key_fails_load(tmp_path: Path) -> None:
    path = _write(tmp_path, _MINIMAL_VALID + "\nbogus_section: 1\n")
    with pytest.raises(ScenarioValidationError, match="bogus_section"):
        ScenarioRegistry.load(path)


def test_unknown_scenario_key_fails_load(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
scenarios:
  bad_key:
    entry_states: [Listening]
    priority: 10
    bogus_field: true
    actions:
      - action: noop
    next_state: Listening
""",
    )
    with pytest.raises(ScenarioValidationError, match="bogus_field"):
        ScenarioRegistry.load(path)


def test_empty_file_fails_load(tmp_path: Path) -> None:
    path = _write(tmp_path, "")
    with pytest.raises(ScenarioValidationError):
        ScenarioRegistry.load(path)


def test_missing_file_fails_load(tmp_path: Path) -> None:
    with pytest.raises(ScenarioValidationError):
        ScenarioRegistry.load(tmp_path / "does_not_exist.yaml")


def test_empty_scenarios_section_fails_load(tmp_path: Path) -> None:
    path = _write(tmp_path, "scenarios: {}\n")
    with pytest.raises(ScenarioValidationError):
        ScenarioRegistry.load(path)


def test_unknown_placeholder_fails_load(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
scenarios:
  bad_placeholder:
    entry_states: [Listening]
    priority: 10
    actions:
      - action: say
        params:
          text: "hello {nonexistent}"
    next_state: Listening
""",
    )
    with pytest.raises(ScenarioValidationError, match="nonexistent"):
        ScenarioRegistry.load(path)


def test_phone_placeholder_without_substitution_fails_load(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
scenarios:
  needs_phone:
    entry_states: [Listening]
    priority: 10
    actions:
      - action: say
        params:
          text: "call {phone}"
    next_state: Listening
""",
    )
    with pytest.raises(ScenarioValidationError, match="phone"):
        ScenarioRegistry.load(path)


def test_interject_accepted_outside_deciding_interject_fails_load(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
scenarios:
  scoped_wrong:
    entry_states: [Listening]
    priority: 10
    when:
      all:
        - interject_accepted
    actions:
      - action: noop
    next_state: Listening
""",
    )
    with pytest.raises(ScenarioValidationError, match="DecidingInterject"):
        ScenarioRegistry.load(path)


def test_bargein_accepted_outside_deciding_bargein_fails_load(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
scenarios:
  scoped_wrong:
    entry_states: [Listening]
    priority: 10
    when:
      all:
        - bargein_accepted
    actions:
      - action: noop
    next_state: Listening
""",
    )
    with pytest.raises(ScenarioValidationError, match="DecidingBargeIn"):
        ScenarioRegistry.load(path)


# ---------------------------------------------------------------------------
# Priority resolution
# ---------------------------------------------------------------------------


def test_priority_resolves_the_higher_number(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
scenarios:
  low:
    entry_states: [Listening]
    priority: 10
    actions:
      - action: noop
    next_state: Listening
  high:
    entry_states: [Listening]
    priority: 90
    actions:
      - action: noop
    next_state: Closing
""",
    )
    registry = ScenarioRegistry.load(path)
    winner = registry.match(AgentState.LISTENING, ScenarioContext())
    assert winner is not None
    assert winner.name == "high"


def test_priority_tie_breaks_by_file_order(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
scenarios:
  first_in_file:
    entry_states: [Listening]
    priority: 50
    actions:
      - action: noop
    next_state: Listening
  second_in_file:
    entry_states: [Listening]
    priority: 50
    actions:
      - action: noop
    next_state: Closing
""",
    )
    registry = ScenarioRegistry.load(path)
    winner = registry.match(AgentState.LISTENING, ScenarioContext())
    assert winner is not None
    assert winner.name == "first_in_file"


def test_match_returns_none_when_nothing_applies(tmp_path: Path) -> None:
    path = _write(tmp_path, _MINIMAL_VALID)
    registry = ScenarioRegistry.load(path)
    assert registry.match(AgentState.SPEAKING, ScenarioContext()) is None


def test_disabled_scenario_never_matches(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
scenarios:
  off:
    enabled: false
    entry_states: [Listening]
    priority: 100
    actions:
      - action: noop
    next_state: Listening
""",
    )
    registry = ScenarioRegistry.load(path)
    assert registry.match(AgentState.LISTENING, ScenarioContext()) is None


# ---------------------------------------------------------------------------
# Nested all/any/not
# ---------------------------------------------------------------------------


def _nested_registry(tmp_path: Path) -> ScenarioRegistry:
    path = _write(
        tmp_path,
        """
scenarios:
  nested:
    entry_states: [Listening]
    priority: 10
    when:
      all:
        - any:
            - intent == "question"
            - intent == "smalltalk"
        - not:
            confidence < 0.5
    actions:
      - action: noop
    next_state: Listening
""",
    )
    return ScenarioRegistry.load(path)


def test_nested_all_any_not_matches_when_satisfied(tmp_path: Path) -> None:
    registry = _nested_registry(tmp_path)
    context = ScenarioContext(intent="question", confidence=0.8)
    assert registry.match(AgentState.LISTENING, context) is not None


def test_nested_all_any_not_rejects_when_any_branch_fails(tmp_path: Path) -> None:
    registry = _nested_registry(tmp_path)
    # intent matches neither "question" nor "smalltalk" -> outer any: is false.
    context = ScenarioContext(intent="goodbye", confidence=0.8)
    assert registry.match(AgentState.LISTENING, context) is None


def test_nested_all_any_not_rejects_when_not_branch_fails(tmp_path: Path) -> None:
    registry = _nested_registry(tmp_path)
    # confidence < 0.5 is true, so not: makes the whole all: false.
    context = ScenarioContext(intent="question", confidence=0.2)
    assert registry.match(AgentState.LISTENING, context) is None


def test_bare_not_prefix_string_form(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
scenarios:
  bare_not:
    entry_states: [DecidingInterject]
    priority: 10
    when:
      all:
        - not interject_accepted
    actions:
      - action: noop
    next_state: Listening
""",
    )
    registry = ScenarioRegistry.load(path)
    assert registry.match(AgentState.DECIDING_INTERJECT, ScenarioContext(interject_accepted=False)) is not None
    assert registry.match(AgentState.DECIDING_INTERJECT, ScenarioContext(interject_accepted=True)) is None


# ---------------------------------------------------------------------------
# Substitutions
# ---------------------------------------------------------------------------


def test_render_replaces_all_four_placeholders(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        _MINIMAL_VALID + '\nsubstitutions:\n  phone: "+7 900 000-00-00"\n',
    )
    registry = ScenarioRegistry.load(path)
    context = ScenarioContext(transcript="привет", intent_query="когда экзамен", rag_answer="15 июля")
    rendered = registry.render(
        "Вы сказали: {transcript}. Запрос: {intent.query}. Ответ: {rag.answer}. Тел: {phone}", context
    )
    assert rendered == (
        "Вы сказали: привет. Запрос: когда экзамен. Ответ: 15 июля. Тел: +7 900 000-00-00"
    )


def test_render_leaves_text_without_placeholders_untouched(tmp_path: Path) -> None:
    path = _write(tmp_path, _MINIMAL_VALID)
    registry = ScenarioRegistry.load(path)
    assert registry.render("Секунду, посмотрю.", ScenarioContext()) == "Секунду, посмотрю."


def test_render_missing_phone_substitution_falls_back_to_empty(tmp_path: Path) -> None:
    # Loader already refuses a scenario that USES {phone} without a
    # substitutions.phone entry (test_phone_placeholder_without_substitution_
    # fails_load); this only exercises render() in isolation with a template
    # not sourced from any scenario's own (validated) text.
    path = _write(tmp_path, _MINIMAL_VALID)
    registry = ScenarioRegistry.load(path)
    assert registry.render("Тел: {phone}", ScenarioContext()) == "Тел: "


# ---------------------------------------------------------------------------
# ScenarioAction shape
# ---------------------------------------------------------------------------


def test_scenario_action_rejects_mismatched_fields() -> None:
    with pytest.raises(ValueError):
        ScenarioAction(kind=ActionKind.NOOP, text="should not be here")


def test_scenario_action_accepts_matching_fields() -> None:
    action = ScenarioAction(kind=ActionKind.SAY, text="hi")
    assert action.text == "hi"
    assert action.query is None


# ---------------------------------------------------------------------------
# The real dialogue/scenarios.yaml -- loads cleanly, and each of its 12
# scenarios matches in the situation its own description promises.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_registry() -> ScenarioRegistry:
    return ScenarioRegistry.load(_REAL_SCENARIOS_PATH)


def test_real_scenarios_file_loads_without_error(real_registry: ScenarioRegistry) -> None:
    assert len(real_registry.scenarios) == 12
    assert real_registry.substitutions["phone"] == "+7 (XXX) XXX-XX-XX"


def test_match_does_not_raise_when_sibling_scenario_needs_unset_field(real_registry: ScenarioRegistry) -> None:
    # transcript empty -> no IntentDecision ran -> intent/confidence are
    # None, yet user_says_goodbye ("intent == \"goodbye\"") shares
    # entry_states: [Listening] with stt_no_speech and must not blow up.
    winner = real_registry.match(AgentState.LISTENING, ScenarioContext(transcript=""))
    assert winner is not None
    assert winner.name == "stt_no_speech"


def test_user_says_goodbye_matches_on_goodbye_intent(real_registry: ScenarioRegistry) -> None:
    context = ScenarioContext(intent="goodbye", confidence=0.9, transcript="спасибо, до свидания")
    winner = real_registry.match(AgentState.LISTENING, context)
    assert winner is not None
    assert winner.name == "user_says_goodbye"
    assert winner.next_state == AgentState.CLOSING


def test_user_says_goodbye_does_not_match_below_confidence_threshold(real_registry: ScenarioRegistry) -> None:
    context = ScenarioContext(intent="goodbye", confidence=0.5, transcript="ну ладно наверное всё")
    winner = real_registry.match(AgentState.LISTENING, context)
    assert winner is None or winner.name != "user_says_goodbye"


def test_stt_no_speech_matches_empty_transcript(real_registry: ScenarioRegistry) -> None:
    winner = real_registry.match(AgentState.LISTENING, ScenarioContext(transcript="   "))
    assert winner is not None
    assert winner.name == "stt_no_speech"
    assert winner.next_state == AgentState.LISTENING


def test_clarify_unclear_matches_low_confidence(real_registry: ScenarioRegistry) -> None:
    context = ScenarioContext(intent="question", confidence=0.3, transcript="а это самое, короче")
    winner = real_registry.match(AgentState.LISTENING, context)
    assert winner is not None
    assert winner.name == "clarify_unclear"


def test_clarify_unclear_matches_unclear_intent_regardless_of_confidence(real_registry: ScenarioRegistry) -> None:
    context = ScenarioContext(intent="unclear", confidence=0.95, transcript="бла бла бла")
    winner = real_registry.match(AgentState.LISTENING, context)
    assert winner is not None
    assert winner.name == "clarify_unclear"


def test_clarify_unclear_matches_clarification_needed_intent(real_registry: ScenarioRegistry) -> None:
    context = ScenarioContext(intent="clarification_needed", confidence=0.95, transcript="а сколько это стоит")
    winner = real_registry.match(AgentState.LISTENING, context)
    assert winner is not None
    assert winner.name == "clarify_unclear"


def test_answer_question_matches_confident_question(real_registry: ScenarioRegistry) -> None:
    context = ScenarioContext(
        intent="question", confidence=0.9, intent_query="сроки подачи документов", transcript="когда подавать документы"
    )
    winner = real_registry.match(AgentState.LISTENING, context)
    assert winner is not None
    assert winner.name == "answer_question"
    assert winner.next_state == AgentState.SPEAKING
    assert [action.kind for action in winner.actions] == [
        ActionKind.SAY,
        ActionKind.RAG_QUERY,
        ActionKind.SAY_GENERATED,
    ]


def test_smalltalk_matches_smalltalk_intent(real_registry: ScenarioRegistry) -> None:
    context = ScenarioContext(intent="smalltalk", confidence=0.9, transcript="как погода у вас там")
    winner = real_registry.match(AgentState.LISTENING, context)
    assert winner is not None
    assert winner.name == "smalltalk"


def test_no_kb_match_matches_empty_rag_results(real_registry: ScenarioRegistry) -> None:
    context = ScenarioContext(rag_count=0, rag_max_score=0.0)
    winner = real_registry.match(AgentState.FORMULATING, context)
    assert winner is not None
    assert winner.name == "no_kb_match"


def test_no_kb_match_matches_low_rag_score(real_registry: ScenarioRegistry) -> None:
    context = ScenarioContext(rag_count=3, rag_max_score=0.1)
    winner = real_registry.match(AgentState.FORMULATING, context)
    assert winner is not None
    assert winner.name == "no_kb_match"


def test_no_kb_match_does_not_fire_on_good_rag_results(real_registry: ScenarioRegistry) -> None:
    context = ScenarioContext(rag_count=5, rag_max_score=0.8)
    winner = real_registry.match(AgentState.FORMULATING, context)
    assert winner is None


def test_interject_summarize_matches_accepted_decision(real_registry: ScenarioRegistry) -> None:
    winner = real_registry.match(AgentState.DECIDING_INTERJECT, ScenarioContext(interject_accepted=True))
    assert winner is not None
    assert winner.name == "interject_summarize"
    assert winner.next_state == AgentState.FORMULATING


def test_interject_keep_listening_matches_rejected_decision(real_registry: ScenarioRegistry) -> None:
    winner = real_registry.match(AgentState.DECIDING_INTERJECT, ScenarioContext(interject_accepted=False))
    assert winner is not None
    assert winner.name == "interject_keep_listening"
    assert winner.next_state == AgentState.LISTENING


def test_bargein_yield_matches_accepted_decision(real_registry: ScenarioRegistry) -> None:
    winner = real_registry.match(AgentState.DECIDING_BARGE_IN, ScenarioContext(bargein_accepted=True))
    assert winner is not None
    assert winner.name == "bargein_yield"
    assert winner.next_state == AgentState.LISTENING
    assert winner.actions[0].kind == ActionKind.COMMIT_PARTIAL


def test_bargein_continue_matches_rejected_decision(real_registry: ScenarioRegistry) -> None:
    winner = real_registry.match(AgentState.DECIDING_BARGE_IN, ScenarioContext(bargein_accepted=False))
    assert winner is not None
    assert winner.name == "bargein_continue"
    assert winner.next_state == AgentState.SPEAKING
    assert winner.actions[0].kind == ActionKind.RESUME_SPEAKING


def test_late_question_matches_unconditionally_in_closing(real_registry: ScenarioRegistry) -> None:
    winner = real_registry.match(AgentState.CLOSING, ScenarioContext())
    assert winner is not None
    assert winner.name == "late_question"
    assert winner.next_state == AgentState.LISTENING


@pytest.mark.parametrize("state", [AgentState.LISTENING, AgentState.FORMULATING, AgentState.SPEAKING])
@pytest.mark.parametrize("error_type", ["timeout", "connection_refused", "server_error"])
def test_service_unavailable_matches_any_error_type_in_any_eligible_state(
    real_registry: ScenarioRegistry, state: AgentState, error_type: str
) -> None:
    winner = real_registry.match(state, ScenarioContext(error_type=error_type))
    assert winner is not None
    assert winner.name == "service_unavailable"
    assert winner.next_state == AgentState.LISTENING


def test_service_unavailable_outranks_everything_else_when_present(real_registry: ScenarioRegistry) -> None:
    # Even a confident goodbye must lose to a live error -- priority 100
    # beats user_says_goodbye's 90.
    context = ScenarioContext(intent="goodbye", confidence=0.99, transcript="до свидания", error_type="timeout")
    winner = real_registry.match(AgentState.LISTENING, context)
    assert winner is not None
    assert winner.name == "service_unavailable"


def test_every_real_scenario_is_reachable_by_at_least_one_context(real_registry: ScenarioRegistry) -> None:
    # Sanity net: every scenario name asserted above must actually exist in
    # the file, so a rename in scenarios.yaml fails this suite loudly
    # instead of the per-scenario tests above quietly asserting `is None`
    # and appearing to pass for the wrong reason.
    expected_names = {
        "user_says_goodbye",
        "stt_no_speech",
        "clarify_unclear",
        "answer_question",
        "smalltalk",
        "no_kb_match",
        "interject_summarize",
        "interject_keep_listening",
        "bargein_yield",
        "bargein_continue",
        "late_question",
        "service_unavailable",
    }
    assert set(real_registry.by_name) == expected_names


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
