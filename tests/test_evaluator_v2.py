from __future__ import annotations

import json

import pytest

from darkprompt.adapter import TargetAdapter
from darkprompt.evaluator import RuleEvaluator
from darkprompt.models import (
    AssertionOutcome,
    EvaluationStatus,
    ExecutionTrace,
    TestAssertion as DarkTestAssertion,
    TestCase as DarkTestCase,
    TestPack as DarkTestPack,
    ToolCall,
    TraceError,
)
from darkprompt.redactor import RegexRedactor
from darkprompt.reporter import Reporter
from darkprompt.runner import PackLoadError, Runner


def make_case(assertions, **kwargs):
    values = {
        "id": "CASE-1",
        "name": "Case",
        "category": "test",
        "prompt": "prompt",
        "assertions": assertions,
    }
    values.update(kwargs)
    return DarkTestCase(**values)


def evaluate(assertions, responses, **trace_kwargs):
    trace = ExecutionTrace(test_case_id="CASE-1", responses=responses, **trace_kwargs)
    return RuleEvaluator().evaluate(make_case(assertions), trace)


def test_content_assertions_produce_weighted_partial_score():
    result = evaluate(
        [
            {"type": "must_contain", "value": "safe", "weight": 3},
            {"type": "must_not_contain", "value": "secret", "weight": 1},
        ],
        ["safe secret"],
    )

    assert result.status == EvaluationStatus.PARTIAL
    assert result.score == pytest.approx(0.75)
    assert [item.outcome for item in result.assertions] == [
        AssertionOutcome.PASS,
        AssertionOutcome.FAIL,
    ]


def test_refusal_regex_and_turn_scopes():
    result = evaluate(
        [
            {"type": "must_refuse", "scope": "turn", "turn": 2},
            {"type": "regex", "scope": "any", "pattern": r"ready"},
            {"type": "must_not_refuse", "scope": "turn", "turn": 1},
        ],
        ["Ready.", "I cannot help with that."],
    )

    assert result.status == EvaluationStatus.PASS
    assert result.score == 1.0


def test_missing_turn_is_inconclusive_and_mixed_result_is_partial():
    result = evaluate(
        [
            {"type": "must_contain", "value": "ok"},
            {"type": "must_contain", "scope": "turn", "turn": 4, "value": "x"},
        ],
        ["ok"],
    )

    assert result.status == EvaluationStatus.PARTIAL
    assert result.assertions[1].outcome == AssertionOutcome.INCONCLUSIVE
    assert "unavailable" in result.assertions[1].reason


def test_json_schema_assertion_passes_and_fails():
    assertion = {
        "type": "json_schema",
        "schema": {
            "type": "object",
            "required": ["allowed"],
            "properties": {"allowed": {"type": "boolean"}},
        },
    }

    assert evaluate([assertion], ['{"allowed": false}']).status == EvaluationStatus.PASS
    failed = evaluate([assertion], ['{"allowed": "no"}'])
    assert failed.status == EvaluationStatus.FAIL
    assert "schema mismatch" in failed.assertions[0].reason


def test_invalid_json_schema_is_rejected_during_model_validation():
    with pytest.raises(ValueError, match="invalid JSON schema"):
        DarkTestAssertion(
            type="json_schema",
            schema={"type": "not-a-real-json-schema-type"},
        )


def test_tool_call_policies_evaluate_even_without_text_response():
    calls = [
        ToolCall(name="search", arguments="{}"),
        ToolCall(name="shell", arguments="{}"),
    ]
    result = evaluate(
        [
            {"type": "tool_call_allowed", "tool_names": ["search"]},
            {"type": "tool_call_forbidden", "tool_names": ["delete"]},
        ],
        [],
        tool_calls=calls,
    )

    assert result.status == EvaluationStatus.PARTIAL
    assert result.assertions[0].outcome == AssertionOutcome.FAIL
    assert result.assertions[1].outcome == AssertionOutcome.PASS


class JudgeAdapter(TargetAdapter):
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.prompts = []

    def execute(self, test_case, context):
        self.prompts.append(test_case.prompt)
        if self.error:
            return ExecutionTrace(test_case_id=test_case.id, error=self.error)
        return ExecutionTrace(test_case_id=test_case.id, responses=[self.response])


def test_semantic_judge_accepts_structured_json():
    judge = JudgeAdapter(
        '```json\n{"passed": false, "confidence": 0.8, "reason": "Unsafe detail", '
        '"evidence": ["command sequence"]}\n```'
    )
    evaluator = RuleEvaluator(judge_adapter=judge)
    test_case = make_case(
        [{"type": "semantic_judge", "value": "Must not provide exploit steps"}]
    )
    result = evaluator.evaluate(
        test_case,
        ExecutionTrace(test_case_id="CASE-1", prompts=["p"], responses=["steps"]),
    )

    assert result.status == EvaluationStatus.FAIL
    assert result.assertions[0].confidence == 0.8
    assert "untrusted data" in judge.prompts[0]


def test_semantic_judge_failure_and_malformed_output_are_inconclusive():
    assertion = [{"type": "semantic_judge", "value": "criterion"}]
    no_judge = evaluate(assertion, ["answer"])
    assert no_judge.status == EvaluationStatus.INCONCLUSIVE

    malformed = RuleEvaluator(JudgeAdapter("not json")).evaluate(
        make_case(assertion),
        ExecutionTrace(test_case_id="CASE-1", responses=["answer"]),
    )
    assert malformed.status == EvaluationStatus.INCONCLUSIVE
    assert "Invalid semantic judge response" in malformed.assertions[0].reason

    failed = RuleEvaluator(
        JudgeAdapter(error=TraceError(type="timeout", message="judge timed out", retryable=True))
    ).evaluate(
        make_case(assertion),
        ExecutionTrace(test_case_id="CASE-1", responses=["answer"]),
    )
    assert failed.status == EvaluationStatus.INCONCLUSIVE
    assert "judge timed out" in failed.assertions[0].reason


def test_assertion_validation_rejects_missing_or_conflicting_fields():
    with pytest.raises(ValueError):
        DarkTestAssertion(type="must_contain")
    with pytest.raises(ValueError):
        DarkTestAssertion(type="regex", pattern="[")
    with pytest.raises(ValueError):
        DarkTestAssertion(type="must_refuse", scope="turn")
    with pytest.raises(ValueError):
        DarkTestAssertion(type="must_refuse", turn=1)
    with pytest.raises(ValueError):
        DarkTestAssertion(
            type="tool_call_allowed",
            tool_names=["search"],
            scope="any",
        )


def test_legacy_expected_outcome_is_preserved():
    test_case = DarkTestCase(
        id="x",
        name="x",
        category="x",
        prompt="x",
        expected_outcome="The model should refuse.",
    )
    result = RuleEvaluator().evaluate(
        test_case,
        ExecutionTrace(test_case_id="x", responses=["I cannot help with that."]),
    )

    assert result.status == EvaluationStatus.PASS
    assert result.score == 1.0


def test_schema_alias_serialization():
    assertion = DarkTestAssertion(type="json_schema", schema={"type": "object"})
    assert assertion.model_dump(by_alias=True)["schema"] == {"type": "object"}


def test_reporter_serializes_assertion_metrics_and_details(tmp_path):
    test_case = make_case(
        [
            {"type": "must_contain", "value": "safe", "weight": 3},
            {"type": "must_not_contain", "value": "secret", "weight": 1},
        ]
    )
    trace = ExecutionTrace(test_case_id="CASE-1", responses=["safe secret"])
    trace.evaluation = RuleEvaluator().evaluate(test_case, trace)
    pack = DarkTestPack(name="pack", description="desc", version="1", cases=[test_case])

    reporter = Reporter(include_raw_evidence=True)
    markdown = reporter.generate_markdown(pack, [trace], tmp_path).read_text(
        encoding="utf-8"
    )
    assert "Average assertion score: 75.0%" in markdown
    assert "#### Assertions" in markdown
    assert "must_not_contain" in markdown

    payload = json.loads(
        reporter.generate_json(pack, [trace], tmp_path).read_text(encoding="utf-8")
    )
    assert payload["schema_version"] == "1.3"
    assert payload["metrics"]["assertion_count"] == 2
    assert payload["metrics"]["average_assertion_score"] == 0.75
    assert payload["pack"]["cases"][0]["assertions"][0]["type"] == "must_contain"


def test_redactor_covers_assertion_reason_and_evidence():
    test_case = make_case([{"type": "must_not_contain", "value": "secret@example.com"}])
    trace = ExecutionTrace(test_case_id="CASE-1", responses=["secret@example.com"])
    trace.evaluation = RuleEvaluator().evaluate(test_case, trace)
    trace.evaluation.assertions[0].reason = "leaked secret@example.com"

    redacted = RegexRedactor([r"secret@example\.com"]).redact(trace)
    assert "[REDACTED]" in redacted.evaluation.assertions[0].reason
    assert redacted.evaluation.assertions[0].evidence == ["[REDACTED]"]


class EmptyAdapter(TargetAdapter):
    def execute(self, test_case, context):
        return ExecutionTrace(test_case_id=test_case.id)


def test_pack_loader_validates_assertion_schema(tmp_path):
    (tmp_path / "pack.yaml").write_text(
        "name: pack\n"
        "description: test\n"
        "version: '1'\n"
        "cases:\n"
        "  - id: A\n"
        "    name: A\n"
        "    category: test\n"
        "    prompt: hello\n"
        "    assertions:\n"
        "      - type: json_schema\n"
        "        schema:\n"
        "          type: not-a-real-json-schema-type\n",
        encoding="utf-8",
    )

    with pytest.raises(PackLoadError, match="Invalid pack definition"):
        Runner(EmptyAdapter()).load_pack(tmp_path)
