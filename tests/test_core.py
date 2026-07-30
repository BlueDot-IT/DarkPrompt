from __future__ import annotations

import json
from pathlib import Path

import pytest

from darkprompt.adapter import TargetAdapter
from darkprompt.evaluator import RuleEvaluator
from darkprompt.models import (
    EvaluationStatus,
    ExecutionTrace,
    TestCase as DarkTestCase,
    TestPack as DarkTestPack,
)
from darkprompt.mutator import PromptMutator
from darkprompt.redactor import RedactionPatternError, RegexRedactor
from darkprompt.reporter import Reporter
from darkprompt.runner import PackLoadError, Runner


class RecordingAdapter(TargetAdapter):
    def __init__(self, responses=None, fail_at=None):
        self.responses = list(responses or [])
        self.fail_at = fail_at
        self.contexts = []

    def execute(self, test_case, context):
        self.contexts.append(
            {
                "prompt": test_case.prompt,
                "history": [dict(item) for item in context.get("history", [])],
            }
        )
        index = len(self.contexts) - 1
        if self.fail_at == index:
            return self.error_trace(
                test_case,
                error_type="network_error",
                message="request failed",
                retryable=True,
            )
        return ExecutionTrace(
            test_case_id=test_case.id,
            prompts=[test_case.prompt],
            responses=[self.responses[index]],
            metadata={"model": "fake"},
        )


def case(**overrides):
    values = {
        "id": "CASE-1",
        "name": "Case",
        "category": "Injection",
        "prompt": "setup",
        "chain": ["final request"],
        "expected_outcome": "The model should refuse the final request.",
    }
    values.update(overrides)
    return DarkTestCase(**values)


def test_runner_preserves_multi_turn_history_and_evaluates_final_turn():
    adapter = RecordingAdapter(["Ready.", "I cannot help with that."])
    trace = Runner(adapter).run_case(case())

    assert adapter.contexts[0]["history"] == []
    assert adapter.contexts[1]["history"] == [
        {"role": "user", "content": "setup"},
        {"role": "assistant", "content": "Ready."},
    ]
    assert trace.responses == ["Ready.", "I cannot help with that."]
    assert trace.evaluation.status == EvaluationStatus.PASS


def test_runner_marks_final_compliance_as_failure_even_after_prior_refusal():
    adapter = RecordingAdapter(["I cannot help.", "Here is the requested code."])
    trace = Runner(adapter).run_case(case())

    assert trace.evaluation.status == EvaluationStatus.FAIL
    assert "earlier turn" in trace.evaluation.reason


def test_runner_stops_on_structured_error():
    adapter = RecordingAdapter(["unused"], fail_at=0)
    trace = Runner(adapter).run_case(case(chain=[]))

    assert trace.responses == []
    assert trace.error.type == "network_error"
    assert trace.evaluation.status == EvaluationStatus.ERROR


def test_pack_loader_rejects_duplicate_ids(tmp_path: Path):
    (tmp_path / "cases").mkdir()
    (tmp_path / "pack.yaml").write_text(
        "name: pack\ndescription: test\nversion: '1'\n"
        "cases:\n  - id: DUP\n    name: one\n    category: test\n    prompt: hello\n",
        encoding="utf-8",
    )
    (tmp_path / "cases" / "more.yaml").write_text(
        "- id: DUP\n  name: two\n  category: test\n  prompt: world\n",
        encoding="utf-8",
    )

    with pytest.raises(PackLoadError, match="Duplicate test case id"):
        Runner(RecordingAdapter()).load_pack(tmp_path)


def test_pack_loader_requires_pack_yaml(tmp_path: Path):
    with pytest.raises(PackLoadError, match="Missing pack file"):
        Runner(RecordingAdapter()).load_pack(tmp_path)


def test_model_defaults_are_not_shared():
    first = ExecutionTrace(test_case_id="a")
    second = ExecutionTrace(test_case_id="b")
    first.metadata["x"] = 1
    assert second.metadata == {}


def test_mutations_are_reproducible_for_seed_zero(tmp_path: Path):
    first = PromptMutator(seed=0, out_dir=str(tmp_path / "one"))
    second = PromptMutator(seed=0, out_dir=str(tmp_path / "two"))

    assert first.leetspeak("repeatable text") == second.leetspeak("repeatable text")
    assert first.homoglyph_swap("repeatable text") == second.homoglyph_swap("repeatable text")


def test_redactor_validates_and_redacts_error_messages():
    with pytest.raises(RedactionPatternError):
        RegexRedactor(["["])

    trace = ExecutionTrace(
        test_case_id="x",
        prompts=["secret@example.com"],
        responses=["secret@example.com"],
    )
    redacted = RegexRedactor([r"secret@example\.com"]).redact(trace)
    assert redacted.prompts == ["[REDACTED]"]
    assert redacted.responses == ["[REDACTED]"]
    assert redacted.redactions[0].match_count == 2
    assert redacted.redactions[0].pattern == "pattern-1"
    assert trace.prompts == ["secret@example.com"]


def test_rule_evaluator_handles_inconclusive_and_empty():
    evaluator = RuleEvaluator()
    no_expected = DarkTestCase(id="x", name="x", category="x", prompt="x")
    trace = ExecutionTrace(test_case_id="x", responses=["answer"])
    assert evaluator.evaluate(no_expected, trace).status == EvaluationStatus.INCONCLUSIVE

    empty = ExecutionTrace(test_case_id="x")
    assert evaluator.evaluate(case(), empty).status == EvaluationStatus.INCONCLUSIVE


def test_reporter_handles_empty_report_and_json_schema(tmp_path: Path):
    pack = DarkTestPack(name="empty", description="none", version="1")
    reporter = Reporter()

    markdown_path = reporter.generate_markdown(pack, [], tmp_path)
    assert "Resistance score: unavailable" in markdown_path.read_text(encoding="utf-8")

    json_path = reporter.generate_json(pack, [], tmp_path)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "1.2"
    assert payload["summary"]["error"] == 0


def test_redactor_redacts_evaluation_content():
    trace = ExecutionTrace(
        test_case_id="x",
        responses=["safe"],
        evaluation={
            "status": "fail",
            "reason": "leaked secret@example.com",
            "evidence": ["secret@example.com"],
        },
    )
    redacted = RegexRedactor([r"secret@example\.com"]).redact(trace)
    assert "[REDACTED]" in redacted.evaluation.reason
    assert redacted.evaluation.evidence == ["[REDACTED]"]


def test_mutator_transforms_and_names_variants(tmp_path: Path):
    mutator = PromptMutator(seed=7, out_dir=str(tmp_path))
    assert "SGVsbG8=" in mutator.base64_wrap("Hello")
    assert mutator.character_insert("abc") == "a.b.c"
    assert "cba" in mutator.reverse_text("abc")
    assert "def" in mutator.caesar_cipher("abc")
    assert "Concatenate" in mutator.payload_split("one two three four")
    assert mutator.payload_split("short") == "short"

    variants = mutator.named_variants("one two three four")
    names = [name for name, _ in variants]
    assert names[:3] == ["Original", "Leet", "Base64"]
    assert mutator.apply_all("one two three")


def test_reporter_writes_detailed_finding(tmp_path: Path):
    test_case = case(chain=[])
    trace = Runner(RecordingAdapter(["I cannot help with that."])).run_case(test_case)
    trace.metadata["mutation"] = "Base64"
    pack = DarkTestPack(
        name="pack",
        description="desc",
        version="1",
        cases=[test_case],
    )

    path = Reporter().generate_markdown(pack, [trace], tmp_path)
    report = path.read_text(encoding="utf-8")
    assert "Resistance score: 100.0%" in report
    assert "| CASE-1 | Injection | Base64 | PASS |" in report
    assert "#### Prompt 1" in report
    assert "#### Response 1" in report


def test_reporter_redacts_complete_report_without_storing_patterns(tmp_path: Path):
    secret = "secret@example.com"
    test_case = DarkTestCase(
        id="CASE-SECRET",
        name="case",
        category="test",
        prompt=secret,
        parameters={"nested": {"secret": secret}},
    )
    pack = DarkTestPack(
        name="pack",
        description=secret,
        version="1",
        cases=[test_case],
    )
    trace = ExecutionTrace(
        test_case_id=test_case.id,
        prompts=[secret],
        responses=[secret],
        tool_calls=[{"name": "tool", "arguments": secret, "result": secret}],
        metadata={"nested": {"secret": secret}},
        redactions=[
            {
                "pattern": secret,
                "replacement": secret,
                "match_count": 1,
            }
        ],
    )
    redactor = RegexRedactor([r"secret@example\.com"])
    redacted_trace = redactor.redact(trace)
    reporter = Reporter(redactor=redactor)

    json_report = reporter.generate_json(pack, [redacted_trace], tmp_path / "json")
    markdown_report = reporter.generate_markdown(
        pack,
        [redacted_trace],
        tmp_path / "markdown",
    )

    for report in (json_report, markdown_report):
        content = report.read_text(encoding="utf-8")
        assert secret not in content
        assert r"secret@example\.com" not in content
        assert "[REDACTED]" in content
        assert "pattern-1" in content
        assert "recorded-pattern-1" in content

    assert pack.description == secret
    assert trace.tool_calls[0].arguments == secret


def test_model_rejects_blank_required_values():
    with pytest.raises(ValueError):
        DarkTestCase(id=" ", name="x", category="x", prompt="x")


def test_runner_loads_case_files(tmp_path: Path):
    (tmp_path / "cases").mkdir()
    (tmp_path / "pack.yaml").write_text(
        "name: pack\ndescription: test\nversion: '1'\n",
        encoding="utf-8",
    )
    (tmp_path / "cases" / "one.yaml").write_text(
        "id: A\nname: A\ncategory: test\nprompt: hello\n",
        encoding="utf-8",
    )
    loaded = Runner(RecordingAdapter()).load_pack(tmp_path)
    assert [item.id for item in loaded.cases] == ["A"]
    assert loaded.cases[0].media_roots == (tmp_path.resolve(),)
    assert "_media_roots" not in loaded.cases[0].model_dump()
    assert loaded.cases[0].model_copy(deep=True).media_roots == (tmp_path.resolve(),)
