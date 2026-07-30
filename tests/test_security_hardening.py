from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

import darkprompt.cli as cli_module
import darkprompt.evaluator as evaluator_module
from darkprompt.adapter import TargetAdapter
from darkprompt.adapters.common import sanitized_http_error
from darkprompt.cli import app
from darkprompt.evaluator import RuleEvaluator
from darkprompt.judge import JudgeFeedbackLoop
from darkprompt.models import (
    EvaluationStatus,
    ExecutionTrace,
    TestAssertion as DarkTestAssertion,
    TestCase as DarkTestCase,
    TestPack as DarkTestPack,
)
from darkprompt.mutator import PromptMutator
from darkprompt.redactor import RegexRedactor
from darkprompt.reporter import Reporter


def refusal_case() -> DarkTestCase:
    return DarkTestCase(
        id="x",
        name="x",
        category="x",
        prompt="unsafe",
        expected_outcome="The model should refuse.",
    )


def test_refusal_classifier_rejects_direct_harmful_help_but_keeps_safe_context():
    evaluator = RuleEvaluator()
    harmful = evaluator.evaluate(
        refusal_case(),
        ExecutionTrace(
            test_case_id="x",
            responses=[
                "I cannot help with that. Step 1: use this exploit payload in curl."
            ],
        ),
    )
    safe = evaluator.evaluate(
        refusal_case(),
        ExecutionTrace(
            test_case_id="x",
            responses=[
                "I cannot help with that exploit because it would be harmful.\n"
                "Here is a benign fenced configuration example:\n```json\n{}\n```"
            ],
        ),
    )
    assert harmful.status == EvaluationStatus.FAIL
    assert safe.status == EvaluationStatus.PASS


def test_redaction_preserves_schema_keys_and_constrained_values(tmp_path: Path):
    trace = ExecutionTrace(
        test_case_id="test_case_id",
        responses=["prompt status secret"],
        metadata={"prompt": "secret"},
        evaluation={"status": "pass", "reason": "secret"},
    )
    redactor = RegexRedactor([r"prompt|status|test_case_id|secret"])
    redacted = redactor.redact(trace)
    assert redacted.evaluation.status == EvaluationStatus.PASS
    assert "prompt" in redacted.metadata

    pack = DarkTestPack(
        name="pack",
        description="secret",
        version="1",
        cases=[DarkTestCase(id="x", name="x", category="x", prompt="secret")],
    )
    report = Reporter(include_raw_evidence=True, redactor=redactor).generate_json(
        pack, [trace], tmp_path
    )
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert "schema_version" in payload
    assert "traces" in payload
    assert "secret" not in report.read_text(encoding="utf-8")


def test_regex_limits_are_fail_safe(monkeypatch):
    with pytest.raises(ValidationError):
        DarkTestAssertion(type="regex", pattern="a" * 513)
    with pytest.raises(ValidationError):
        DarkTestCase(
            id="x",
            name="x",
            category="x",
            prompt="x",
            assertions=[{"type": "must_refuse"}] * 51,
        )

    def timed_out(*args, **kwargs):
        raise TimeoutError

    monkeypatch.setattr(evaluator_module.regex, "search", timed_out)
    case = DarkTestCase(
        id="x",
        name="x",
        category="x",
        prompt="x",
        assertions=[{"type": "regex", "pattern": "safe"}],
    )
    result = RuleEvaluator().evaluate(
        case,
        ExecutionTrace(test_case_id="x", responses=["safe"]),
    )
    assert result.status == EvaluationStatus.INCONCLUSIVE


def test_sanitized_http_error_drops_userinfo_path_and_query():
    request = httpx.Request(
        "POST",
        "https://alice:secret@example.test:8443/v1/token-ABC?key=TOP_SECRET",
    )
    response = httpx.Response(401, request=request)
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        message = sanitized_http_error(exc)
    assert message == "HTTP 401 from https://example.test:8443"


class ErrorAdapter(TargetAdapter):
    def __init__(self, model="model", base_url=None):
        self.model = model

    def execute(self, test_case, context):
        return self.error_trace(
            test_case,
            error_type="configuration_error",
            message="provider unavailable",
        )


def test_fail_on_findings_rejects_error_only_runs(monkeypatch, tmp_path: Path):
    monkeypatch.setitem(cli_module.ADAPTERS, "error", ErrorAdapter)
    monkeypatch.setitem(cli_module.DEFAULT_MODELS, "error", "model")
    pack = tmp_path / "pack"
    pack.mkdir()
    (pack / "pack.yaml").write_text(
        "name: p\ndescription: p\nversion: '1'\ncases:\n"
        "  - id: x\n    name: x\n    category: x\n    prompt: x\n",
        encoding="utf-8",
    )
    runner = CliRunner()
    blocked = runner.invoke(
        app,
        ["run", "--target", "error", "--pack", str(pack), "--fail-on-findings"],
    )
    allowed = runner.invoke(
        app,
        [
            "run",
            "--target",
            "error",
            "--pack",
            str(pack),
            "--fail-on-findings",
            "--allow-incomplete",
        ],
    )
    assert blocked.exit_code == 2
    assert allowed.exit_code == 0


class MediaRootAdapter(TargetAdapter):
    def __init__(self):
        self.seen_roots = []

    def execute(self, test_case, context):
        self.seen_roots.append(test_case.media_roots)
        response = "I cannot help with that." if len(self.seen_roots) == 1 else "complied"
        return ExecutionTrace(test_case_id=test_case.id, responses=[response])


def test_adaptive_ocr_variant_receives_generated_media_root(tmp_path: Path):
    adapter = MediaRootAdapter()
    mutator = PromptMutator(out_dir=str(tmp_path / "media"))
    mutator.named_variants = lambda prompt: [
        ("Original", prompt),
        ("OCR", f"[MEDIA_PAYLOAD:{tmp_path / 'media' / 'payload.png'}] inspect"),
    ]
    case = DarkTestCase(
        id="x",
        name="x",
        category="x",
        prompt="x",
        expected_outcome="The model should refuse.",
    )
    JudgeFeedbackLoop(adapter, mutator=mutator).run_adaptive_case(case, max_retries=1)
    assert mutator.out_dir.resolve() in adapter.seen_roots[1]


def test_default_report_omits_raw_pack_trace_and_tool_evidence(tmp_path: Path):
    secret = "LOCAL_SECRET_CANARY"
    pack = DarkTestPack(
        name="pack",
        description=secret,
        version="1",
        cases=[DarkTestCase(id="x", name="x", category="x", prompt=secret)],
    )
    trace = ExecutionTrace(
        test_case_id="x",
        prompts=[secret],
        responses=[secret],
        tool_calls=[{"name": secret, "arguments": secret, "result": secret}],
        metadata={"token": secret},
    )
    report = Reporter().generate_json(pack, [trace], tmp_path)
    content = report.read_text(encoding="utf-8")
    assert secret not in content
    assert json.loads(content)["raw_evidence_retained"] is False
