from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, List, Optional

from . import __version__
from .models import EvaluationStatus, ExecutionTrace, TestPack
from .redactor import RegexRedactor


class Reporter:
    def __init__(
        self,
        *,
        include_raw_evidence: bool = False,
        redactor: Optional[RegexRedactor] = None,
    ):
        self.include_raw_evidence = include_raw_evidence
        self.redactor = redactor

    @staticmethod
    def _summary(traces: Iterable[ExecutionTrace]) -> dict[str, int]:
        counts = Counter(
            (
                trace.evaluation.status.value
                if trace.evaluation
                else EvaluationStatus.INCONCLUSIVE.value
            )
            for trace in traces
        )
        return {status.value: counts.get(status.value, 0) for status in EvaluationStatus}

    @staticmethod
    def _metrics(traces: Iterable[ExecutionTrace]) -> dict[str, object]:
        trace_list = list(traces)
        scored = [
            trace.evaluation.score
            for trace in trace_list
            if trace.evaluation and trace.evaluation.score is not None
        ]
        assertion_count = sum(
            len(trace.evaluation.assertions)
            for trace in trace_list
            if trace.evaluation
        )
        return {
            "assertion_count": assertion_count,
            "average_assertion_score": round(sum(scored) / len(scored), 4) if scored else None,
        }

    @staticmethod
    def _escape_table(value: object) -> str:
        return str(value).replace("|", "\\|").replace("\n", "<br>")

    @staticmethod
    def _fence(value: str) -> str:
        return value.replace("```", "` ` `")

    @staticmethod
    def _format_score(trace: ExecutionTrace) -> str:
        if not trace.evaluation or trace.evaluation.score is None:
            return "N/A"
        return f"{trace.evaluation.score:.0%}"

    @staticmethod
    def _byte_count(value: Any) -> int:
        if isinstance(value, str):
            return len(value.encode("utf-8"))
        return len(json.dumps(value, ensure_ascii=False, default=str).encode("utf-8"))

    @classmethod
    def _omitted(cls, value: Any) -> dict[str, object]:
        return {"retained": False, "bytes": cls._byte_count(value)}

    @classmethod
    def _omitted_collection(cls, values: list[Any]) -> dict[str, object]:
        return {
            "retained": False,
            "items": len(values),
            "bytes": cls._byte_count(values),
        }

    def _markdown_models(
        self, pack: TestPack, traces: List[ExecutionTrace]
    ) -> tuple[TestPack, List[ExecutionTrace]]:
        if not self.redactor:
            return pack, traces
        safe_pack = pack.model_copy(deep=True)
        safe_pack.name, _ = self.redactor.redact_value(safe_pack.name)
        safe_pack.description, _ = self.redactor.redact_value(safe_pack.description)
        safe_pack.version, _ = self.redactor.redact_value(safe_pack.version)
        return safe_pack, [self.redactor.redact(trace) for trace in traces]

    def _pack_json(self, pack: TestPack) -> dict[str, Any]:
        if self.include_raw_evidence:
            return pack.model_dump(mode="json", by_alias=True)
        return {
            "name": pack.name,
            "version": pack.version,
            "description": self._omitted(pack.description),
            "case_count": len(pack.cases),
            "cases": [
                {"id": case.id, "category": case.category}
                for case in pack.cases
            ],
        }

    def _trace_json(self, trace: ExecutionTrace) -> dict[str, Any]:
        if self.include_raw_evidence:
            return trace.model_dump(mode="json", by_alias=True)

        evaluation = None
        if trace.evaluation:
            evaluation = {
                "status": trace.evaluation.status.value,
                "confidence": trace.evaluation.confidence,
                "score": trace.evaluation.score,
                "reason": self._omitted(trace.evaluation.reason),
                "evidence": self._omitted_collection(trace.evaluation.evidence),
                "assertions": [
                    {
                        "type": assertion.type.value,
                        "outcome": assertion.outcome.value,
                        "scope": assertion.scope.value,
                        "weight": assertion.weight,
                        "confidence": assertion.confidence,
                        "turn": assertion.turn,
                        "reason": self._omitted(assertion.reason),
                        "evidence": self._omitted_collection(assertion.evidence),
                    }
                    for assertion in trace.evaluation.assertions
                ],
            }

        error = None
        if trace.error:
            error = {
                "type": trace.error.type,
                "retryable": trace.error.retryable,
                "status_code": trace.error.status_code,
                "message": self._omitted(trace.error.message),
            }

        return {
            "timestamp": trace.timestamp.isoformat(),
            "test_case_id": trace.test_case_id,
            "prompts": [self._omitted(value) for value in trace.prompts],
            "responses": [self._omitted(value) for value in trace.responses],
            "tool_calls": [
                {
                    "name": self._omitted(call.name),
                    "arguments": self._omitted(call.arguments),
                    "result": self._omitted(call.result) if call.result is not None else None,
                }
                for call in trace.tool_calls
            ],
            "redactions": [
                {
                    "pattern": redaction.pattern,
                    "replacement": redaction.replacement,
                    "match_count": redaction.match_count,
                }
                for redaction in trace.redactions
            ],
            "metadata": {
                "retained": False,
                "fields": len(trace.metadata),
                "bytes": self._byte_count(trace.metadata),
            },
            "error": error,
            "evaluation": evaluation,
        }

    def generate_markdown(
        self,
        pack: TestPack,
        traces: List[ExecutionTrace],
        out_dir: Path,
    ) -> Path:
        pack, traces = self._markdown_models(pack, traces)
        out_dir.mkdir(parents=True, exist_ok=True)
        report_path = out_dir / "report.md"
        summary = self._summary(traces)
        metrics = self._metrics(traces)

        completed = summary["pass"] + summary["fail"] + summary["partial"]
        resistance = round((summary["pass"] / completed) * 100, 1) if completed else None

        with report_path.open("w", encoding="utf-8") as handle:
            handle.write(f"# DarkPrompt Security Audit Report (v{__version__})\n\n")
            handle.write(f"## Pack: {pack.name} (v{pack.version})\n\n")
            if self.include_raw_evidence:
                handle.write(f"{pack.description}\n\n")
            else:
                handle.write(
                    f"Description retained=false; bytes={self._byte_count(pack.description)}\n\n"
                )
            handle.write("## Summary\n\n")
            handle.write(f"- Total traces: {len(traces)}\n")
            handle.write(
                "- Results: "
                + ", ".join(f"{name}={count}" for name, count in summary.items())
                + "\n"
            )
            handle.write(
                f"- Resistance score: {resistance}%\n"
                if resistance is not None
                else "- Resistance score: unavailable\n"
            )
            assertion_score = metrics["average_assertion_score"]
            handle.write(
                f"- Average assertion score: {float(assertion_score):.1%}\n"
                if assertion_score is not None
                else "- Average assertion score: unavailable\n"
            )
            handle.write(f"- Assertions evaluated: {metrics['assertion_count']}\n")
            handle.write(
                f"- Raw evidence retained: {str(self.include_raw_evidence).lower()}\n"
            )
            handle.write(
                f"- Redactions triggered: {sum(len(trace.redactions) for trace in traces)}\n\n"
            )

            handle.write("## Findings\n\n")
            handle.write("| Case ID | Category | Mutation | Status | Score | Reason |\n")
            handle.write("| :--- | :--- | :--- | :--- | :--- | :--- |\n")
            for trace in traces:
                evaluation = trace.evaluation
                status = (
                    evaluation.status.value.upper()
                    if evaluation
                    else EvaluationStatus.INCONCLUSIVE.value.upper()
                )
                if self.include_raw_evidence:
                    reason = evaluation.reason if evaluation else "Not evaluated."
                else:
                    reason_value = evaluation.reason if evaluation else "Not evaluated."
                    reason = f"retained=false; bytes={self._byte_count(reason_value)}"
                category = (
                    trace.metadata.get("category", "Unknown")
                    if self.include_raw_evidence
                    else "retained=false"
                )
                mutation = (
                    trace.metadata.get("mutation", "Original")
                    if self.include_raw_evidence
                    else "retained=false"
                )
                handle.write(
                    "| "
                    + " | ".join(
                        self._escape_table(value)
                        for value in (
                            trace.test_case_id,
                            category,
                            mutation,
                            status,
                            self._format_score(trace),
                            reason,
                        )
                    )
                    + " |\n"
                )

            handle.write("\n## Technical Details\n\n")
            for trace in traces:
                handle.write(f"### Case: {trace.test_case_id}\n\n")
                handle.write(f"- Timestamp: {trace.timestamp.isoformat()}\n")
                if trace.evaluation:
                    handle.write(
                        f"- Evaluation: {trace.evaluation.status.value.upper()} "
                        f"({trace.evaluation.confidence:.0%} confidence)\n"
                    )
                    handle.write(f"- Assertion score: {self._format_score(trace)}\n")
                    if self.include_raw_evidence:
                        handle.write(f"- Reason: {trace.evaluation.reason}\n")
                    else:
                        handle.write(
                            "- Reason: retained=false; "
                            f"bytes={self._byte_count(trace.evaluation.reason)}\n"
                        )
                if trace.error:
                    if self.include_raw_evidence:
                        handle.write(
                            f"- Error: {trace.error.type}: {trace.error.message}\n"
                        )
                    else:
                        handle.write(
                            f"- Error: {trace.error.type}; message retained=false; "
                            f"bytes={self._byte_count(trace.error.message)}\n"
                        )
                handle.write("\n")

                if trace.evaluation and trace.evaluation.assertions:
                    handle.write("#### Assertions\n\n")
                    handle.write("| Type | Scope | Weight | Outcome | Confidence | Reason |\n")
                    handle.write("| :--- | :--- | ---: | :--- | ---: | :--- |\n")
                    for result in trace.evaluation.assertions:
                        scope = (
                            f"turn:{result.turn}"
                            if result.turn is not None
                            else result.scope.value
                        )
                        reason = (
                            result.reason
                            if self.include_raw_evidence
                            else f"retained=false; bytes={self._byte_count(result.reason)}"
                        )
                        handle.write(
                            "| "
                            + " | ".join(
                                self._escape_table(value)
                                for value in (
                                    result.type.value,
                                    scope,
                                    result.weight,
                                    result.outcome.value.upper(),
                                    f"{result.confidence:.0%}",
                                    reason,
                                )
                            )
                            + " |\n"
                        )
                    handle.write("\n")

                for index, prompt in enumerate(trace.prompts, start=1):
                    handle.write(f"#### Prompt {index}\n\n")
                    if self.include_raw_evidence:
                        handle.write(f"```text\n{self._fence(prompt)}\n```\n\n")
                    else:
                        handle.write(
                            f"retained=false; bytes={self._byte_count(prompt)}\n\n"
                        )
                    if index <= len(trace.responses):
                        response = trace.responses[index - 1]
                        handle.write(f"#### Response {index}\n\n")
                        if self.include_raw_evidence:
                            handle.write(
                                f"```text\n{self._fence(response)}\n```\n\n"
                            )
                        else:
                            handle.write(
                                f"retained=false; bytes={self._byte_count(response)}\n\n"
                            )

                if trace.redactions:
                    handle.write("#### Redactions\n\n")
                    for redaction in trace.redactions:
                        handle.write(
                            f"- `{redaction.pattern}`: {redaction.match_count} matches\n"
                        )
                    handle.write("\n")

                handle.write("---\n\n")

        return report_path

    def generate_json(
        self,
        pack: TestPack,
        traces: List[ExecutionTrace],
        out_dir: Path,
    ) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        report_path = out_dir / "report.json"
        data = {
            "schema_version": "1.3",
            "darkprompt_version": __version__,
            "raw_evidence_retained": self.include_raw_evidence,
            "pack": self._pack_json(pack),
            "summary": self._summary(traces),
            "metrics": self._metrics(traces),
            "traces": [self._trace_json(trace) for trace in traces],
        }
        if self.redactor:
            data, _ = self.redactor.redact_value(data)
        with report_path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        return report_path
