from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, List, Optional

from . import __version__
from .models import EvaluationStatus, ExecutionTrace, TestPack

if TYPE_CHECKING:
    from .redactor import RegexRedactor


class Reporter:
    def __init__(self, redactor: Optional["RegexRedactor"] = None):
        self.redactor = redactor

    def _safe_inputs(
        self,
        pack: TestPack,
        traces: List[ExecutionTrace],
    ) -> tuple[TestPack, List[ExecutionTrace]]:
        if not self.redactor:
            return pack, traces
        return (
            self.redactor.redact_pack(pack),
            [self.redactor.redact(trace) for trace in traces],
        )

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

    def generate_markdown(
        self,
        pack: TestPack,
        traces: List[ExecutionTrace],
        out_dir: Path,
    ) -> Path:
        pack, traces = self._safe_inputs(pack, traces)
        out_dir.mkdir(parents=True, exist_ok=True)
        report_path = out_dir / "report.md"
        summary = self._summary(traces)
        metrics = self._metrics(traces)

        completed = summary["pass"] + summary["fail"] + summary["partial"]
        resistance = round((summary["pass"] / completed) * 100, 1) if completed else None

        with report_path.open("w", encoding="utf-8") as handle:
            handle.write(f"# DarkPrompt Security Audit Report (v{__version__})\n\n")
            handle.write(f"## Pack: {pack.name} (v{pack.version})\n\n")
            handle.write(f"{pack.description}\n\n")
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
                reason = evaluation.reason if evaluation else "Not evaluated."
                handle.write(
                    "| "
                    + " | ".join(
                        self._escape_table(value)
                        for value in (
                            trace.test_case_id,
                            trace.metadata.get("category", "Unknown"),
                            trace.metadata.get("mutation", "Original"),
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
                handle.write(f"- Category: {trace.metadata.get('category', 'Unknown')}\n")
                handle.write(f"- Timestamp: {trace.timestamp.isoformat()}\n")
                if trace.evaluation:
                    handle.write(
                        f"- Evaluation: {trace.evaluation.status.value.upper()} "
                        f"({trace.evaluation.confidence:.0%} confidence)\n"
                    )
                    handle.write(f"- Assertion score: {self._format_score(trace)}\n")
                    handle.write(f"- Reason: {trace.evaluation.reason}\n")
                if trace.error:
                    handle.write(f"- Error: {trace.error.type}: {trace.error.message}\n")
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
                                    result.reason,
                                )
                            )
                            + " |\n"
                        )
                    handle.write("\n")

                for index, prompt in enumerate(trace.prompts, start=1):
                    handle.write(f"#### Prompt {index}\n\n")
                    handle.write(f"```text\n{self._fence(prompt)}\n```\n\n")
                    if index <= len(trace.responses):
                        handle.write(f"#### Response {index}\n\n")
                        handle.write(
                            f"```text\n{self._fence(trace.responses[index - 1])}\n```\n\n"
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
        pack, traces = self._safe_inputs(pack, traces)
        out_dir.mkdir(parents=True, exist_ok=True)
        report_path = out_dir / "report.json"
        data = {
            "schema_version": "1.2",
            "darkprompt_version": __version__,
            "pack": pack.model_dump(mode="json", by_alias=True),
            "summary": self._summary(traces),
            "metrics": self._metrics(traces),
            "traces": [trace.model_dump(mode="json", by_alias=True) for trace in traces],
        }
        with report_path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        return report_path
