from __future__ import annotations

from typing import Any, Iterable

import regex

from .models import ExecutionTrace, Redaction

MAX_PATTERN_LENGTH = 512
REGEX_TIMEOUT_SECONDS = 0.025


class RedactionPatternError(ValueError):
    pass


class RegexRedactor:
    def __init__(self, patterns: Iterable[str]):
        self.patterns: list[tuple[str, regex.Pattern[str]]] = []
        for index, pattern in enumerate(patterns, start=1):
            if len(pattern) > MAX_PATTERN_LENGTH:
                raise RedactionPatternError(
                    f"Redaction pattern exceeds the {MAX_PATTERN_LENGTH}-character limit."
                )
            try:
                compiled = regex.compile(pattern)
            except regex.error as exc:
                raise RedactionPatternError(
                    f"Invalid redaction pattern pattern-{index}: {exc}"
                ) from exc
            self.patterns.append((f"pattern-{index}", compiled))

    @staticmethod
    def _subn(value: str, pattern: regex.Pattern[str]) -> tuple[str, int]:
        try:
            return pattern.subn(
                "[REDACTED]",
                value,
                timeout=REGEX_TIMEOUT_SECONDS,
            )
        except TimeoutError as exc:
            raise RedactionPatternError("A redaction pattern exceeded its time limit.") from exc

    def _redact_value(
        self,
        value: Any,
        pattern: regex.Pattern[str],
    ) -> tuple[Any, int]:
        if isinstance(value, str):
            return self._subn(value, pattern)
        if isinstance(value, dict):
            redacted: dict[Any, Any] = {}
            total = 0
            for key, item in value.items():
                safe_item, count = self._redact_value(item, pattern)
                redacted[key] = safe_item
                total += count
            return redacted, total
        if isinstance(value, list):
            redacted_list = []
            total = 0
            for item in value:
                safe_item, count = self._redact_value(item, pattern)
                redacted_list.append(safe_item)
                total += count
            return redacted_list, total
        if isinstance(value, tuple):
            redacted_list, total = self._redact_value(list(value), pattern)
            return tuple(redacted_list), total
        return value, 0

    def redact_value(self, value: Any) -> tuple[Any, list[Redaction]]:
        redacted = value
        results: list[Redaction] = []
        for pattern_id, pattern in self.patterns:
            redacted, count = self._redact_value(redacted, pattern)
            if count:
                results.append(Redaction(pattern=pattern_id, match_count=count))
        return redacted, results

    def redact(self, trace: ExecutionTrace) -> ExecutionTrace:
        redacted = trace.model_copy(deep=True)
        existing = [
            Redaction(
                pattern=f"recorded-pattern-{index}",
                match_count=item.match_count,
            )
            for index, item in enumerate(redacted.redactions, start=1)
        ]
        redacted.redactions = existing

        for pattern_id, pattern in self.patterns:
            total = 0
            redacted.test_case_id, count = self._subn(redacted.test_case_id, pattern)
            total += count
            redacted.prompts, count = self._redact_value(redacted.prompts, pattern)
            total += count
            redacted.responses, count = self._redact_value(redacted.responses, pattern)
            total += count

            for call in redacted.tool_calls:
                call.name, count = self._subn(call.name, pattern)
                total += count
                call.arguments, count = self._subn(call.arguments, pattern)
                total += count
                if call.result is not None:
                    call.result, count = self._subn(call.result, pattern)
                    total += count

            redacted.metadata, count = self._redact_value(redacted.metadata, pattern)
            total += count

            if redacted.error:
                redacted.error.message, count = self._subn(
                    redacted.error.message,
                    pattern,
                )
                total += count

            if redacted.evaluation:
                redacted.evaluation.reason, count = self._subn(
                    redacted.evaluation.reason,
                    pattern,
                )
                total += count
                redacted.evaluation.evidence, count = self._redact_value(
                    redacted.evaluation.evidence,
                    pattern,
                )
                total += count
                for assertion in redacted.evaluation.assertions:
                    assertion.reason, count = self._subn(assertion.reason, pattern)
                    total += count
                    assertion.evidence, count = self._redact_value(
                        assertion.evidence,
                        pattern,
                    )
                    total += count

            if total:
                redacted.redactions.append(
                    Redaction(pattern=pattern_id, match_count=total)
                )

        return redacted
