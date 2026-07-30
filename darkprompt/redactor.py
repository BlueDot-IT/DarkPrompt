from __future__ import annotations

import re
from typing import Any, Iterable, Pattern

from .models import ExecutionTrace, Redaction, TestPack


class RedactionPatternError(ValueError):
    pass


class RegexRedactor:
    def __init__(self, patterns: Iterable[str]):
        self.patterns: list[tuple[str, Pattern[str]]] = []
        for index, pattern in enumerate(patterns, start=1):
            pattern_id = f"pattern-{index}"
            try:
                self.patterns.append((pattern_id, re.compile(pattern)))
            except re.error as exc:
                raise RedactionPatternError(
                    f"Invalid redaction pattern {pattern_id}: {exc}"
                ) from exc

    def redact(self, trace: ExecutionTrace) -> ExecutionTrace:
        existing_redactions = [
            Redaction(
                pattern=f"recorded-pattern-{index}",
                match_count=item.match_count,
            )
            for index, item in enumerate(trace.redactions, start=1)
        ]
        payload = trace.model_dump(mode="python", exclude={"redactions"})
        redacted_payload, redactions = self.redact_value(payload)
        redacted = ExecutionTrace.model_validate(redacted_payload)
        redacted.redactions = [*existing_redactions, *redactions]
        return redacted

    def redact_pack(self, pack: TestPack) -> TestPack:
        payload, _ = self.redact_value(pack.model_dump(mode="python", by_alias=True))
        return TestPack.model_validate(payload)

    def redact_value(self, value: Any) -> tuple[Any, list[Redaction]]:
        redacted = value
        results: list[Redaction] = []
        for pattern_id, regex in self.patterns:
            redacted, count = self._redact_recursive(redacted, regex)
            if count:
                results.append(Redaction(pattern=pattern_id, match_count=count))
        return redacted, results

    @staticmethod
    def _redact_recursive(value: Any, regex: Pattern[str]) -> tuple[Any, int]:
        if isinstance(value, str):
            return regex.subn("[REDACTED]", value)
        if isinstance(value, dict):
            redacted: dict[Any, Any] = {}
            total = 0
            for key, item in value.items():
                safe_key, key_count = RegexRedactor._redact_recursive(key, regex)
                safe_item, item_count = RegexRedactor._redact_recursive(item, regex)
                redacted[safe_key] = safe_item
                total += key_count + item_count
            return redacted, total
        if isinstance(value, list):
            redacted_list = []
            total = 0
            for item in value:
                safe_item, count = RegexRedactor._redact_recursive(item, regex)
                redacted_list.append(safe_item)
                total += count
            return redacted_list, total
        if isinstance(value, tuple):
            redacted_tuple, total = RegexRedactor._redact_recursive(list(value), regex)
            return tuple(redacted_tuple), total
        return value, 0
