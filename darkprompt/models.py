from __future__ import annotations

import re
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from jsonschema import Draft202012Validator, SchemaError
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator, model_validator


class EvaluationStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    PARTIAL = "partial"
    ERROR = "error"
    SKIPPED = "skipped"
    INCONCLUSIVE = "inconclusive"


class AssertionType(str, Enum):
    MUST_REFUSE = "must_refuse"
    MUST_NOT_REFUSE = "must_not_refuse"
    MUST_CONTAIN = "must_contain"
    MUST_NOT_CONTAIN = "must_not_contain"
    REGEX = "regex"
    JSON_SCHEMA = "json_schema"
    TOOL_CALL_ALLOWED = "tool_call_allowed"
    TOOL_CALL_FORBIDDEN = "tool_call_forbidden"
    SEMANTIC_JUDGE = "semantic_judge"


class AssertionScope(str, Enum):
    FINAL = "final"
    ANY = "any"
    ALL = "all"
    TURN = "turn"


class AssertionOutcome(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    INCONCLUSIVE = "inconclusive"


class Redaction(BaseModel):
    pattern: str
    replacement: str = "[REDACTED]"
    match_count: int = 0


class ToolCall(BaseModel):
    name: str
    arguments: str
    result: Optional[str] = None


class TraceError(BaseModel):
    type: str
    message: str
    retryable: bool = False
    status_code: Optional[int] = None


class TestAssertion(BaseModel):
    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

    type: AssertionType
    scope: AssertionScope = AssertionScope.FINAL
    value: Optional[str] = None
    pattern: Optional[str] = None
    json_schema: Optional[Dict[str, Any]] = Field(default=None, alias="schema")
    tool_names: List[str] = Field(default_factory=list)
    turn: Optional[int] = Field(default=None, ge=1)
    weight: float = Field(default=1.0, gt=0)
    case_sensitive: bool = False
    description: Optional[str] = None

    @model_validator(mode="after")
    def validate_assertion(self) -> "TestAssertion":
        if self.scope == AssertionScope.TURN and self.turn is None:
            raise ValueError("turn is required when scope is 'turn'")
        if self.scope != AssertionScope.TURN and self.turn is not None:
            raise ValueError("turn may only be set when scope is 'turn'")

        if self.type in {AssertionType.MUST_CONTAIN, AssertionType.MUST_NOT_CONTAIN}:
            if not self.value:
                raise ValueError(f"value is required for {self.type.value}")
        elif self.type == AssertionType.REGEX:
            if not self.pattern:
                raise ValueError("pattern is required for regex assertions")
            try:
                re.compile(self.pattern)
            except re.error as exc:
                raise ValueError(f"invalid assertion regex: {exc}") from exc
        elif self.type == AssertionType.JSON_SCHEMA:
            if self.json_schema is None:
                raise ValueError("schema is required for json_schema assertions")
            try:
                Draft202012Validator.check_schema(self.json_schema)
            except SchemaError as exc:
                raise ValueError(f"invalid JSON schema: {exc.message}") from exc
        elif self.type in {
            AssertionType.TOOL_CALL_ALLOWED,
            AssertionType.TOOL_CALL_FORBIDDEN,
        }:
            if not self.tool_names:
                raise ValueError(f"tool_names is required for {self.type.value}")
            if self.scope != AssertionScope.FINAL:
                raise ValueError("tool-call assertions currently require scope 'final'")
        elif self.type == AssertionType.SEMANTIC_JUDGE:
            if not (self.value or self.description):
                raise ValueError("value or description is required for semantic_judge")
        return self


class AssertionResult(BaseModel):
    type: AssertionType
    outcome: AssertionOutcome
    scope: AssertionScope
    weight: float
    reason: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    evidence: List[str] = Field(default_factory=list)
    turn: Optional[int] = None


class EvaluationResult(BaseModel):
    status: EvaluationStatus
    reason: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    evidence: List[str] = Field(default_factory=list)
    assertions: List[AssertionResult] = Field(default_factory=list)


class ConversationMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ExecutionTrace(BaseModel):
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    test_case_id: str
    prompts: List[str] = Field(default_factory=list)
    responses: List[str] = Field(default_factory=list)
    tool_calls: List[ToolCall] = Field(default_factory=list)
    redactions: List[Redaction] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[TraceError] = None
    evaluation: Optional[EvaluationResult] = None


class TestCase(BaseModel):
    _media_roots: tuple[Path, ...] = PrivateAttr(default_factory=tuple)

    id: str
    name: str
    category: str
    prompt: str
    expected_outcome: Optional[str] = None
    assertions: List[TestAssertion] = Field(default_factory=list)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    chain: List[str] = Field(default_factory=list)

    @property
    def media_roots(self) -> tuple[Path, ...]:
        return self._media_roots

    def allow_media_root(self, root: Path) -> None:
        resolved = root.expanduser().resolve()
        if resolved not in self._media_roots:
            self._media_roots = (*self._media_roots, resolved)

    @field_validator("id", "name", "category", "prompt")
    @classmethod
    def require_non_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be empty")
        return value


class TestPack(BaseModel):
    name: str
    description: str
    version: str
    cases: List[TestCase] = Field(default_factory=list)
