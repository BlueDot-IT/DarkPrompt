from __future__ import annotations

import os
from typing import Any, Dict, Optional

from openai import OpenAI

from ..adapter import AdapterCapabilities, TargetAdapter
from ..models import ExecutionTrace, TestCase, ToolCall
from .common import parse_media_payload


class OpenAIAdapter(TargetAdapter):
    capabilities = AdapterCapabilities(multi_turn=True, images=True, tools=True)

    def __init__(
        self,
        model: str = "gpt-5.6-luna",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 30.0,
    ):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = (
            OpenAI(api_key=self.api_key, base_url=base_url, timeout=timeout)
            if self.api_key
            else None
        )

    def execute(self, test_case: TestCase, context: Dict[str, Any]) -> ExecutionTrace:
        if not self.client:
            return self.error_trace(
                test_case,
                error_type="configuration_error",
                message="OpenAI API key missing. Set OPENAI_API_KEY.",
            )

        try:
            messages: list[dict[str, Any]] = list(self.history(context))
            media = parse_media_payload(
                test_case.prompt, allowed_roots=test_case.media_roots
            )
            if media:
                content: Any = [
                    {"type": "text", "text": media.instruction},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{media.media_type};base64,{media.data_b64}"
                        },
                    },
                ]
            else:
                content = test_case.prompt

            messages.append({"role": "user", "content": content})
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                **test_case.parameters,
            )

            message = response.choices[0].message
            response_text = message.content or ""
            trace = ExecutionTrace(
                test_case_id=test_case.id,
                prompts=[test_case.prompt],
                responses=[response_text],
                metadata={
                    "model": self.model,
                    "usage": response.usage.model_dump() if response.usage else {},
                },
            )

            for call in message.tool_calls or []:
                trace.tool_calls.append(
                    ToolCall(name=call.function.name, arguments=call.function.arguments)
                )
            return trace
        except FileNotFoundError as exc:
            return self.error_trace(
                test_case,
                error_type="media_error",
                message=str(exc),
            )
        except Exception as exc:
            status_code = getattr(exc, "status_code", None)
            retryable = status_code in {408, 409, 429} or (
                isinstance(status_code, int) and status_code >= 500
            )
            return self.error_trace(
                test_case,
                error_type=type(exc).__name__,
                message="Provider request failed.",
                retryable=retryable,
                status_code=status_code,
            )
