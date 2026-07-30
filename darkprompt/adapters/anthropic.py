from __future__ import annotations

import os
from typing import Any, Dict, Optional

import httpx

from ..adapter import AdapterCapabilities, TargetAdapter
from ..models import ExecutionTrace, TestCase
from .common import parse_media_payload


class AnthropicAdapter(TargetAdapter):
    capabilities = AdapterCapabilities(multi_turn=True, images=True)

    def __init__(
        self,
        model: str = "claude-sonnet-5",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 30.0,
    ):
        self.model = model
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.url = f"{(base_url or 'https://api.anthropic.com').rstrip('/')}/v1/messages"
        self.timeout = timeout

    def execute(self, test_case: TestCase, context: Dict[str, Any]) -> ExecutionTrace:
        if not self.api_key:
            return self.error_trace(
                test_case,
                error_type="configuration_error",
                message="Anthropic API key missing. Set ANTHROPIC_API_KEY.",
            )

        try:
            history = self.history(context)
            system_parts = [item["content"] for item in history if item["role"] == "system"]
            messages: list[dict[str, Any]] = [
                item for item in history if item["role"] in {"user", "assistant"}
            ]

            media = parse_media_payload(
                test_case.prompt, allowed_roots=test_case.media_roots
            )
            if media:
                content: Any = [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media.media_type,
                            "data": media.data_b64,
                        },
                    },
                    {"type": "text", "text": media.instruction},
                ]
            else:
                content = test_case.prompt
            messages.append({"role": "user", "content": content})

            payload: dict[str, Any] = {
                "model": self.model,
                "max_tokens": 1024,
                "messages": messages,
                **test_case.parameters,
            }
            if system_parts:
                payload["system"] = "\n\n".join(system_parts)

            headers = {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(self.url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()

            response_text = "".join(
                block.get("text", "")
                for block in data.get("content", [])
                if block.get("type") == "text"
            )
            return ExecutionTrace(
                test_case_id=test_case.id,
                prompts=[test_case.prompt],
                responses=[response_text],
                metadata={"model": self.model, "usage": data.get("usage", {})},
            )
        except FileNotFoundError as exc:
            return self.error_trace(
                test_case, error_type="media_error", message=str(exc)
            )
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            return self.error_trace(
                test_case,
                error_type="http_error",
                message=str(exc),
                retryable=status in {408, 409, 429} or status >= 500,
                status_code=status,
            )
        except Exception as exc:
            return self.error_trace(
                test_case,
                error_type=type(exc).__name__,
                message=str(exc),
            )
