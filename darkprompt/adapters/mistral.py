from __future__ import annotations

import os
from typing import Any, Dict, Optional

import httpx

from ..adapter import AdapterCapabilities, TargetAdapter
from ..models import ExecutionTrace, TestCase
from .common import parse_media_payload


class MistralAdapter(TargetAdapter):
    capabilities = AdapterCapabilities(multi_turn=True, images=True)

    def __init__(
        self,
        model: str = "mistral-medium-latest",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 30.0,
    ):
        self.model = model
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
        self.url = f"{(base_url or 'https://api.mistral.ai').rstrip('/')}/v1/chat/completions"
        self.timeout = timeout

    def execute(self, test_case: TestCase, context: Dict[str, Any]) -> ExecutionTrace:
        if not self.api_key:
            return self.error_trace(
                test_case,
                error_type="configuration_error",
                message="Mistral API key missing. Set MISTRAL_API_KEY.",
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
                        "image_url": {"url": f"data:{media.media_type};base64,{media.data_b64}"},
                    },
                ]
            else:
                content = test_case.prompt
            messages.append({"role": "user", "content": content})

            payload = {
                "model": self.model,
                "messages": messages,
                **test_case.parameters,
            }
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(self.url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()

            return ExecutionTrace(
                test_case_id=test_case.id,
                prompts=[test_case.prompt],
                responses=[data["choices"][0]["message"].get("content", "")],
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
