from __future__ import annotations

import os
from typing import Any, Dict, Optional

import httpx

from ..adapter import AdapterCapabilities, TargetAdapter
from ..models import ExecutionTrace, TestCase
from .common import sanitized_http_error


class HuggingFaceAdapter(TargetAdapter):
    capabilities = AdapterCapabilities(multi_turn=True)

    def __init__(
        self,
        model: str = "meta-llama/Llama-3.2-3B-Instruct",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 30.0,
    ):
        self.model = model
        self.api_key = (
            api_key
            or os.getenv("HUGGINGFACE_API_KEY")
            or os.getenv("HF_TOKEN")
        )
        self.url = f"{(base_url or 'https://router.huggingface.co').rstrip('/')}/v1/chat/completions"
        self.timeout = timeout

    def execute(self, test_case: TestCase, context: Dict[str, Any]) -> ExecutionTrace:
        if not self.api_key:
            return self.error_trace(
                test_case,
                error_type="configuration_error",
                message="Hugging Face token missing. Set HUGGINGFACE_API_KEY or HF_TOKEN.",
            )

        if test_case.prompt.lstrip().startswith("[MEDIA_PAYLOAD:"):
            return self.error_trace(
                test_case,
                error_type="unsupported_capability",
                message="The generic Hugging Face chat adapter does not support image payloads.",
            )

        try:
            messages: list[dict[str, Any]] = list(self.history(context))
            messages.append({"role": "user", "content": test_case.prompt})
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
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            return self.error_trace(
                test_case,
                error_type="http_error",
                message=sanitized_http_error(exc),
                retryable=status in {408, 409, 429} or status >= 500,
                status_code=status,
            )
        except Exception as exc:
            return self.error_trace(
                test_case,
                error_type=type(exc).__name__,
                message="Provider request failed.",
            )
