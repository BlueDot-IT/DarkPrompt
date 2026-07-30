from __future__ import annotations

from typing import Any, Dict, Optional

import httpx

from ..adapter import AdapterCapabilities, TargetAdapter
from ..models import ExecutionTrace, TestCase
from .common import parse_media_payload, sanitized_http_error


class OllamaAdapter(TargetAdapter):
    capabilities = AdapterCapabilities(multi_turn=True, images=True)

    def __init__(
        self,
        model: str = "mistral",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 60.0,
    ):
        del api_key
        self.model = model
        self.url = f"{(base_url or 'http://localhost:11434').rstrip('/')}/api/chat"
        self.timeout = timeout

    def execute(self, test_case: TestCase, context: Dict[str, Any]) -> ExecutionTrace:
        try:
            messages: list[dict[str, Any]] = list(self.history(context))
            media = parse_media_payload(
                test_case.prompt, allowed_roots=test_case.media_roots
            )
            current: dict[str, Any] = {
                "role": "user",
                "content": media.instruction if media else test_case.prompt,
            }
            if media:
                current["images"] = [media.data_b64]
            messages.append(current)

            payload = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": test_case.parameters,
            }
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(self.url, json=payload)
                response.raise_for_status()
                data = response.json()

            return ExecutionTrace(
                test_case_id=test_case.id,
                prompts=[test_case.prompt],
                responses=[data.get("message", {}).get("content", "")],
                metadata={
                    "model": self.model,
                    "total_duration": data.get("total_duration"),
                    "prompt_eval_count": data.get("prompt_eval_count"),
                    "eval_count": data.get("eval_count"),
                },
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
