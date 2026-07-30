from __future__ import annotations

import os
from typing import Any, Dict, Optional

import httpx

from ..adapter import AdapterCapabilities, TargetAdapter
from ..models import ExecutionTrace, TestCase
from .common import parse_media_payload, sanitized_http_error


class GeminiAdapter(TargetAdapter):
    capabilities = AdapterCapabilities(multi_turn=True, images=True)

    def __init__(
        self,
        model: str = "gemini-3.5-flash",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 30.0,
    ):
        self.model = model
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        root = (base_url or "https://generativelanguage.googleapis.com").rstrip("/")
        self.url = f"{root}/v1beta/models/{model}:generateContent"
        self.timeout = timeout

    def execute(self, test_case: TestCase, context: Dict[str, Any]) -> ExecutionTrace:
        if not self.api_key:
            return self.error_trace(
                test_case,
                error_type="configuration_error",
                message="Gemini API key missing. Set GEMINI_API_KEY.",
            )

        try:
            history = self.history(context)
            contents: list[dict[str, Any]] = []
            system_parts: list[str] = []
            for item in history:
                if item["role"] == "system":
                    system_parts.append(item["content"])
                    continue
                contents.append(
                    {
                        "role": "model" if item["role"] == "assistant" else "user",
                        "parts": [{"text": item["content"]}],
                    }
                )

            media = parse_media_payload(
                test_case.prompt, allowed_roots=test_case.media_roots
            )
            parts: list[dict[str, Any]]
            if media:
                parts = [
                    {
                        "inline_data": {
                            "mime_type": media.media_type,
                            "data": media.data_b64,
                        }
                    },
                    {"text": media.instruction},
                ]
            else:
                parts = [{"text": test_case.prompt}]
            contents.append({"role": "user", "parts": parts})

            payload: dict[str, Any] = {
                "contents": contents,
                "generationConfig": test_case.parameters,
            }
            if system_parts:
                payload["system_instruction"] = {
                    "parts": [{"text": "\n\n".join(system_parts)}]
                }

            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    self.url,
                    params={"key": self.api_key},
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()

            candidates = data.get("candidates") or []
            if not candidates:
                raise ValueError("Gemini returned no candidates.")
            response_text = "".join(
                part.get("text", "")
                for part in candidates[0].get("content", {}).get("parts", [])
            )
            return ExecutionTrace(
                test_case_id=test_case.id,
                prompts=[test_case.prompt],
                responses=[response_text],
                metadata={
                    "model": self.model,
                    "usage": data.get("usageMetadata", {}),
                    "finish_reason": candidates[0].get("finishReason"),
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
