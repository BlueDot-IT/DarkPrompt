from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest

from darkprompt.adapters import anthropic, gemini, huggingface, mistral, ollama, openai
from darkprompt.adapters.common import MAX_MEDIA_BYTES, parse_media_payload
from darkprompt.models import TestCase as DarkTestCase


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code
        self.request = httpx.Request("POST", "https://example.test")
        self._response = httpx.Response(
            status_code,
            request=self.request,
            json=payload,
        )

    def raise_for_status(self):
        self._response.raise_for_status()

    def json(self):
        return self.payload


class FakeHTTPClient:
    def __init__(self, sink, payload, status_code=200, **kwargs):
        self.sink = sink
        self.payload = payload
        self.status_code = status_code
        self.kwargs = kwargs

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def post(self, url, **kwargs):
        self.sink.append({"url": url, **kwargs})
        return FakeResponse(self.payload, self.status_code)


def test_parse_media_payload(tmp_path):
    image = tmp_path / "payload.png"
    image.write_bytes(b"png-data")
    parsed = parse_media_payload(
        f"[MEDIA_PAYLOAD:{image}] Read this image.",
        allowed_roots=[tmp_path],
    )
    assert parsed.instruction == "Read this image."
    assert parsed.media_type == "image/png"
    assert parsed.data_b64

    with pytest.raises(FileNotFoundError):
        parse_media_payload(
            "[MEDIA_PAYLOAD:/missing/file.png] Read.",
            allowed_roots=[tmp_path],
        )


def test_parse_media_payload_confines_paths_and_size(tmp_path):
    pack = tmp_path / "pack"
    pack.mkdir()
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"secret")

    with pytest.raises(ValueError, match="allowed root"):
        parse_media_payload(f"[MEDIA_PAYLOAD:{outside}] inspect")
    with pytest.raises(ValueError, match="within"):
        parse_media_payload(
            f"[MEDIA_PAYLOAD:{outside}] inspect",
            allowed_roots=[pack],
        )

    link = pack / "link.png"
    link.symlink_to(outside)
    with pytest.raises(ValueError, match="within"):
        parse_media_payload(
            "[MEDIA_PAYLOAD:link.png] inspect",
            allowed_roots=[pack],
        )

    large = pack / "large.png"
    large.write_bytes(b"x" * (MAX_MEDIA_BYTES + 1))
    with pytest.raises(ValueError, match="size limit"):
        parse_media_payload(
            "[MEDIA_PAYLOAD:large.png] inspect",
            allowed_roots=[pack],
        )


def test_openai_sends_history_and_image(tmp_path):
    image = tmp_path / "payload.png"
    image.write_bytes(b"png-data")
    calls = []

    class FakeCompletions:
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content="I cannot help with that.",
                            tool_calls=[],
                        )
                    )
                ],
                usage=SimpleNamespace(model_dump=lambda: {"total_tokens": 3}),
            )

    adapter = openai.OpenAIAdapter(api_key="key", model="model")
    adapter.client = SimpleNamespace(
        chat=SimpleNamespace(completions=FakeCompletions())
    )
    test_case = DarkTestCase(
        id="x",
        name="x",
        category="x",
        prompt=f"[MEDIA_PAYLOAD:{image}] inspect",
    )
    test_case.allow_media_root(tmp_path)
    trace = adapter.execute(
        test_case,
        {"history": [{"role": "user", "content": "prior"}]},
    )

    assert trace.error is None
    assert calls[0]["messages"][0]["content"] == "prior"
    assert calls[0]["messages"][1]["content"][1]["type"] == "image_url"


@pytest.mark.parametrize(
    ("module", "adapter_class", "payload", "response_text"),
    [
        (
            anthropic,
            anthropic.AnthropicAdapter,
            {"content": [{"type": "text", "text": "anthropic"}], "usage": {}},
            "anthropic",
        ),
        (
            gemini,
            gemini.GeminiAdapter,
            {
                "candidates": [
                    {
                        "content": {"parts": [{"text": "gemini"}]},
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {},
            },
            "gemini",
        ),
        (
            mistral,
            mistral.MistralAdapter,
            {"choices": [{"message": {"content": "mistral"}}], "usage": {}},
            "mistral",
        ),
        (
            huggingface,
            huggingface.HuggingFaceAdapter,
            {"choices": [{"message": {"content": "huggingface"}}], "usage": {}},
            "huggingface",
        ),
        (
            ollama,
            ollama.OllamaAdapter,
            {"message": {"content": "ollama"}, "eval_count": 2},
            "ollama",
        ),
    ],
)
def test_http_adapters_send_history(
    monkeypatch,
    module,
    adapter_class,
    payload,
    response_text,
):
    calls = []
    monkeypatch.setattr(
        module.httpx,
        "Client",
        lambda **kwargs: FakeHTTPClient(calls, payload, **kwargs),
    )
    kwargs = {"model": "model", "base_url": "https://example.test"}
    if adapter_class is not ollama.OllamaAdapter:
        kwargs["api_key"] = "key"
    adapter = adapter_class(**kwargs)
    trace = adapter.execute(
        DarkTestCase(id="x", name="x", category="x", prompt="current"),
        {
            "history": [
                {"role": "user", "content": "prior"},
                {"role": "assistant", "content": "answer"},
            ]
        },
    )

    assert trace.responses == [response_text]
    body = calls[0]["json"]
    if adapter_class is gemini.GeminiAdapter:
        assert body["contents"][0]["parts"][0]["text"] == "prior"
    else:
        assert body["messages"][0]["content"] == "prior"


@pytest.mark.parametrize(
    "adapter",
    [
        openai.OpenAIAdapter(api_key=None),
        anthropic.AnthropicAdapter(api_key=None),
        gemini.GeminiAdapter(api_key=None),
        mistral.MistralAdapter(api_key=None),
        huggingface.HuggingFaceAdapter(api_key=None),
    ],
)
def test_missing_credentials_return_structured_error(monkeypatch, adapter):
    for key in (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "MISTRAL_API_KEY",
        "HUGGINGFACE_API_KEY",
        "HF_TOKEN",
    ):
        monkeypatch.delenv(key, raising=False)

    adapter.api_key = None
    if hasattr(adapter, "client"):
        adapter.client = None
    trace = adapter.execute(
        DarkTestCase(id="x", name="x", category="x", prompt="hello"),
        {},
    )
    assert trace.responses == []
    assert trace.error.type == "configuration_error"


def test_http_error_is_retryable(monkeypatch):
    calls = []
    monkeypatch.setattr(
        ollama.httpx,
        "Client",
        lambda **kwargs: FakeHTTPClient(
            calls,
            {"error": "busy"},
            status_code=503,
            **kwargs,
        ),
    )
    trace = ollama.OllamaAdapter(base_url="https://example.test").execute(
        DarkTestCase(id="x", name="x", category="x", prompt="hello"),
        {},
    )
    assert trace.error.status_code == 503
    assert trace.error.retryable is True


def test_huggingface_rejects_media_capability():
    adapter = huggingface.HuggingFaceAdapter(api_key="key")
    trace = adapter.execute(
        DarkTestCase(
            id="x",
            name="x",
            category="x",
            prompt="[MEDIA_PAYLOAD:/tmp/a.png] inspect",
        ),
        {},
    )
    assert trace.error.type == "unsupported_capability"
