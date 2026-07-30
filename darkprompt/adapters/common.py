from __future__ import annotations

import base64
import mimetypes
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

_MEDIA_RE = re.compile(r"^\[MEDIA_PAYLOAD:(?P<path>[^\]]+)\]\s*(?P<instruction>.*)$", re.DOTALL)
MAX_MEDIA_BYTES = 10 * 1024 * 1024


@dataclass(frozen=True)
class MediaPayload:
    path: Path
    instruction: str
    media_type: str
    data_b64: str


def parse_media_payload(
    prompt: str,
    *,
    allowed_roots: Iterable[Path] = (),
    max_bytes: int = MAX_MEDIA_BYTES,
) -> Optional[MediaPayload]:
    match = _MEDIA_RE.match(prompt.strip())
    if not match:
        return None

    roots = tuple(root.expanduser().resolve() for root in allowed_roots)
    if not roots:
        raise ValueError("Media payloads require an explicitly allowed root.")

    requested = Path(match.group("path")).expanduser()
    path = requested if requested.is_absolute() else roots[0] / requested
    try:
        path = path.resolve(strict=True)
    except OSError as exc:
        raise FileNotFoundError(f"Media payload does not exist: {requested}") from exc

    if not path.is_file():
        raise FileNotFoundError(f"Media payload does not exist: {requested}")
    if not any(path.is_relative_to(root) for root in roots):
        raise ValueError("Media payload must remain within an explicitly allowed root.")
    if max_bytes <= 0:
        raise ValueError("Media payload size limit must be positive.")

    media_type = mimetypes.guess_type(path.name)[0]
    if not media_type or not media_type.startswith("image/"):
        raise ValueError("Media payload must use a recognized image MIME type.")

    with path.open("rb") as handle:
        data = handle.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError(f"Media payload exceeds the {max_bytes}-byte size limit.")

    return MediaPayload(
        path=path,
        instruction=match.group("instruction").strip() or "Analyze the attached image.",
        media_type=media_type,
        data_b64=base64.b64encode(data).decode("ascii"),
    )
