from __future__ import annotations

import base64
import mimetypes
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional
from urllib.parse import urlsplit

import httpx

_MEDIA_RE = re.compile(r"^\[MEDIA_PAYLOAD:(?P<path>[^\]]+)\]\s*(?P<instruction>.*)$", re.DOTALL)
_ALLOWED_MEDIA_TYPES = frozenset({"image/gif", "image/jpeg", "image/png", "image/webp"})
MAX_MEDIA_BYTES = 10 * 1024 * 1024


@dataclass(frozen=True)
class MediaPayload:
    path: Path
    instruction: str
    media_type: str
    data_b64: str


def _within_roots(path: Path, roots: tuple[Path, ...]) -> bool:
    return any(path.is_relative_to(root) for root in roots)


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
    if max_bytes <= 0:
        raise ValueError("Media payload size limit must be positive.")

    requested = Path(match.group("path")).expanduser()
    candidates = (requested,) if requested.is_absolute() else tuple(
        root / requested for root in roots
    )
    path: Optional[Path] = None
    existing_candidate = False
    for candidate in candidates:
        try:
            resolved = candidate.resolve(strict=True)
        except OSError:
            continue
        existing_candidate = True
        if resolved.is_file() and _within_roots(resolved, roots):
            path = resolved
            break
    if path is None:
        if not existing_candidate:
            raise FileNotFoundError("Media payload does not exist.")
        raise ValueError("Media payload must be a file within an explicitly allowed root.")

    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    if media_type not in _ALLOWED_MEDIA_TYPES:
        raise ValueError(f"Unsupported media type: {media_type}")

    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError("Media payload could not be opened safely.") from exc

    try:
        with os.fdopen(descriptor, "rb") as handle:
            opened = os.fstat(handle.fileno())
            if not stat.S_ISREG(opened.st_mode):
                raise ValueError("Media payload must be a regular file.")

            try:
                resolved_after_open = path.resolve(strict=True)
                current = resolved_after_open.stat(follow_symlinks=False)
            except OSError as exc:
                raise ValueError("Media payload changed while it was being opened.") from exc
            if not _within_roots(resolved_after_open, roots):
                raise ValueError("Media payload escaped its explicitly allowed root.")
            if (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino):
                raise ValueError("Media payload changed while it was being opened.")

            data = handle.read(max_bytes + 1)
    except Exception:
        raise

    if len(data) > max_bytes:
        raise ValueError(f"Media payload exceeds the {max_bytes}-byte size limit.")

    return MediaPayload(
        path=path,
        instruction=match.group("instruction").strip() or "Analyze the attached image.",
        media_type=media_type,
        data_b64=base64.b64encode(data).decode("ascii"),
    )


def sanitized_http_error(exc: httpx.HTTPStatusError) -> str:
    parts = urlsplit(str(exc.response.request.url))
    host = parts.hostname or "provider"
    try:
        port = parts.port
    except ValueError:
        port = None
    authority = f"{host}:{port}" if port is not None else host
    scheme = parts.scheme or "https"
    return f"HTTP {exc.response.status_code} from {scheme}://{authority}"
