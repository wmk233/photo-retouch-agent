"""Locate (and, on demand, fetch) the pretrained MediaPipe face-landmark model.

The strict face lock depends on the pretrained ``google-ai-edge/mediapipe`` Face
Landmarker task bundle. The bundle is a ~3.6 MB ``.task`` file that Google
publishes for direct download; it is *not* committed to this repository
(binary blobs do not belong in git), so this module owns the one place that
knows its filename, URL and checksum.

Nothing here trains a model, and nothing here calls a paid service: the download
is a single GET against Google's public storage bucket, and it is only performed
when a human explicitly asks for it (``python -m tools.download_face_landmarker``
or :func:`ensure_model`), never as a side effect of generating a photo.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import urllib.request
from pathlib import Path
from typing import Any, Callable, ContextManager

from app.core.config import Settings, settings

FACE_LANDMARK_MODEL_FILENAME = "face_landmarker.task"

#: Official MediaPipe model card asset (float16 / version 1).
FACE_LANDMARK_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

#: Integrity pin for the file above. Verified locally before it is used at all.
FACE_LANDMARK_MODEL_SHA256 = (
    "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"
)
FACE_LANDMARK_MODEL_SIZE_BYTES = 3_758_596


class ModelDownloadError(RuntimeError):
    """Raised when the landmark bundle cannot be fetched or verified."""


def default_model_path(config: Settings | None = None) -> Path:
    return Path((config or settings).face_landmark_model_path)


def sha256_of(path: Path | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_model(
    path: Path | str,
    *,
    expected_sha256: str = FACE_LANDMARK_MODEL_SHA256,
    expected_size: int | None = FACE_LANDMARK_MODEL_SIZE_BYTES,
) -> tuple[bool, str]:
    """Return ``(ok, reason)`` for a candidate model file."""
    candidate = Path(path)
    if not candidate.is_file():
        return False, f"missing: {candidate}"
    if expected_size is not None and candidate.stat().st_size != expected_size:
        return False, (
            f"size mismatch: {candidate.stat().st_size} != {expected_size} bytes"
        )
    if expected_sha256:
        actual = sha256_of(candidate)
        if actual != expected_sha256:
            return False, f"sha256 mismatch: {actual} != {expected_sha256}"
    return True, "ok"


def model_available(path: Path | str | None = None, config: Settings | None = None) -> bool:
    """Cheap presence check used by status endpoints (no hashing)."""
    target = Path(path) if path is not None else default_model_path(config)
    return target.is_file() and target.stat().st_size > 0


def download_model(
    destination: Path | str | None = None,
    *,
    url: str = FACE_LANDMARK_MODEL_URL,
    expected_sha256: str | None = FACE_LANDMARK_MODEL_SHA256,
    expected_size: int | None = FACE_LANDMARK_MODEL_SIZE_BYTES,
    open_url: Callable[[str], ContextManager[Any]] | None = None,
    progress: Callable[[int, int], None] | None = None,
    config: Settings | None = None,
) -> Path:
    """Download the landmark bundle and return the verified file path.

    The file lands in a temporary sibling first and is moved into place with
    :func:`os.replace`, so an interrupted download can never leave a truncated
    model behind for a later run to trust.
    """
    target = Path(destination) if destination is not None else default_model_path(config)
    target.parent.mkdir(parents=True, exist_ok=True)
    opener = open_url or urllib.request.urlopen
    temp_path = target.with_suffix(target.suffix + ".partial")

    try:
        with opener(url) as response:
            total = int(response.headers.get("Content-Length") or 0)
            written = 0
            with open(temp_path, "wb") as handle:
                while True:
                    chunk = response.read(256 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
                    written += len(chunk)
                    if progress is not None:
                        progress(written, total)
    except Exception as exc:  # pragma: no cover - network failure path
        temp_path.unlink(missing_ok=True)
        raise ModelDownloadError(f"Could not download {url}: {exc}") from exc

    ok, reason = verify_model(
        temp_path, expected_sha256=expected_sha256 or "", expected_size=expected_size
    )
    if not ok:
        temp_path.unlink(missing_ok=True)
        raise ModelDownloadError(f"Downloaded landmark bundle failed verification ({reason}).")

    os.replace(temp_path, target)
    return target


def ensure_model(
    destination: Path | str | None = None,
    *,
    download: bool = True,
    **kwargs: Any,
) -> Path:
    """Return a usable model path, downloading it once when necessary."""
    target = Path(destination) if destination is not None else default_model_path()
    ok, _ = verify_model(
        target,
        expected_sha256=kwargs.get("expected_sha256", FACE_LANDMARK_MODEL_SHA256),
        expected_size=kwargs.get("expected_size", FACE_LANDMARK_MODEL_SIZE_BYTES),
    )
    if ok:
        return target
    if not download:
        raise ModelDownloadError(f"Face landmark bundle unavailable at {target}.")
    return download_model(target, **kwargs)


def copy_model(
    source: Path | str,
    destination: Path | str | None = None,
    *,
    expected_sha256: str | None = None,
    expected_size: int | None = None,
) -> Path:
    """Install a locally supplied model file (offline / air-gapped setup).

    The installed copy is verified against the pinned size and checksum before it
    is trusted; ``expected_*`` only exists so a caller that carries its own pin
    (a mirror, an air-gapped bundle) can be verified the same way. The file is
    staged next to the destination and moved into place only once it verifies, so
    a rejected bundle never leaves an unverified model where a later run -- which
    only checks that the file exists -- would pick it up.
    """
    target = Path(destination) if destination is not None else default_model_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target.with_suffix(target.suffix + ".partial")
    shutil.copyfile(source, temp_path)
    ok, reason = verify_model(
        temp_path,
        expected_sha256=expected_sha256 or FACE_LANDMARK_MODEL_SHA256,
        expected_size=expected_size if expected_size is not None else FACE_LANDMARK_MODEL_SIZE_BYTES,
    )
    if not ok:
        temp_path.unlink(missing_ok=True)
        raise ModelDownloadError(f"Installed model failed verification ({reason}).")
    os.replace(temp_path, target)
    return target
