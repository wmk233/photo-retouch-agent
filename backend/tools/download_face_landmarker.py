"""Fetch the pretrained MediaPipe Face Landmarker bundle used by the face lock.

Usage (from ``backend/``)::

    python -m tools.download_face_landmarker              # download if missing
    python -m tools.download_face_landmarker --status      # report without network
    python -m tools.download_face_landmarker --force       # re-download
    python -m tools.download_face_landmarker --from-file m.task   # offline install

The file is written to ``data/models/face_landmarker.task`` (override with
``--destination`` or ``PHOTO_AGENT_FACE_LANDMARK_MODEL``) and verified against a
pinned sha256 before it is trusted. No paid service is involved.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):  # allow ``python tools/download_face_landmarker.py``
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.infrastructure.face_landmark_model import (  # noqa: E402
    FACE_LANDMARK_MODEL_SHA256,
    FACE_LANDMARK_MODEL_SIZE_BYTES,
    FACE_LANDMARK_MODEL_URL,
    ModelDownloadError,
    copy_model,
    default_model_path,
    download_model,
    verify_model,
)


def _human(size: int) -> str:
    return f"{size / (1024 * 1024):.2f} MiB"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--destination",
        type=Path,
        default=None,
        help="Target .task path (defaults to data/models/face_landmarker.task).",
    )
    parser.add_argument("--status", action="store_true", help="Only report whether the model is present.")
    parser.add_argument("--force", action="store_true", help="Re-download even if the file is valid.")
    parser.add_argument(
        "--from-file",
        type=Path,
        default=None,
        help="Install an already-downloaded bundle (no network access).",
    )
    parser.add_argument("--url", default=FACE_LANDMARK_MODEL_URL, help="Override the download URL.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    target = args.destination or default_model_path()

    if args.status:
        ok, reason = verify_model(target)
        print(f"{target}: {'OK' if ok else reason}")
        return 0 if ok else 1

    if args.from_file is not None:
        try:
            installed = copy_model(args.from_file, target)
        except (ModelDownloadError, OSError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(f"installed {installed} ({_human(installed.stat().st_size)})")
        return 0

    ok, _ = verify_model(target)
    if ok and not args.force:
        print(f"already present: {target} ({_human(target.stat().st_size)})")
        return 0

    def _progress(written: int, total: int) -> None:
        if total:
            print(f"\r  downloading {written / total * 100:5.1f}%", end="", flush=True)

    print(f"downloading {args.url}")
    try:
        path = download_model(target, url=args.url, progress=_progress)
    except ModelDownloadError as exc:
        print(f"\nerror: {exc}", file=sys.stderr)
        print(
            "hint: install offline with --from-file, or set "
            "PHOTO_AGENT_FACE_LANDMARK_MODEL.",
            file=sys.stderr,
        )
        return 1
    print(f"\nwrote {path} ({_human(path.stat().st_size)})")
    print(f"sha256 {FACE_LANDMARK_MODEL_SHA256} (expected {FACE_LANDMARK_MODEL_SIZE_BYTES} bytes)")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
