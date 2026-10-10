"""Run the pretrained SCHP ATR human parser over one photo (CPU, ONNX).

Usage -- from ``backend/``, with a Python that has ``onnxruntime`` installed::

    ../.venv/bin/python -m tools.run_human_parsing --image data/inputs/IMG_3651.jpg

``onnxruntime`` is deliberately *not* a dependency of the application
environment (it drags in a large native runtime for one offline step), so this
tool is normally run from the isolated experiment environment documented in
``experiments/p27-human-parsing``. It only performs inference: no training, no
fine-tuning, no paid provider, no download (the weight must already be on disk,
and its checksum is verified before it is loaded).

The output is the ``uint8`` ATR class map **in the photo's own coordinates** --
the input :func:`~app.domains.portrait.edit_mask.build_semantic_edit_masks`
consumes. It is a real inference result, not a hand-drawn mask.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

if __package__ in (None, ""):  # allow ``python tools/run_human_parsing.py``
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import settings  # noqa: E402
from app.domains.portrait.human_parsing import (  # noqa: E402
    ATR_CLASSES,
    HUMAN_PARSING_SHA256,
    SchpOnnxSegmenter,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image",
        default=str(REPO_ROOT / "data" / "inputs" / "IMG_3651.jpg"),
        help="photo to parse",
    )
    parser.add_argument(
        "--model",
        default=str(settings.human_parsing_model_path),
        help=(
            "SCHP ATR ONNX export (its .onnx.data sibling must sit next to it). "
            "Defaults to PHOTO_AGENT_HUMAN_PARSING_MODEL, else data/models."
        ),
    )
    parser.add_argument(
        "--out-labels",
        default=str(REPO_ROOT / "evidence" / "p27_labels.png"),
        help="where the uint8 class map is written (photo coordinates)",
    )
    parser.add_argument(
        "--stats-json",
        default=str(REPO_ROOT / "evidence" / "p27-parsing-stats.json"),
        help="where the timing/coverage record is written",
    )
    parser.add_argument("--runs", type=int, default=3, help="timed repetitions after warm-up")
    parser.add_argument(
        "--skip-checksum",
        action="store_true",
        help="skip the pinned sha256 check (recorded in the stats either way)",
    )
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _peak_rss_mib() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, Linux kilobytes.
    return peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    image_path = Path(args.image)
    if not image_path.is_file():
        raise SystemExit(f"Image not found: {image_path} (no download is attempted).")

    model_path = Path(args.model)
    if not model_path.is_file():
        raise SystemExit(
            f"Human-parsing model not found: {model_path}. Fetch the MIT-licensed "
            "ONNX export named in app/domains/portrait/human_parsing.py "
            "(no automatic download)."
        )

    checksum = _sha256(model_path)
    verified = checksum == HUMAN_PARSING_SHA256
    if not verified and not args.skip_checksum:
        raise SystemExit(
            f"Checksum mismatch for {model_path}:\n  expected {HUMAN_PARSING_SHA256}\n"
            f"  actual   {checksum}\nRefusing to run an unverified weight "
            "(pass --skip-checksum to override and record the override)."
        )

    image_rgb = np.asarray(Image.open(image_path).convert("RGB"))
    segmenter = SchpOnnxSegmenter(model_path)
    try:
        first = time.perf_counter()
        labels = segmenter.segment(image_rgb)
        cold_ms = (time.perf_counter() - first) * 1000.0
        timings = []
        for _ in range(max(int(args.runs), 1)):
            started = time.perf_counter()
            labels = segmenter.segment(image_rgb)
            timings.append((time.perf_counter() - started) * 1000.0)
    finally:
        segmenter.close()

    if labels.shape != image_rgb.shape[:2]:
        raise SystemExit(
            f"Parser returned {labels.shape}, expected the photo's own "
            f"{image_rgb.shape[:2]} -- the mask would not be in photo coordinates."
        )

    out_labels = Path(args.out_labels)
    out_labels.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(labels).save(out_labels)

    values, counts = np.unique(labels, return_counts=True)
    if len(values) < 3:
        raise SystemExit(
            f"The parser labelled the whole photo with {len(values)} class(es) "
            f"({[ATR_CLASSES.get(int(v), v) for v in values]}). That is a broken "
            "inference result, not a segmentation; refusing to write a label map "
            "that would produce a meaningless mask."
        )
    total = float(labels.size)
    shares = {
        ATR_CLASSES[int(value)]: round(float(count) / total, 4)
        for value, count in zip(values.tolist(), counts.tolist())
        if int(value) in ATR_CLASSES
    }

    record = {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "command": "python -m tools.run_human_parsing",
        "model": {
            "file": str(model_path),
            "sha256": checksum,
            "checksumVerified": bool(verified),
            "network": "SCHP ATR (ResNet-101, 18 classes)",
            "upstream": "GoGoDuck912/Self-Correction-Human-Parsing",
            "export": "pirocheto/schp-atr-18 (MIT)",
        },
        "image": str(image_path),
        "imageSize": [int(image_rgb.shape[1]), int(image_rgb.shape[0])],
        "device": "CPUExecutionProvider",
        "timingMs": {
            "cold": round(cold_ms, 1),
            "warm": [round(value, 1) for value in timings],
            "warmMedian": round(float(np.median(timings)), 1) if timings else None,
        },
        "peakRssMiB": round(_peak_rss_mib(), 1),
        "classShares": shares,
        "paidGenerations": 0,
        "paidVisionCalls": 0,
        "note": (
            "Real pretrained-model inference on the local CPU. No training, no "
            "download, no paid provider. The PNG holds the raw ATR class ids in "
            "the photo's own coordinates."
        ),
        "outputs": [str(out_labels)],
    }

    stats_path = Path(args.stats_json)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"labels {labels.shape} -> {out_labels}")
    print(
        "cold {cold:.0f} ms | warm median {warm:.0f} ms | peak RSS {rss:.0f} MiB".format(
            cold=cold_ms,
            warm=record["timingMs"]["warmMedian"] or 0.0,
            rss=record["peakRssMiB"],
        )
    )
    print(f"checksum verified: {verified} | wrote {stats_path}")
    for name, share in sorted(shares.items(), key=lambda item: -item[1]):
        print(f"  {name:<16} {share:6.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
