"""Build the standard local edit package for one photo and one styled look.

    photo + style id
      -> original image, hair / clothing / protected masks
      -> prompt_bundle.json, manifest.json, mask previews

It is a local-only step: no provider is called, no key is read, and nothing is
submitted anywhere. The output directory is what a local generation engine (the
current candidate is the ComfyUI ``Inpaint-CropAndStitch`` pair) is pointed at
afterwards, by a human.

Usage -- from ``backend/``::

    # masks from the live SCHP checkpoint (needs onnxruntime, see P2.7)
    ../.venv/bin/python -m tools.build_local_edit_package \\
        --photo data/inputs/IMG_3651.jpg --style artistic_wave

    # or reuse an ATR label map a real SCHP run already produced (no onnxruntime)
    ../.venv/bin/python -m tools.build_local_edit_package \\
        --photo data/inputs/IMG_3651.jpg --style artistic_wave \\
        --labels evidence/p28_collar_check/p28_labels.png

    # what ids the stylist owns
    ../.venv/bin/python -m tools.build_local_edit_package --list-styles

    # or drive a *stored* AI plan instead of an axis: the prompt bundle then
    # carries that plan's own text verbatim (see --list-plans)
    ../.venv/bin/python -m tools.build_local_edit_package \\
        --photo data/inputs/IMG_3651.jpg --plan ai_elegant_refined
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

if __package__ in (None, ""):  # allow ``python tools/build_local_edit_package.py``
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import settings  # noqa: E402
from app.services.local_edit_package import (  # noqa: E402
    LocalEditPackageBuilder,
    LocalEditPackageUnavailable,
    StoredPlanError,
    UnknownStyleError,
    decode_stored_plan,
    find_stored_plan,
    resolve_style_id,
    stored_plan_ids,
    style_ids,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


class CachedLabelMapSegmenter:
    """Serve an ATR label map a real SCHP run already produced.

    Same interface as ``SchpOnnxSegmenter`` (``segment``/``close``), so the
    production mask service cannot tell the two apart and the geometry under
    test is exactly the shipped one. It exists so this tool runs on a machine
    without onnxruntime, not to replace the real parser.
    """

    def __init__(self, labels: np.ndarray) -> None:
        self._labels = np.asarray(labels)

    def segment(self, image_rgb: np.ndarray) -> np.ndarray:
        height, width = np.asarray(image_rgb).shape[:2]
        if self._labels.shape != (height, width):
            raise ValueError(
                f"cached label map is {self._labels.shape}, the photo is "
                f"{(height, width)}; it is not in the photo's own coordinates"
            )
        return self._labels

    def close(self) -> None:  # parity with an ONNX session
        return None


def build_builder(
    segmenter_factory: Any = None, config: Any = None
) -> LocalEditPackageBuilder:
    """Construct the package builder.

    A seam, not indirection: the suite drives the real CLI end to end while
    injecting the label-map shim a machine without onnxruntime needs.
    """
    return LocalEditPackageBuilder(
        settings if config is None else config, segmenter_factory=segmenter_factory
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--photo",
        default=None,
        help="the original photograph, any format Pillow can read",
    )
    parser.add_argument(
        "--style",
        default=None,
        help=(
            "a style id or alias: " + ", ".join(("urban_short", "artistic_wave", "bold_street"))
        ),
    )
    parser.add_argument(
        "--plan",
        default=None,
        help=(
            "a stored AI plan id (see --list-plans); the package then carries "
            "that plan's own edit prompt verbatim instead of an axis composition"
        ),
    )
    parser.add_argument(
        "--job",
        default=None,
        help="the job file a --plan id lives in (defaults to searching data/jobs)",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="package directory (defaults to artifacts/local-edit-package/<timestamp>)",
    )
    parser.add_argument(
        "--labels",
        default=None,
        help=(
            "an ATR uint8 class map PNG already produced by tools.run_human_parsing; "
            "when omitted SCHP is run live (requires onnxruntime)"
        ),
    )
    parser.add_argument(
        "--list-styles",
        action="store_true",
        help="print every style id this project owns and exit",
    )
    parser.add_argument(
        "--list-plans",
        action="store_true",
        help="print every plan id the local job store holds and exit",
    )
    parser.add_argument(
        "--json",
        default=None,
        help="optional path for a copy of the package manifest",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_styles:
        print("style ids accepted by --style:")
        for style_id in style_ids():
            axis = resolve_style_id(style_id)
            marker = "*" if style_id == axis else " "
            print(f"  {marker} {style_id}  ->  {axis}")
        return 0

    if args.list_plans:
        print("plan ids accepted by --plan (from data/jobs):")
        ids = stored_plan_ids()
        if not ids:
            print("  (the job store holds no readable plan)")
            return 1
        for plan_id in ids:
            print(f"    {plan_id}")
        return 0

    if args.style and args.plan:
        print("BLOCKED -- --style and --plan are mutually exclusive", file=sys.stderr)
        return 2

    if not args.photo or not (args.style or args.plan):
        print(
            "--photo plus one of --style / --plan is required "
            "(or use --list-styles / --list-plans)",
            file=sys.stderr,
        )
        return 2

    photo = Path(args.photo).expanduser().resolve()
    if not photo.is_file():
        print(f"BLOCKED -- photo not found: {photo}", file=sys.stderr)
        return 1

    stored_plan = None
    if args.plan:
        try:
            stored_plan = (
                find_stored_plan(args.plan)
                if not args.job
                else decode_stored_plan(args.job, args.plan)
            )
        except StoredPlanError as exc:
            print(f"BLOCKED -- {exc}", file=sys.stderr)
            return 2
        axis = stored_plan.plan_id
    else:
        try:
            axis = resolve_style_id(args.style)
        except UnknownStyleError as exc:
            print(f"BLOCKED -- {exc}", file=sys.stderr)
            return 2

    segmenter_factory = None
    labels_path: Path | None = None
    if args.labels:
        labels_path = Path(args.labels).expanduser()
        if not labels_path.is_file():
            print(f"BLOCKED -- label map not found: {labels_path}", file=sys.stderr)
            return 1
        labels = np.asarray(Image.open(labels_path))
        if labels.ndim != 2:
            print(
                "BLOCKED -- the label map must be a single-channel PNG of ATR class ids",
                file=sys.stderr,
            )
            return 1
        segmenter_factory = lambda _model_path: CachedLabelMapSegmenter(labels)  # noqa: E731

    builder = build_builder(segmenter_factory)
    print("=== local edit package (no network, no key, no generation) ===")
    print(f"photo          : {photo}")
    if stored_plan is not None:
        print(
            f"plan           : {axis}  ({stored_plan.proposal.title!r}, "
            f"{stored_plan.job_id}, {stored_plan.model_provider} "
            f"{stored_plan.model_name}, status={stored_plan.status})"
        )
        print(f"plan source    : {stored_plan.job_path}")
        print("prompt source  : plan.edit_prompt (verbatim, not an axis)")
    else:
        print(f"style          : {axis}  (requested {args.style!r})")
    print(
        "labels from    : "
        + (str(labels_path) if labels_path else "live SCHP inference")
    )
    try:
        package = builder.build(photo, axis, args.out_dir, plan=stored_plan)
    except LocalEditPackageUnavailable as exc:
        print()
        print("BLOCKED")
        print(f"  - {exc}")
        return 1

    manifest = package.manifest
    coverage, overlap = manifest["coverage"], manifest["overlap"]
    print()
    for line in package.summary_lines():
        print(line)
    print()
    print("files:")
    for key, path in sorted(package.files.items()):
        print(f"  {key:26s} {path.relative_to(package.root)}")
    print()
    print("engine handoff:")
    for step in manifest["engine_handoff"]["steps"]:
        print(f"  pass {step['order']} ({step['id']}): mask={step['mask']}")
    print()
    print(f"api keys required: none (local package only)")
    print(f"paid generations : 0")
    print(f"manifest         : {package.manifest_path}")

    if args.json:
        target = Path(args.json).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"manifest copy    : {target}")

    print()
    failed = [rule for rule in manifest["mask_rules"] if not rule["passed"]]
    if failed or not manifest["mask_rules_passed"]:
        print("BLOCKED")
        for rule in failed:
            print(f"  - mask rule failed: {rule['id']} ({rule['detail']})")
        return 1
    print("READY_FOR_LOCAL_ENGINE")
    print(
        "  the package is complete and the masks passed their rules; the next "
        "step is a human deciding which engine consumes it."
    )
    print(
        f"  face coverage {coverage['protected_identity']['pixels']} px, "
        f"edit area {coverage['editable_combined']['pixels']} px, "
        f"overlap {overlap['protected_and_edit']['pixels']} px"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
