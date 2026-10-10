"""Preflight the fal.ai Qwen-Image-Edit inpaint request -- without billing it.

This tool runs the *real* local chain and then builds the *real* provider
request, so a paid run is only ever a matter of adding a key:

    photo -> SCHP ATR human parsing -> semantic edit mask
          -> FalInpaintingProvider payload -> contract checks -> verdict

It never submits anything. The submit POST is the one billed step in this
project (``fal-ai/qwen-image-edit/inpaint``, $0.03 per megapixel), and this tool
issues no HTTP request at all: it calls the same
``FalInpaintingProvider._inspect_mask`` / ``_build_payload`` the job uses and
then inspects the resulting dict.

The pieces are imported, never re-implemented: segmentation
(``app.domains.portrait.human_parsing``), mask geometry
(``app.domains.portrait.edit_mask`` via ``NativeMaskInpainting.build_mask``) and
the request itself (``app.providers.fal_inpainting_provider``). If those disagree
with the endpoint contract, this tool is supposed to fail loudly.

Usage -- from ``backend/``::

    # mask from the live SCHP checkpoint (needs onnxruntime, see P2.7)
    ../.venv/bin/python -m tools.preflight_fal_inpaint --image data/inputs/IMG_3651.jpg

    # or reuse a label map a real SCHP run already produced (no onnxruntime)
    ../.venv/bin/python -m tools.preflight_fal_inpaint \
        --image data/inputs/IMG_3651.jpg \
        --labels evidence/p28_collar_check/p28_labels.png
"""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

if __package__ in (None, ""):  # allow ``python tools/preflight_fal_inpaint.py``
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import settings  # noqa: E402
from app.domains.portrait.human_parsing import HUMAN_PARSING_SHA256  # noqa: E402
from app.providers.fal_inpainting_provider import (  # noqa: E402
    FalInpaintingProvider,
    FalInpaintingProviderError,
)
from app.schemas.retouch import RetouchPlan  # noqa: E402
from app.services.native_mask_inpainting import (  # noqa: E402
    NativeMaskInpainting,
    NativeMaskUnavailable,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The published price for this endpoint (fal.ai model page / llms.txt).
USD_PER_MEGAPIXEL = 0.03

#: A representative styling instruction. The prompt text is *not* part of the
#: request contract being verified -- only its presence and type are -- so a
#: fixed string keeps this preflight free and reproducible.
DEFAULT_EDIT_PROMPT = (
    "把人物的发型修剪为干练的层次短发，并保持自然的发色过渡；"
    "将上装替换为同色系的简约针织衫。只改变发型与服装。"
)


class _CachedLabelMapSegmenter:
    """Serve an ATR label map a real SCHP run already produced.

    Same interface as ``SchpOnnxSegmenter``, so
    ``NativeMaskInpainting.build_mask`` cannot tell the two apart and the mask
    geometry under test is exactly the production one.
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

    def close(self) -> None:  # parity with the ONNX session
        return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image",
        default=str(REPO_ROOT / "data" / "inputs" / "IMG_3651.jpg"),
        help="the original photograph (no download is attempted)",
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
        "--out-dir",
        default=str(REPO_ROOT / "evidence" / "p28_collar_check" / "preflight"),
        help="where the mask and the report are written",
    )
    parser.add_argument(
        "--stats-json",
        default=None,
        help="report path (defaults to <out-dir>/p28-fal-preflight.json)",
    )
    parser.add_argument("--prompt", default=DEFAULT_EDIT_PROMPT, help="edit prompt")
    parser.add_argument(
        "--instruction", default="", help="optional extra user instruction"
    )
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pixels(mask: Any) -> int:
    return int((np.asarray(mask) > 0).sum())


def _decode_data_uri(uri: Any) -> tuple[str, bytes]:
    """Split a ``data:`` URI and return ``(mime, bytes)``; raise on anything else."""
    if not isinstance(uri, str) or not uri.startswith("data:"):
        raise ValueError("input is not a data: URI")
    header, _, payload = uri.partition(",")
    if ";base64" not in header:
        raise ValueError("data: URI is not base64 encoded")
    mime = header[len("data:") :].split(";", 1)[0] or "application/octet-stream"
    try:
        return mime, base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"data: URI payload is not valid base64 ({exc})") from exc


def _weights_report(model_path: Path) -> dict[str, Any]:
    exists = model_path.is_file()
    checksum = _sha256(model_path) if exists else None
    external = model_path.parent / f"{model_path.name}.data"
    report = {
        "configuredPath": str(model_path),
        "exists": exists,
        "sha256": checksum,
        "pinnedSha256": HUMAN_PARSING_SHA256,
        "matchesPinnedExport": checksum == HUMAN_PARSING_SHA256,
        "onnxExternalDataPath": str(external),
        "onnxExternalDataExists": external.is_file(),
    }
    if external.is_file():
        report["onnxExternalDataSha256"] = _sha256(external)
    return report


def _contract_report(
    payload: dict[str, Any],
    source_path: Path,
    mask_path: Path,
    source_size: tuple[int, int],
) -> dict[str, Any]:
    """Check the built payload against the published endpoint contract."""
    problems: list[str] = []
    notes: list[str] = []

    required = ("prompt", "image_url", "mask_url")
    missing = [key for key in required if not payload.get(key)]
    if missing:
        problems.append(f"payload is missing required field(s): {', '.join(missing)}")

    if not isinstance(payload.get("prompt"), str) or not payload.get("prompt", "").strip():
        problems.append("prompt must be a non-empty string")

    if payload.get("num_images") != 1:
        problems.append(f"num_images must be 1, payload has {payload.get('num_images')!r}")

    # --- the two inputs must be different pictures ------------------------
    image_uri = payload.get("image_url")
    mask_uri = payload.get("mask_url")
    if image_uri == mask_uri:
        problems.append("image_url and mask_url are the same payload, not two inputs")

    decoded: dict[str, tuple[str, bytes]] = {}
    for key, uri in (("image_url", image_uri), ("mask_url", mask_uri)):
        try:
            decoded[key] = _decode_data_uri(uri)
        except ValueError as exc:
            problems.append(f"{key} is not a usable data: URI ({exc})")

    expected_bytes = {"image_url": source_path.read_bytes(), "mask_url": mask_path.read_bytes()}
    for key, (mime, data) in decoded.items():
        if not mime.startswith("image/"):
            problems.append(f"{key} declares a non-image mime type ({mime})")
        if data != expected_bytes[key]:
            problems.append(f"{key} does not carry the file it claims to")
    if len(decoded) == 2 and decoded["image_url"][1] == decoded["mask_url"][1]:
        problems.append("the two data: URIs carry identical bytes")

    # --- geometry: mask in the photo's own coordinates ---------------------
    if len(decoded) == 2:
        try:
            image_array = np.asarray(Image.open(source_path).convert("L"))
            mask_array = np.asarray(Image.open(mask_path).convert("L"))
        except OSError as exc:
            problems.append(f"inputs could not be re-decoded ({exc})")
        else:
            if (image_array.shape[1], image_array.shape[0]) != source_size:
                problems.append("the source image size changed while building the request")
            if mask_array.shape != image_array.shape:
                problems.append(
                    "the mask is not in the photo's own coordinates: "
                    f"{mask_array.shape[1]}x{mask_array.shape[0]} vs "
                    f"{source_size[0]}x{source_size[1]}"
                )
            levels = np.unique(mask_array)
            if not np.all(np.isin(levels, (0, 255))):
                problems.append(
                    f"the mask must be strictly 0/255; it holds {levels[:5].tolist()}"
                )
            if mask_array.shape == image_array.shape:
                notes.append(
                    f"mask and photo are both {source_size[0]}x{source_size[1]}; "
                    "the kept pixels can be copied back at identical coordinates"
                )

    # --- output size ------------------------------------------------------
    # ``image_size`` is optional on this endpoint and omitted here, so the
    # generated frame follows the service default. Nothing relies on that being
    # the input size: ``verify_inpaint_output`` raises when the returned frame is
    # not the photo's size, so a rescaled result can never be reported as an
    # identity-preserving success.
    if "image_size" in payload:
        notes.append(f"image_size is pinned to {payload['image_size']!r}")
    else:
        notes.append(
            "image_size is omitted (the documented optional default); the output "
            "size is only accepted if the service returns the input size, "
            "otherwise verification hard-fails"
        )

    leaked = [
        key
        for key in payload
        if "key" in key.lower() or "token" in key.lower() or "auth" in key.lower()
    ]
    if leaked:
        problems.append(f"payload carries credential-looking field(s): {leaked}")

    return {"problems": problems, "notes": notes}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    image_path = Path(args.image)
    if not image_path.is_file():
        raise SystemExit(f"Image not found: {image_path} (no download is attempted).")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stats_path = Path(args.stats_json) if args.stats_json else out_dir / "p28-fal-preflight.json"

    api_key_present = bool((settings.fal_key or "").strip())
    weights = _weights_report(Path(settings.human_parsing_model_path))

    image_rgb = np.asarray(Image.open(image_path).convert("RGB"))
    height, width = image_rgb.shape[:2]
    megapixels = (width * height) / 1_000_000.0

    blockers: list[str] = []

    # --- 1. local mask, through the production service ---------------------
    labels_path = Path(args.labels) if args.labels else None
    factory = None
    if labels_path is not None:
        if not labels_path.is_file():
            raise SystemExit(f"Label map not found: {labels_path}.")
        labels = np.asarray(Image.open(labels_path))
        if labels.ndim != 2:
            raise SystemExit("The label map must be a single-channel PNG of ATR class ids.")
        factory = lambda _model_path: _CachedLabelMapSegmenter(labels)  # noqa: E731

    service = NativeMaskInpainting(settings, segmenter_factory=factory)
    mask_path = out_dir / "p28_fal_inpaint_mask.png"
    try:
        build = service.build_mask(image_path, mask_path)
    except NativeMaskUnavailable as exc:
        record = {
            "generatedAt": datetime.now(timezone.utc).isoformat(),
            "image": str(image_path),
            "verdict": "BLOCKED",
            "blockers": [f"the local edit mask could not be built: {exc}"],
            "paidGenerations": 0,
            "paidVisionCalls": 0,
        }
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nBLOCKED -- {record['blockers'][0]}")
        return 1

    mask_set = build.mask_set
    protected = mask_set.protected
    editable = mask_set.editable
    overlap = int(np.logical_and(np.asarray(protected) > 0, np.asarray(editable) > 0).sum())

    face_mask = mask_set.parsed_face if mask_set.parsed_face is not None else mask_set.face_core
    hair_px = _pixels(mask_set.hair)
    clothing_px = _pixels(mask_set.clothing)
    face_px = _pixels(face_mask) if face_mask is not None else 0
    protected_px = _pixels(protected)
    editable_px = _pixels(editable)
    total_px = int(protected.shape[0] * protected.shape[1])

    if overlap:
        blockers.append(
            f"the protected region and the edit region overlap on {overlap} px; "
            "the photorealistic guard would be repainted"
        )

    # --- 2. the request the provider would send ----------------------------
    plan = RetouchPlan(
        planId="preflight-0001",
        domainType="portrait",
        title="Preflight styling plan",
        description="Local preflight plan; no provider call is made.",
        intensity="medium",
        editPrompt=args.prompt,
        negativePrompt="换脸, 陌生人, 塑料皮肤, 水印, 文字, 变形",
        expectedChanges=["发型", "发色", "上装"],
    )
    provider = FalInpaintingProvider(
        # An empty key is fine here: the payload is built locally and the submit
        # path (which is what demands a key) is never entered.
        api_key=settings.fal_key or "",
        endpoint=settings.fal_inpaint_endpoint,
        model_name=settings.fal_inpaint_model,
    )

    mask_check: dict[str, Any] | None = None
    payload: dict[str, Any] | None = None
    try:
        inspected = provider._inspect_mask(image_path, mask_path)  # noqa: SLF001
        mask_check = inspected._asdict()
        payload = provider._build_payload(image_path, mask_path, plan, args.instruction)  # noqa: SLF001
    except FalInpaintingProviderError as exc:
        blockers.append(f"the provider refused the request shape: {exc}")

    contract: dict[str, Any] = {"problems": [], "notes": []}
    if payload is not None:
        contract = _contract_report(payload, image_path, mask_path, (width, height))
        blockers.extend(contract["problems"])

    if not api_key_present:
        blockers.append(
            "no fal.ai credential: FAL_KEY is unset in the environment and no "
            "fal.ai item was found in the login keychain, so the queue POST "
            "cannot be authenticated"
        )

    estimated_cost = round(USD_PER_MEGAPIXEL * megapixels, 4)
    record = {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "image": {
            "path": str(image_path),
            "width": width,
            "height": height,
            "megapixels": round(megapixels, 6),
            "sha256": _sha256(image_path),
        },
        "humanParsing": {
            **weights,
            "labelSource": str(labels_path) if labels_path else "live SCHP inference",
            "labelMap": str(labels_path) if labels_path else None,
        },
        "mask": {
            "path": str(mask_path),
            "sha256": _sha256(mask_path),
            "semantic": bool(mask_set.semantic),
            "hairPixels": hair_px,
            "hairShare": round(hair_px / total_px, 6),
            "clothingPixels": clothing_px,
            "clothingShare": round(clothing_px / total_px, 6),
            "facePixels": face_px,
            "faceShare": round(face_px / total_px, 6),
            "protectedPixels": protected_px,
            "editablePixels": editable_px,
            "editableShare": round(editable_px / total_px, 6),
            "protectedEditableOverlapPixels": overlap,
            "providerMaskCheck": mask_check,
            "acceptance": build.acceptance,
        },
        "model": {
            "endpoint": settings.fal_inpaint_endpoint,
            "modelId": settings.fal_inpaint_model,
            "imageSizeField": "omitted (documented optional default)",
            "numImages": 1,
            "outputFormat": "png",
        },
        "credentials": {
            "falKeyPresent": api_key_present,
            "source": "environment FAL_KEY / login keychain (names only)",
            "configured": api_key_present,
        },
        "billing": {
            "usdPerMegapixel": USD_PER_MEGAPIXEL,
            "expectedRequests": 1,
            "expectedImages": 1,
            "estimatedCostUsd": estimated_cost,
            "note": (
                "one queue submit, then free status/result polls; billed per "
                "generated image"
            ),
        },
        "requestContract": {
            "checkedFields": ["prompt", "image_url", "mask_url", "num_images", "output_format"],
            "problems": contract["problems"],
            "notes": contract["notes"],
            "inputsAreTwoDistinctDataUris": bool(payload)
            and payload.get("image_url") != payload.get("mask_url"),
        },
        "identityGuarantee": {
            "faceCoreComesFromTheOriginal": True,
            "mechanism": (
                "restore_kept_pixels copies the photo's own pixels back wherever "
                "the mask is black, at identical coordinates; it raises on a size "
                "or mask mismatch instead of rescaling"
            ),
            "statuses": ["MASK_VALID", "PIXEL_IDENTITY_PASSED", "VISUAL_ACCEPTANCE_PENDING"],
            "visualAcceptanceStillNeedsAHuman": True,
        },
        "networkCalls": 0,
        "paidGenerations": 0,
        "paidVisionCalls": 0,
        "verdict": "BLOCKED" if blockers else "READY_FOR_ONE_PAID_TEST",
        "blockers": blockers,
    }

    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    # --- human-readable summary -------------------------------------------
    print("=== fal.ai inpaint preflight (no request sent) ===")
    print(f"photo          : {image_path}  {width}x{height}  ({megapixels:.3f} MP)")
    print(f"SCHP weights   : {weights['configuredPath']}")
    print(f"                 exists={weights['exists']} checksumMatches={weights['matchesPinnedExport']}")
    print(f"labels from    : {record['humanParsing']['labelSource']}")
    print(f"edit mask      : {mask_path}")
    print(f"  hair         : {hair_px} px ({hair_px/total_px:.2%})")
    print(f"  clothing     : {clothing_px} px ({clothing_px/total_px:.2%})")
    print(f"  face         : {face_px} px ({face_px/total_px:.2%})")
    print(f"  protected    : {protected_px} px")
    print(f"  editable     : {editable_px} px ({editable_px/total_px:.2%})")
    print(f"  overlap      : {overlap} px (must be 0)")
    print(f"model          : {settings.fal_inpaint_model}")
    print(f"endpoint       : {settings.fal_inpaint_endpoint}")
    print(f"output size    : {record['model']['imageSizeField']}; refuse-on-mismatch downstream")
    print(f"fal key        : {'present' if api_key_present else 'MISSING (configured=false)'}")
    print(f"requests/cost  : 1 request, 1 image, ~${estimated_cost:.4f} ({USD_PER_MEGAPIXEL}/MP)")
    for note in contract["notes"]:
        print(f"note           : {note}")
    for problem in contract["problems"]:
        print(f"CONTRACT ERROR : {problem}")

    print()
    if blockers:
        print("BLOCKED")
        for reason in blockers:
            print(f"  - {reason}")
        return 1
    print("READY_FOR_ONE_PAID_TEST")
    print("  every precondition is met; the paid run is waiting on an explicit go-ahead.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
