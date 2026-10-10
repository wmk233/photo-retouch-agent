"""Build the native-mask edit mask and verify the masked inpaint result.

This is the job-layer half of the P2.8 flow: it turns the source photograph into
the black/white instruction the masked backend needs, and it owns the decision
to *refuse* when that instruction cannot be built honestly.

    original photo
      -> SCHP ATR human parsing          (app.domains.portrait.human_parsing)
      -> MediaPipe face landmarks        (app.domains.portrait.face_lock)
      -> black/white edit mask           (app.domains.portrait.edit_mask)
      -> fal.ai Qwen Inpaint             (app.providers.fal_inpainting_provider)
      -> kept-pixel restore + checks     (app.domains.portrait.inpaint_verify)

It is deliberately *not* a second job framework: the existing job store and
dispatcher drive it, and it only ever runs inside a job that already exists. It
does not generate, train or bill anything -- the one paid step is the provider
call the job already makes.

Two hard rules are enforced here:

* the mask is built from *this* photo, in *this* photo's coordinates, by local
  inference only, so it can never be stale or mismatched;
* when the mask cannot be trusted -- the parser is missing, no face is found, or
  the old collar cannot be separated from the neck skin -- the run raises and the
  job fails, instead of falling back to a whole-image edit that would repaint the
  face.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from app.core.config import Settings, settings
from app.domains.portrait.edit_mask import (
    EditMaskSet,
    SemanticMaskConfig,
    build_semantic_edit_masks,
)
from app.domains.portrait.face_lock import FaceLockConfig
from app.domains.portrait.inpaint_verify import (
    InpaintVerification,
    load_rgb,
    verify_inpaint_output,
)

#: Gap ids that mean "the mask is not trustworthy yet". The collar split one is
#: the exact case the brief calls out: if the old collar cannot be told apart
#: from the neck skin, the mask is not quietly shipped as good.
BLOCKING_GAP_PREFIXES = ("collar_split_unresolved",)

#: Background that leaks into the edit area. A few pixels along a segmented edge
#: are expected; a quarter of the edit area is a mask that would repaint the
#: room around the subject.
MAX_BACKGROUND_SHARE_OF_EDIT_AREA = 0.25


class NativeMaskUnavailable(RuntimeError):
    """The mask could not be built honestly; the job must fail, not degrade."""


@dataclass(frozen=True)
class NativeMaskBuild:
    """The mask that was written, plus the evidence behind it."""

    mask_path: Path
    mask_set: EditMaskSet
    acceptance: dict[str, Any]

    @property
    def stats(self) -> dict[str, Any]:
        return dict(self.mask_set.stats)

    def to_dict(self) -> dict[str, Any]:
        return {
            "maskFile": self.mask_path.name,
            "editablePixels": int(self.mask_set.stats.get("editablePixels", 0)),
            "editableShare": self.mask_set.stats.get("editableShare"),
            "collarSplitApplied": self.mask_set.stats.get("collarSplitApplied"),
            "collarEditablePixels": self.mask_set.stats.get("collarEditablePixels"),
            "neckSkinProtectedPixels": self.mask_set.stats.get(
                "neckSkinProtectedPixels"
            ),
            "identityCoversEditable": self.mask_set.stats.get(
                "identityCoversEditable"
            ),
            "backgroundEditableShare": self.mask_set.stats.get(
                "backgroundEditableShare"
            ),
            "gaps": list(self.mask_set.gaps),
            "acceptance": self.acceptance,
        }


class NativeMaskInpainting:
    """Local mask build + result verification for the masked backend."""

    def __init__(
        self,
        config: Settings = settings,
        *,
        detector_factory: Callable[[Path], Any] | None = None,
        segmenter_factory: Callable[[Path], Any] | None = None,
        mask_config: SemanticMaskConfig | None = None,
        face_lock_config: FaceLockConfig | None = None,
    ) -> None:
        self.config = config
        self._detector_factory = detector_factory
        self._segmenter_factory = segmenter_factory
        self.mask_config = mask_config or SemanticMaskConfig()
        self.face_lock_config = face_lock_config or FaceLockConfig()

    # ------------------------------------------------------------------ build

    def build_mask(self, source_path: Path, mask_path: Path) -> NativeMaskBuild:
        """Parse ``source_path`` and write the black/white edit mask to ``mask_path``."""
        source_path = Path(source_path)
        mask_path = Path(mask_path)
        image = load_rgb(source_path)

        landmarks = self._detect_landmarks(source_path, image)
        labels = self._parse_labels(image)

        try:
            mask_set = build_semantic_edit_masks(
                image,
                landmarks,
                labels,
                config=self.mask_config,
                face_lock_config=self.face_lock_config,
            )
        except (ValueError, RuntimeError) as exc:
            raise NativeMaskUnavailable(
                f"The semantic edit mask could not be built: {exc}"
            ) from exc

        acceptance = self._accept(mask_set, image.shape[:2])
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(mask_set.editable).save(mask_path)
        self._write_mask_metadata(mask_path, mask_set, acceptance)
        return NativeMaskBuild(mask_path=mask_path, mask_set=mask_set, acceptance=acceptance)

    def _detect_landmarks(self, source_path: Path, image: np.ndarray) -> np.ndarray:
        from app.domains.portrait.face_lock import (
            FaceLockUnavailable,
            default_detector,
        )

        model_path = Path(
            getattr(self.config, "face_landmark_model_path", None)
            or self.config.models_dir / "face_landmarker.task"
        )
        if not model_path.is_file():
            raise NativeMaskUnavailable(
                "The face landmark model is missing at "
                f"{model_path}; the identity guard cannot be placed, so no mask "
                "was built and no photo was sent anywhere."
            )
        factory = self._detector_factory or default_detector
        try:
            detector = factory(model_path)
        except FaceLockUnavailable as exc:
            raise NativeMaskUnavailable(str(exc)) from exc
        try:
            faces = detector.detect(image)
        finally:
            close = getattr(detector, "close", None)
            if callable(close):
                close()
        if not faces:
            raise NativeMaskUnavailable(
                f"No face was detected in {source_path.name}; refusing to build an "
                "edit mask that has no identity guard."
            )
        return np.asarray(faces[0], dtype=np.float64)

    def _parse_labels(self, image: np.ndarray) -> np.ndarray:
        from app.domains.portrait.human_parsing import (
            HumanParsingUnavailable,
            SchpOnnxSegmenter,
            default_model_path,
        )

        model_path = Path(
            getattr(self.config, "human_parsing_model_path", None)
            or default_model_path(self.config.models_dir)
        )
        if not model_path.is_file():
            raise NativeMaskUnavailable(
                f"The SCHP ATR human-parsing model is missing at {model_path}; the "
                "hair, clothing and neck regions cannot be parsed, so no mask was "
                "built and no photo was sent anywhere. Install the model "
                "(tools/run_human_parsing.py) or keep using the whole-image "
                "SiliconFlow backend."
            )
        factory = self._segmenter_factory or (lambda path: SchpOnnxSegmenter(path))
        try:
            segmenter = factory(model_path)
        except HumanParsingUnavailable as exc:
            raise NativeMaskUnavailable(str(exc)) from exc
        try:
            labels = segmenter.segment(image)
        except HumanParsingUnavailable as exc:
            raise NativeMaskUnavailable(str(exc)) from exc
        finally:
            close = getattr(segmenter, "close", None)
            if callable(close):
                close()
        return np.asarray(labels)

    # ------------------------------------------------------------- acceptance

    def _accept(self, mask_set: EditMaskSet, shape: tuple[int, int]) -> dict[str, Any]:
        """Refuse a mask that is not safe to hand to a paid generator."""
        stats = mask_set.stats
        height, width = int(shape[0]), int(shape[1])
        editable = np.asarray(mask_set.editable)
        reasons: list[str] = []

        blocking_gaps = [
            gap
            for gap in mask_set.gaps
            if gap.split(":", 1)[0] in BLOCKING_GAP_PREFIXES
        ]
        reasons.extend(blocking_gaps)

        if editable.shape != (height, width):
            reasons.append(
                f"the mask is {editable.shape[1]}x{editable.shape[0]}, not the "
                f"photo's {width}x{height}"
            )
        editable_pixels = int((editable > 0).sum())
        editable_share = editable_pixels / float(max(height * width, 1))
        if editable_pixels < self.mask_config.min_region_px:
            reasons.append(
                f"the edit area holds {editable_pixels} px (< "
                f"{self.mask_config.min_region_px}); there is nothing meaningful "
                "to restyle"
            )
        if editable_share > 0.90:
            reasons.append(
                f"the edit area covers {editable_share:.1%} of the frame, which is "
                "a whole-image edit rather than a masked one"
            )
        if not stats.get("identityCoversEditable", False):
            reasons.append(
                "the edit area overlaps the identity guard (eyes, brows, nose, "
                "lips, teeth): a repaint could reach the face"
            )
        if not stats.get("protectedCoversEditable", False):
            reasons.append("the edit area and the protected region are not disjoint")
        background_share = float(stats.get("backgroundEditableShare", 0.0) or 0.0)
        if background_share > MAX_BACKGROUND_SHARE_OF_EDIT_AREA:
            reasons.append(
                "background makes up "
                f"{background_share:.1%} of the edit area (limit "
                f"{MAX_BACKGROUND_SHARE_OF_EDIT_AREA:.0%}); the mask would let the "
                "generator repaint the room around the subject"
            )

        acceptance = {
            "passed": not reasons,
            "reasons": reasons,
            "editablePixels": editable_pixels,
            "editableShare": round(editable_share, 6),
            "backgroundShareOfEditArea": round(background_share, 6),
            "collarSplitApplied": bool(stats.get("collarSplitApplied", False)),
            "collarEditablePixels": int(stats.get("collarEditablePixels", 0)),
            "neckSkinProtectedPixels": int(stats.get("neckSkinProtectedPixels", 0)),
            "identityCoversEditable": bool(stats.get("identityCoversEditable", False)),
        }
        if reasons:
            raise NativeMaskUnavailable(
                "The semantic edit mask failed its acceptance checks and was not "
                "shipped (" + "; ".join(reasons) + "). Fix the parse or correct the "
                "mask by hand before a paid generation."
            )
        return acceptance

    def _write_mask_metadata(
        self, mask_path: Path, mask_set: EditMaskSet, acceptance: dict[str, Any]
    ) -> None:
        """Keep the mask's provenance beside the provider's own ``_meta`` record."""
        metadata_dir = mask_path.parent / "_meta"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "kind": "native-edit-mask",
            "maskFile": mask_path.name,
            "generatedAt": datetime.now(timezone.utc).isoformat(),
            "semantics": "white=repaint black=keep original pixels",
            "size": [int(mask_set.editable.shape[1]), int(mask_set.editable.shape[0])],
            "stats": mask_set.stats,
            "gaps": list(mask_set.gaps),
            "acceptance": acceptance,
            "note": (
                "built locally from SCHP parsing + MediaPipe landmarks; no "
                "generator ran and no photo left the machine"
            ),
        }
        (metadata_dir / f"{mask_path.name}.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # ------------------------------------------------------------ verification

    def finalise(
        self,
        source_path: Path,
        generated_path: Path,
        mask_path: Path,
        final_path: Path,
        *,
        mask_set: EditMaskSet | None = None,
    ) -> InpaintVerification:
        """Verify the raw provider output and write the kept-pixels-restored final."""
        identity_guard = None if mask_set is None else np.asarray(mask_set.protected)
        return verify_inpaint_output(
            source_path,
            generated_path,
            mask_path,
            restore_path=final_path,
            identity_guard=identity_guard,
            region_of_interest=identity_guard,
        )


__all__ = [
    "MAX_BACKGROUND_SHARE_OF_EDIT_AREA",
    "NativeMaskBuild",
    "NativeMaskInpainting",
    "NativeMaskUnavailable",
]
