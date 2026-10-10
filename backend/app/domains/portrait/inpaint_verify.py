"""Verify a native-mask inpaint result and put the kept pixels back.

A black/white inpaint mask is an *instruction*, not a guarantee. The hosted
models honour it well but not perfectly: an edge can bleed a few pixels into the
black side, a pipeline can return an image at a slightly different size, and a
queue can hand back a picture of something else entirely. The masked-backend
brief is explicit that none of that may reach the user:

* the model's output is checked for size and geometric alignment against the
  original **before** anything is called a success;
* the final picture is composed so that every pixel the mask kept is the
  *original* pixel, byte for byte;
* the identity features (eyes, brows, nose, lips, teeth) must be inside the kept
  region, so no repainted eye can ever be the final output;
* the original features are never reprojected -- this module only copies pixels
  from the original at their own coordinates, it never warps or rescales them;
* when the result cannot be registered reliably the run is *rejected* instead of
  being reported as good.

Three statuses are reported and deliberately never merged, because they answer
three different questions:

``MASK_VALID``
    Is the mask itself usable -- photo coordinates, strict black/white, opens
    something but not the whole frame, and never overlaps the identity guard?
``PIXEL_IDENTITY_PASSED``
    Are the final image's kept pixels the original pixels, in a result that
    registered against the original?
``VISUAL_ACCEPTANCE_PENDING``
    Has a human looked at the repainted hair/clothing? Always ``True`` from this
    module: the machine can prove identity, only an eye can accept a hairstyle.

Nothing here generates, trains or calls a provider.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

#: A mask that opens less than this share of the frame asks for nothing; one
#: that opens more than the remainder is a whole-image edit wearing a mask's
#: clothes. The provider's pre-flight cost guard uses the same bounds.
MIN_EDITABLE_SHARE = 0.001
MAX_EDITABLE_SHARE = 0.90

#: Mean absolute difference (0-255) allowed between the original and the model's
#: output over the *kept* region, per channel. Above this the output has drifted
#: enough that it is worth refusing even though the composite would hide it --
#: a large drift means the geometry moved, which is exactly what must not be
#: silently papered over.
DEFAULT_MAX_KEPT_MEAN_ABS_DIFF = 24.0

#: The alignment probe searches shifts in ``[-N, N]`` pixels on each axis.
DEFAULT_MAX_SHIFT_PX = 2

#: If offsetting the output reduces the kept-region residual to below this share
#: of the zero-shift residual, the output is systematically shifted and the run
#: is rejected rather than composited.
SHIFT_IMPROVEMENT_RATIO = 0.60


class InpaintVerificationError(RuntimeError):
    """The result cannot be verified at all -- refuse to call it a success."""


@dataclass(frozen=True)
class MaskReview:
    """Structural verdict on the black/white edit mask."""

    valid: bool
    reasons: tuple[str, ...]
    width: int
    height: int
    editable_pixels: int
    editable_share: float
    protected_pixels: int
    identity_guard_overlap: int
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "reasons": list(self.reasons),
            "width": self.width,
            "height": self.height,
            "editablePixels": self.editable_pixels,
            "editableShare": self.editable_share,
            "protectedPixels": self.protected_pixels,
            "identityGuardOverlap": self.identity_guard_overlap,
            "sha256": self.sha256,
            "semantics": "white=repaint black=keep original pixels",
        }


@dataclass(frozen=True)
class InpaintVerification:
    """The three independent statuses plus the evidence behind them."""

    mask: MaskReview
    aligned: bool
    kept_pixels: int
    kept_mean_abs_diff: float
    best_shift: tuple[int, int]
    kept_mean_abs_diff_at_best_shift: float
    restored_path: Path | None = None
    reasons: tuple[str, ...] = ()
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def mask_valid(self) -> bool:
        return self.mask.valid

    @property
    def pixel_identity_passed(self) -> bool:
        """Kept pixels are the original's, in a result that registered."""
        return bool(self.mask.valid and self.aligned)

    @property
    def visual_acceptance_pending(self) -> bool:
        """Always pending: a human still has to look at the repainted region."""
        return True

    @property
    def ok(self) -> bool:
        """Machine gate only; ``visual_acceptance_pending`` stays ``True``."""
        return self.pixel_identity_passed

    @property
    def statuses(self) -> dict[str, bool]:
        return {
            "MASK_VALID": self.mask_valid,
            "PIXEL_IDENTITY_PASSED": self.pixel_identity_passed,
            "VISUAL_ACCEPTANCE_PENDING": self.visual_acceptance_pending,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "statuses": self.statuses,
            "maskValid": self.mask_valid,
            "pixelIdentityPassed": self.pixel_identity_passed,
            "visualAcceptancePending": self.visual_acceptance_pending,
            "mask": self.mask.to_dict(),
            "aligned": self.aligned,
            "keptPixels": self.kept_pixels,
            "keptMeanAbsDiff": self.kept_mean_abs_diff,
            "keptMeanAbsDiffAtBestShift": self.kept_mean_abs_diff_at_best_shift,
            "bestShift": list(self.best_shift),
            "restoredPath": self.restored_path.name if self.restored_path else None,
            "reasons": list(self.reasons),
            "diagnostics": self.diagnostics,
            "note": (
                "MASK_VALID, PIXEL_IDENTITY_PASSED and VISUAL_ACCEPTANCE_PENDING "
                "are independent: the kept pixels are provably the original's, "
                "the repainted hair/clothing still needs a human eye."
            ),
        }


# --------------------------------------------------------------------- loading


def load_rgb(path: Path | str) -> np.ndarray:
    """Read a picture as an HxWx3 uint8 RGB array, or raise."""
    try:
        with Image.open(path) as image:
            return np.asarray(image.convert("RGB"))
    except (UnidentifiedImageError, OSError) as exc:
        raise InpaintVerificationError(f"Could not decode image: {Path(path).name}.") from exc


def load_mask(mask: np.ndarray | Path | str) -> tuple[np.ndarray, str]:
    """Normalise a mask argument to a 2-D uint8 array plus its sha256."""
    if isinstance(mask, (str, Path)):
        path = Path(mask)
        try:
            with Image.open(path) as image:
                image.load()
                array = np.asarray(image.convert("L"), dtype=np.uint8)
        except (UnidentifiedImageError, OSError) as exc:
            raise InpaintVerificationError(
                f"Could not decode edit mask: {path.name}."
            ) from exc
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return array, digest
    array = np.asarray(mask)
    if array.ndim == 3:
        array = array[..., 0]
    array = array.astype(np.uint8, copy=False)
    return array, hashlib.sha256(array.tobytes()).hexdigest()


# ----------------------------------------------------------------- mask review


def review_mask(
    mask: np.ndarray | Path | str,
    size: tuple[int, int],
    *,
    identity_guard: np.ndarray | None = None,
    min_editable_share: float = MIN_EDITABLE_SHARE,
    max_editable_share: float = MAX_EDITABLE_SHARE,
) -> MaskReview:
    """Judge whether ``mask`` can be trusted as an inpainting instruction."""
    width, height = int(size[0]), int(size[1])
    array, digest = load_mask(mask)
    reasons: list[str] = []

    if array.shape != (height, width):
        reasons.append(
            "mask is not in the photo's own coordinates: "
            f"mask {array.shape[1]}x{array.shape[0]} vs photo {width}x{height}"
        )
        return MaskReview(
            valid=False,
            reasons=tuple(reasons),
            width=int(array.shape[1]),
            height=int(array.shape[0]),
            editable_pixels=0,
            editable_share=0.0,
            protected_pixels=0,
            identity_guard_overlap=0,
            sha256=digest,
        )

    levels = np.unique(array)
    if not np.all(np.isin(levels, (0, 255))):
        reasons.append(
            "mask must be strictly black/white (0 = keep, 255 = repaint); found "
            f"{int(levels.size)} grey level(s)"
        )

    editable = array == 255
    total = int(array.size)
    editable_pixels = int(editable.sum())
    editable_share = editable_pixels / float(max(total, 1))
    if editable_share < min_editable_share:
        reasons.append(
            f"mask opens almost nothing to repaint ({editable_share:.4%} editable)"
        )
    if editable_share > max_editable_share:
        reasons.append(
            "mask opens almost the whole frame "
            f"({editable_share:.1%} editable) -- that is a whole-image edit"
        )

    identity_guard_overlap = 0
    if identity_guard is not None:
        guard = np.asarray(identity_guard)
        if guard.shape != array.shape:
            raise InpaintVerificationError(
                "identity guard is not in the photo's own coordinates: "
                f"{guard.shape} vs {array.shape}"
            )
        identity_guard_overlap = int(np.logical_and(editable, guard > 0).sum())
        if identity_guard_overlap > 0:
            reasons.append(
                "the editable region overlaps the protected facial features by "
                f"{identity_guard_overlap} px; a repaint could reach an eye, a "
                "nose, a mouth or the teeth"
            )

    return MaskReview(
        valid=not reasons,
        reasons=tuple(reasons),
        width=width,
        height=height,
        editable_pixels=editable_pixels,
        editable_share=round(editable_share, 6),
        protected_pixels=total - editable_pixels,
        identity_guard_overlap=identity_guard_overlap,
        sha256=digest,
    )


# ------------------------------------------------------------------ alignment


def _mean_abs_diff(generated: np.ndarray, original: np.ndarray, region: Any) -> float:
    if region is None or not region.any():
        return 0.0
    difference = np.abs(
        generated.astype(np.int16) - original.astype(np.int16)
    ).astype(np.float32)
    return float(difference[region].mean())


def best_alignment_shift(
    generated: np.ndarray,
    original: np.ndarray,
    region: Any,
    *,
    max_shift: int = DEFAULT_MAX_SHIFT_PX,
) -> tuple[tuple[int, int], float, float]:
    """Search integer shifts of ``generated`` in ``[-max_shift, max_shift]``.

    Returns ``(best_shift, best_residual, zero_shift_residual)``. The residual is
    the mean absolute difference over ``region`` (the kept pixels) at that shift.
    A result that is *systematically* offset scores far better once re-registered,
    which is how a moved or rescaled output is caught before it is composited.
    """
    height, width = original.shape[:2]
    if not region.any():
        return (0, 0), 0.0, 0.0

    best_shift = (0, 0)
    best_residual = None
    zero_residual = None
    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            # Compare original[y, x] with generated[y - dy, x - dx].
            y0, y1 = max(0, dy), min(height, height + dy)
            x0, x1 = max(0, dx), min(width, width + dx)
            if y0 >= y1 or x0 >= x1:
                continue
            original_slice = original[y0:y1, x0:x1]
            generated_slice = generated[y0 - dy : y1 - dy, x0 - dx : x1 - dx]
            region_slice = region[y0:y1, x0:x1]
            residual = _mean_abs_diff(generated_slice, original_slice, region_slice)
            if zero_residual is None and (dy, dx) == (0, 0):
                zero_residual = residual
            if best_residual is None or residual < best_residual:
                best_residual = residual
                best_shift = (dy, dx)
    return best_shift, float(best_residual or 0.0), float(zero_residual or 0.0)


# ----------------------------------------------------------------- compositing


def restore_kept_pixels(
    original_rgb: np.ndarray,
    generated_rgb: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Return the final picture: generated where white, **original** where black.

    This is the only place the two images are combined, and it is a straight
    pixel copy at identical coordinates. There is deliberately no warp, no
    rescale and no alignment transform: the brief forbids "correcting" the
    original features with a non-rigid deformation, and a rigid one would move
    them just as surely.
    """
    original = np.asarray(original_rgb)
    generated = np.asarray(generated_rgb)
    if original.shape != generated.shape:
        raise InpaintVerificationError(
            "the model output does not match the original's size ("
            f"{original.shape[1]}x{original.shape[0]} vs "
            f"{generated.shape[1]}x{generated.shape[0]}); it cannot be registered"
        )
    array, _ = load_mask(mask)
    if array.shape != original.shape[:2]:
        raise InpaintVerificationError(
            "the edit mask is not in the photo's own coordinates; the kept pixels "
            "cannot be restored"
        )
    editable = (array == 255)[..., None]
    return np.where(editable, generated, original).astype(np.uint8)


# ---------------------------------------------------------------- verification


def verify_inpaint_output(
    original_path: Path | str,
    generated_path: Path | str,
    mask: np.ndarray | Path | str,
    *,
    restore_path: Path | str | None = None,
    identity_guard: np.ndarray | None = None,
    region_of_interest: Any | None = None,
    max_kept_mean_abs_diff: float = DEFAULT_MAX_KEPT_MEAN_ABS_DIFF,
    max_shift: int = DEFAULT_MAX_SHIFT_PX,
) -> InpaintVerification:
    """Check the model's raw output, then write the kept-pixels-restored final.

    ``region_of_interest`` narrows the alignment probe to the person (the identity
    guard grown to the face island when the caller has one); when it is omitted
    the whole kept region is used.
    """
    original = load_rgb(original_path)
    generated = load_rgb(generated_path)
    if original.shape != generated.shape:
        raise InpaintVerificationError(
            "the model output does not match the original's size ("
            f"{original.shape[1]}x{original.shape[0]} vs "
            f"{generated.shape[1]}x{generated.shape[0]}); refusing to report a "
            "success that cannot be registered"
        )

    review = review_mask(
        mask, (original.shape[1], original.shape[0]), identity_guard=identity_guard
    )
    kept = review_protected_region(mask, original.shape[:2])
    probe = kept if region_of_interest is None else np.logical_and(kept, np.asarray(region_of_interest) > 0)

    shift, best_residual, zero_residual = best_alignment_shift(
        generated, original, probe, max_shift=max_shift
    )
    reasons: list[str] = []
    aligned = True
    if shift != (0, 0) and zero_residual > 0 and best_residual < SHIFT_IMPROVEMENT_RATIO * zero_residual:
        aligned = False
        reasons.append(
            "the output is systematically shifted by "
            f"({shift[0]}, {shift[1]}) px relative to the original; the framing "
            "and geometry do not match"
        )
    if zero_residual > max_kept_mean_abs_diff:
        aligned = False
        reasons.append(
            "the kept region differs from the original by "
            f"{zero_residual:.1f}/255 on average (limit "
            f"{max_kept_mean_abs_diff:.1f}); the model repainted pixels the mask "
            "kept"
        )

    if not review.valid:
        reasons.extend(review.reasons)

    validated = review.valid and aligned
    restored_path = None
    kept_pixels = int(kept.sum())
    if validated:
        final = restore_kept_pixels(original, generated, mask)
        if restore_path is not None:
            restored_path = Path(restore_path)
            restored_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(final).save(restored_path)

    return InpaintVerification(
        mask=review,
        aligned=aligned,
        kept_pixels=kept_pixels,
        kept_mean_abs_diff=round(zero_residual, 4),
        best_shift=shift,
        kept_mean_abs_diff_at_best_shift=round(best_residual, 4),
        restored_path=restored_path,
        reasons=tuple(reasons),
        diagnostics={
            "maxKeptMeanAbsDiff": max_kept_mean_abs_diff,
            "maxShiftPx": max_shift,
            "keptRegionSource": "mask==0",
            "probeIsIdentityGuard": region_of_interest is not None,
            "probePixels": int(probe.sum()),
            "restored": validated,
            "note": (
                "final image = original pixels where the mask is black, model "
                "pixels where it is white; no warp, no rescale, no reprojection"
            ),
        },
    )


def review_protected_region(mask: np.ndarray | Path | str, shape: tuple[int, int]) -> np.ndarray:
    """Boolean map of the pixels the mask keeps (black = keep)."""
    array, _ = load_mask(mask)
    if array.shape != tuple(shape):
        raise InpaintVerificationError(
            "the edit mask is not in the photo's own coordinates"
        )
    return array == 0


__all__ = [
    "DEFAULT_MAX_KEPT_MEAN_ABS_DIFF",
    "DEFAULT_MAX_SHIFT_PX",
    "InpaintVerification",
    "InpaintVerificationError",
    "MaskReview",
    "best_alignment_shift",
    "load_mask",
    "load_rgb",
    "restore_kept_pixels",
    "review_mask",
    "review_protected_region",
    "verify_inpaint_output",
]
