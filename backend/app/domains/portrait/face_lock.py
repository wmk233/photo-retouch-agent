"""Strict face identity lock: keep the *original* facial pixels in a styled photo.

The styling pipeline (SiliconFlow ``Qwen/Qwen-Image-Edit-2509``) happily repaints
the whole person, which is why the P2.3 acceptance image changed the eyes, smile
and jaw even though the prompt asked for identity preservation. A prompt is not a
guarantee, so this module enforces the guarantee *after* generation:

    original face landmarks
        -> original face protection mask
        -> original face content
        -> generated image
        -> localised restore + compositing
        -> verification
        -> final output

Only non-generative operations are used: face landmarks come from the pretrained
``google-ai-edge/mediapipe`` Face Landmarker, and the restore is a similarity
transform plus an alpha composite. Nothing here redraws, upscales or "beautifies"
the face: inside the protected region the output is the original photograph,
resampled.

Hard rules encoded below:

* the eyes, nose, mouth, teeth, cheeks, chin and the outer face contour come from
  the original photo -- the generated repaint never shows through;
* the original *expression* (including an open, toothy smile) is preserved
  because the mouth pixels are original;
* the person count must not change;
* a registration that would need to stretch or distort the original features is
  reported as a failure (``face_lock_alignment_failed`` /
  ``face_lock_coverage_failed`` / ``face_lock_seam_unsafe``) instead of being
  force-fitted into a "succeeded" output;
* nothing here calls a paid model.

The module is deliberately free of ``app.services`` / ``app.providers`` /
``app.brains`` imports so it can stay in the domain layer.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, Protocol, Sequence

try:  # numpy is only needed for the actual lock, not for importing the module
    import numpy as np
except Exception:  # pragma: no cover - exercised only on stripped installs
    np = None  # type: ignore[assignment]

try:
    import cv2
except Exception:  # pragma: no cover - exercised only on stripped installs
    cv2 = None  # type: ignore[assignment]


FACE_LOCK_OFF = "off"
FACE_LOCK_STRICT = "strict"
FACE_LOCK_MODES = (FACE_LOCK_OFF, FACE_LOCK_STRICT)
DEFAULT_FACE_LOCK_MODE = FACE_LOCK_STRICT

#: Filename (relative to the outputs directory) of the model used for landmarks.
FACE_LANDMARK_MODEL_FILENAME = "face_landmarker.task"


class FaceLockUnavailable(RuntimeError):
    """Raised when the landmark stack (mediapipe/numpy/opencv) is unusable."""


class FaceLockStatus(str, Enum):
    APPLIED = "applied"
    DISABLED = "disabled"
    UNAVAILABLE = "face_lock_unavailable"
    NO_FACE = "face_lock_no_face"
    MULTIPLE_FACES = "face_lock_multiple_faces"
    ALIGNMENT_FAILED = "face_lock_alignment_failed"
    COVERAGE_FAILED = "face_lock_coverage_failed"
    SEAM_UNSAFE = "face_lock_seam_unsafe"
    IDENTITY_CHECK_FAILED = "face_lock_identity_check_failed"
    RENDER_ERROR = "face_lock_render_error"


#: Statuses that must never be reported as a successful styling output.
FAILURE_STATUSES = frozenset(
    {
        FaceLockStatus.UNAVAILABLE,
        FaceLockStatus.NO_FACE,
        FaceLockStatus.MULTIPLE_FACES,
        FaceLockStatus.ALIGNMENT_FAILED,
        FaceLockStatus.COVERAGE_FAILED,
        FaceLockStatus.SEAM_UNSAFE,
        FaceLockStatus.IDENTITY_CHECK_FAILED,
        FaceLockStatus.RENDER_ERROR,
    }
)


# --------------------------------------------------------------- landmark sets
# MediaPipe Face Landmarker indices (478-point face mesh).

FACE_OVAL = (
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379,
    378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
    162, 21, 54, 103, 67, 109,
)
EYE_A = (33, 133, 159, 145, 7, 163, 144, 153, 154, 155, 246, 161, 160, 158, 157, 173)
EYE_B = (263, 362, 386, 374, 249, 390, 373, 380, 381, 382, 466, 388, 387, 385, 384, 398)
BROW_A = (70, 63, 105, 66, 107, 55, 65, 52, 53, 46)
BROW_B = (300, 293, 334, 296, 336, 285, 295, 282, 283, 276)
LIPS_OUTER = (
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317,
    14, 87, 178, 88, 95, 78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308,
)
LIPS_INNER = (78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191)
NOSE = (168, 6, 197, 195, 5, 4, 1, 2, 98, 327, 326, 294, 278, 344, 440, 275, 45, 220, 115, 48, 131, 134, 236, 198, 49, 279, 429, 358)
FACE_SIDES = (234, 454, 93, 323, 127, 356)

#: Landmarks used to *fit* the similarity transform. Only rigid, non-deformable
#: anatomy (eye corners, nose bridge/tip/nostrils, face sides, forehead) is used;
#: the mouth and chin are deliberately excluded so a mouth the model redrew
#: smaller cannot pull the registration off the rest of the face.
RIGID_FIT = (33, 133, 362, 263, 168, 6, 1, 98, 327, 234, 454, 93, 323, 10)

MOUTH_LEFT = 61
MOUTH_RIGHT = 291
NOSE_TIP = 1
CHIN = 152
EYE_CORNERS = (33, 133, 362, 263)

#: Named regions reported in the diagnostics so a human can see what was locked.
LOCKED_REGIONS = {
    "eyes": EYE_A + EYE_B,
    "eyebrows": BROW_A + BROW_B,
    "nose": NOSE,
    "mouth_and_teeth": LIPS_OUTER + LIPS_INNER,
    "jaw_and_chin": (172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397),
    "cheeks": (50, 101, 118, 117, 123, 147, 187, 205, 206, 216, 212, 202, 280, 330, 347, 346, 352, 376, 411, 425, 426, 436, 432, 422, 234, 454),
    "outer_contour": FACE_OVAL,
}


@dataclass(frozen=True)
class FaceLockConfig:
    """Thresholds for the strict lock. All ratios are relative to face width."""

    mode: str = DEFAULT_FACE_LOCK_MODE

    # Mask shaping. The paste region is the hull of *both* face outlines (so no
    # leftover generated facial feature can peek out) grown by ``paste_dilate``,
    # then feathered. Measured on the real P2.3 job: a 0.018 feather leaves a
    # visible contour, 0.000 leaves the generated jaw/forehead skin sticking out
    # around the original outline; 0.030/0.035 is the compromise that keeps the
    # introduced edge lowest.
    paste_dilate_ratio: float = 0.030
    feather_ratio: float = 0.035

    # Registration gates (fail the job rather than distort the original).
    max_scale_deviation: float = 0.10
    max_rotation_deg: float = 8.0
    #: Hard gate. Above this the generated head is in a *different* geometry
    #: (turned, re-posed, re-proportioned), so pasting the original face would
    #: leave leftover generated facial features around it. The restore itself is
    #: a similarity transform, so "stretching the original features to fit" is
    #: impossible by construction -- this gate rejects the *scene*, not the fit.
    max_rigid_residual_ratio: float = 0.045
    #: Reporting-only budget. A generated face that drifts past this is recorded
    #: as "the model rewrote the face non-rigidly" (``strictResidualPassed``)
    #: instead of being silently advertised as a faithful render.
    strict_rigid_residual_ratio: float = 0.030
    max_coverage_ratio: float = 0.070

    # Compositing + verification gates. ``max_seam_excess_p95`` limits the edge
    # the blend may *introduce* (present in the output but in neither source);
    # see the verification block in ``_lock_and_render``.
    max_seam_excess_p95: float = 24.0
    core_identity_mean_tol: float = 2.5
    core_identity_max_tol: float = 24.0

    # Skin-adaptive refinement of the protection region. The protection mask is
    # derived from the *face outline*, and on a real portrait that hull also
    # covers the hair the subject had in the original photo (the temples, the
    # fringe above the forehead). Pasting those pixels untouched prints the old
    # hairstyle over the new one -- the "sticker" band along the hairline. The
    # refinement keeps only pixels that plausibly belong to the *face*: the
    # subject's own skin (measured against their own cheek/forehead tone, so a
    # dark fringe is not mistaken for a warm shadow) plus the hull of the
    # facial features, which is what carries identity.
    skin_refine: bool = True
    #: Luma floor as a fraction of the reference skin luma. Pixels darker than
    #: this are hair, brows, glasses or shadow, never the skin to protect.
    skin_y_ratio: float = 0.65
    #: Chroma tolerance (YCrCb, 0-255 scale) around the reference skin tone.
    skin_chroma_tolerance: float = 16.0
    skin_close_ratio: float = 0.020
    skin_open_ratio: float = 0.012
    #: Grow the detected skin so thin occluders crossing it (glasses wire, the
    #: lip line, specular highlights) are covered rather than punched out.
    skin_grow_ratio: float = 0.030
    #: The binary skin/feature union has a ragged, notched outline (morphology
    #: on a noisy test). Compositing against that outline prints hard, angular
    #: steps along the hairline. Blurring the union and re-thresholding turns it
    #: back into the smooth curve a hairline actually is.
    skin_smooth_ratio: float = 0.018
    #: The skin/feature union above is the *content* boundary: inside it the
    #: photograph is pasted back, outside it the render stands. Pinning alpha to
    #: 1 right up to that boundary would print its ragged edge (the notches left
    #: by morphology on textured hair) as a hard sawtooth along the hairline.
    #: The bit-exact core -- the region that must be the original photograph to
    #: the byte -- is therefore shaved back from that boundary by this fraction
    #: of the face width, which leaves the existing feather free to carry the
    #: transition. Identity is unaffected: the features are protected by their
    #: own guard well inside this margin.
    core_erode_ratio: float = 0.035
    #: The feature hull is always protected, whatever the skin test says.
    feature_guard_ratio: float = 0.020
    #: Safety floor. If the refinement would keep less than this share of the
    #: geometric face hull it is treated as unreliable (degenerate or stylised
    #: input) and the unrefined geometric mask is used instead.
    min_core_coverage: float = 0.80

    # Non-generative tone harmonisation of the *generated* pixels. The styling
    # model renders a brighter, cooler face than the photograph, so pasting the
    # original face leaves a low-frequency tone step where it meets the new
    # skin. This is a smooth additive field fitted on skin-like pixels around
    # the protection boundary and applied to the generated image BEFORE
    # compositing -- the protected core is written at alpha 1, so its pixels are
    # bit-for-bit the original whatever this does.
    harmonise: bool = True
    harmonise_sigma_ratio: float = 0.080
    harmonise_band_ratio: float = 0.180
    harmonise_clamp: float = 26.0
    #: The correction is confined to the render's *skin*. Darkening the new
    #: hairstyle towards the photograph's older, dimmer lighting would (a) undo
    #: the colour the stylist just designed and (b) *add* a step at the hairline,
    #: where the pasted original skin meets the untouched new hair. The render's
    #: skin sits ~25 luma from the photograph's, so the test that classifies it
    #: is deliberately looser than the protection test.
    harmonise_skin_tolerance_scale: float = 2.0


def config_from_settings(settings: Any) -> FaceLockConfig:
    """Build a :class:`FaceLockConfig` from the application settings object.

    Duck-typed on purpose: the domain layer must not import ``app.core``.
    """
    defaults = FaceLockConfig()
    return FaceLockConfig(
        mode=str(getattr(settings, "face_lock_mode", defaults.mode) or defaults.mode).lower(),
        max_rigid_residual_ratio=float(
            getattr(settings, "face_lock_residual_ratio", defaults.max_rigid_residual_ratio)
        ),
        strict_rigid_residual_ratio=float(
            getattr(
                settings,
                "face_lock_strict_residual_ratio",
                defaults.strict_rigid_residual_ratio,
            )
        ),
    )


@dataclass
class FaceLockReport:
    """Outcome of one face lock pass."""

    status: FaceLockStatus
    applied: bool
    message: str
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.applied and self.status is FaceLockStatus.APPLIED

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "applied": self.applied,
            "ok": self.ok,
            "message": self.message,
            "diagnostics": self.diagnostics,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2, default=str)


class LandmarkDetector(Protocol):
    """Anything that can return one (N, 2) pixel-space landmark array per face."""

    def detect(self, image_rgb: Any) -> list[Any]:
        ...


# ------------------------------------------------------------------ detection


class MediaPipeLandmarkDetector:
    """Thin wrapper around the pretrained MediaPipe Face Landmarker task."""

    def __init__(self, model_path: Path | str, min_face_detection_confidence: float = 0.4) -> None:
        if np is None:  # pragma: no cover - defensive
            raise FaceLockUnavailable("numpy is required for face landmarks.")
        try:
            import mediapipe as mp  # local import: keeps module import cheap
        except Exception as exc:  # pragma: no cover - defensive
            raise FaceLockUnavailable(f"mediapipe is not installed: {exc}") from exc

        model_file = Path(model_path)
        if not model_file.is_file():
            raise FaceLockUnavailable(
                f"Face landmark model not found at {model_file}. "
                "Run backend/tools/download_face_landmarker.py once, or set "
                "PHOTO_AGENT_FACE_LANDMARK_MODEL."
            )

        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_file)),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=4,
            min_face_detection_confidence=min_face_detection_confidence,
        )
        self._landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)

    def detect(self, image_rgb: Any) -> list[Any]:
        import mediapipe as mp

        height, width = image_rgb.shape[:2]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(image_rgb))
        result = self._landmarker.detect(mp_image)
        faces: list[Any] = []
        for landmarks in result.face_landmarks:
            points = np.array(
                [[point.x * width, point.y * height] for point in landmarks],
                dtype=np.float64,
            )
            faces.append(points)
        return faces

    def close(self) -> None:  # pragma: no cover - best effort cleanup
        try:
            self._landmarker.close()
        except Exception:
            pass


def default_detector(model_path: Path | str) -> LandmarkDetector:
    """Build the production detector, or raise :class:`FaceLockUnavailable`."""
    return MediaPipeLandmarkDetector(model_path)


# ------------------------------------------------------------------- geometry


def _require_stack() -> None:
    if np is None or cv2 is None:  # pragma: no cover - defensive
        raise FaceLockUnavailable(
            "Face lock needs numpy and opencv (installed with mediapipe)."
        )


def face_width(landmarks: Any) -> float:
    left = landmarks[FACE_SIDES[0]]
    right = landmarks[FACE_SIDES[1]]
    return float(np.linalg.norm(left - right))


def similarity_transform(source: Any, target: Any) -> Any:
    """Least-median similarity (rotation + uniform scale + translation)."""
    matrix, _ = cv2.estimateAffinePartial2D(
        np.asarray(source, dtype=np.float32),
        np.asarray(target, dtype=np.float32),
        method=cv2.LMEDS,
    )
    if matrix is None:
        raise FaceLockUnavailable("Could not estimate a face registration transform.")
    return matrix.astype(np.float64)


def apply_transform(matrix: Any, points: Any) -> Any:
    pts = np.asarray(points, dtype=np.float64)
    return pts @ matrix[:, :2].T + matrix[:, 2]


def transform_decomposition(matrix: Any) -> tuple[float, float]:
    """Return ``(scale, rotation_degrees)`` of a similarity matrix."""
    scale = float(math.hypot(matrix[0, 0], matrix[1, 0]))
    rotation = float(math.degrees(math.atan2(matrix[1, 0], matrix[0, 0])))
    return scale, rotation


def invert_similarity(matrix: Any) -> Any:
    """Invert a 2x3 similarity matrix (linear block is rotation x uniform scale)."""
    scale, _ = transform_decomposition(matrix)
    inverse_linear = (1.0 / (scale * scale)) * np.array(
        [
            [matrix[0, 0], matrix[1, 0]],
            [matrix[0, 1], matrix[1, 1]],
        ],
        dtype=np.float64,
    )
    inverse_translation = -inverse_linear @ matrix[:, 2]
    return np.hstack([inverse_linear, inverse_translation.reshape(2, 1)])


def polygon_distance_outside(point: Any, polygon: Any) -> float:
    """Signed-outside distance of ``point`` from ``polygon`` (0 when inside)."""
    ring = np.asarray(polygon, dtype=np.float32).reshape(-1, 1, 2)
    signed = cv2.pointPolygonTest(ring, (float(point[0]), float(point[1])), True)
    return 0.0 if signed >= 0 else float(-signed)


def _convex_hull(points: Any) -> Any:
    return cv2.convexHull(np.asarray(points, dtype=np.float32))


def fill_polygon(shape: tuple[int, int], polygon: Any, value: int = 255) -> Any:
    mask = np.zeros((shape[0], shape[1]), dtype=np.uint8)
    ring = np.asarray(polygon, dtype=np.int32).reshape(-1, 1, 2)
    cv2.fillConvexPoly(mask, ring, value)
    return mask


def _disk_kernel(radius_px: float) -> Any:
    size = max(int(round(radius_px)), 0) * 2 + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def dilate_mask(mask: Any, radius_px: float) -> Any:
    if radius_px <= 0:
        return mask
    return cv2.dilate(mask, _disk_kernel(radius_px))


def erode_mask(mask: Any, radius_px: float) -> Any:
    if radius_px <= 0:
        return mask
    return cv2.erode(mask, _disk_kernel(radius_px))


def _ycc(image_rgb: Any) -> tuple[Any, Any, Any]:
    """YCbCr planes of an RGB image (8-bit coding range, 0-255)."""
    f = np.asarray(image_rgb, dtype=np.float32)
    luma = 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]
    cr = 0.5 * f[..., 0] - 0.419 * f[..., 1] - 0.081 * f[..., 2] + 128.0
    cb = -0.169 * f[..., 0] - 0.331 * f[..., 1] + 0.5 * f[..., 2] + 128.0
    return luma, cr, cb


#: The features that carry identity, each protected by its *own* outline.
FEATURE_GROUPS = (EYE_A, EYE_B, BROW_A, BROW_B, NOSE, LIPS_OUTER)


def _feature_outlines(landmarks: Any) -> list[Any]:
    """One tight outline per facial feature.

    A single hull over *all* of the features bulges across the forehead and the
    temples -- precisely where the subject's previous fringe lives -- so pasting
    that bulge back prints a hair-shaped wedge on the new hairline. Tight
    outlines cover the features and nothing else; the skin around and between
    them is handled by the skin test, which knows hair is not skin.
    """
    return [_convex_hull(landmarks[list(group)]) for group in FEATURE_GROUPS]


def fill_feature_guard(shape: tuple[int, int], landmarks: Any) -> Any:
    """Union of the identity-carrying features (eyes, brows, nose, lips)."""
    mask = np.zeros(shape, np.uint8)
    for outline in _feature_outlines(landmarks):
        mask = np.maximum(mask, fill_polygon(shape, outline))
    return mask


def skin_reference_colour(
    original_rgb: Any,
    landmarks: Any,
    face_width_px: float,
    *,
    erode_ratio: float = 0.10,
    feature_ratio: float = 0.035,
) -> tuple[Any | None, int]:
    """Median skin colour of *this* subject, measured from their own face.

    Sampling the eroded face oval minus the feature hull means the reference is
    always the person in front of us, so the skin test below adapts to their
    complexion and to the photo's white balance instead of assuming a fixed
    "Caucasian portrait" skin range. Returns ``(None, 0)`` when the face is too
    small or too stylised to sample.
    """
    shape = (int(original_rgb.shape[0]), int(original_rgb.shape[1]))
    oval = fill_polygon(shape, landmarks[list(FACE_OVAL)])
    features = dilate_mask(
        fill_feature_guard(shape, landmarks), feature_ratio * face_width_px
    )
    sample = cv2.bitwise_and(
        erode_mask(oval, erode_ratio * face_width_px), cv2.bitwise_not(features)
    )
    pixels = np.asarray(original_rgb)[sample > 0]
    if pixels.shape[0] < 200:
        return None, int(pixels.shape[0])
    return np.median(pixels.astype(np.float32), axis=0), int(pixels.shape[0])


def adaptive_skin_mask(
    image_rgb: Any, reference: Any, face_width_px: float, config: FaceLockConfig
) -> Any:
    """Skin pixels of ``image_rgb`` judged against ``reference`` (its own face)."""
    luma, cr, cb = _ycc(image_rgb)
    ref_luma, ref_cr, ref_cb = _ycc(np.asarray(reference, dtype=np.float32).reshape(1, 1, 3))
    ref_luma, ref_cr, ref_cb = float(ref_luma[0, 0]), float(ref_cr[0, 0]), float(ref_cb[0, 0])
    tolerance = config.skin_chroma_tolerance
    mask = (
        (luma > config.skin_y_ratio * ref_luma)
        & (np.abs(cr - ref_cr) <= tolerance)
        & (np.abs(cb - ref_cb) <= tolerance)
    ).astype(np.uint8) * 255
    if config.skin_close_ratio > 0:
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, _disk_kernel(config.skin_close_ratio * face_width_px)
        )
    if config.skin_open_ratio > 0:
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN, _disk_kernel(config.skin_open_ratio * face_width_px)
        )
    return mask


def generated_skin_weight(
    generated_rgb: Any, reference: Any, face_width_px: float, config: FaceLockConfig
) -> Any:
    """Soft 0-1 map of the *render's* skin, used to confine the tone correction.

    The render's face sits roughly 25 luma from the photograph's, so the
    protection test's chroma tolerance is widened here (the two are the same
    person under different lighting, not a different complexion). Blurring the
    result keeps the correction's edge soft instead of printing a new contour
    where the skin mask stops.
    """
    relaxed = replace(
        config,
        skin_chroma_tolerance=config.skin_chroma_tolerance
        * max(config.harmonise_skin_tolerance_scale, 1.0),
    )
    mask = adaptive_skin_mask(generated_rgb, reference, face_width_px, relaxed)
    sigma = max(0.02 * face_width_px, 0.5)
    soft = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (0, 0), sigmaX=sigma)
    return np.clip(soft, 0.0, 1.0)


def protection_keep_mask(
    original_rgb: Any,
    landmarks: Any,
    face_width_px: float,
    reference: Any,
    config: FaceLockConfig,
) -> tuple[Any | None, dict[str, Any]]:
    """Pixels of the original photo that may be pasted back over the render.

    ``max(own_skin, feature_hull)``: the face the subject actually has, not the
    hair they used to have. Returns ``(None, info)`` when the skin reference is
    unusable so the caller can keep the purely geometric mask.
    """
    info: dict[str, Any] = {"applied": False}
    if reference is None:
        info["reason"] = "no reliable skin reference"
        return None, info
    shape = (int(original_rgb.shape[0]), int(original_rgb.shape[1]))
    skin = adaptive_skin_mask(original_rgb, reference, face_width_px, config)
    skin = dilate_mask(skin, config.skin_grow_ratio * face_width_px)
    features = dilate_mask(
        fill_feature_guard(shape, landmarks),
        config.feature_guard_ratio * face_width_px,
    )
    keep = np.maximum(skin, features)
    if config.skin_smooth_ratio > 0:
        sigma = max(config.skin_smooth_ratio * face_width_px, 0.5)
        info["smoothSigmaPx"] = round(sigma, 2)
        soft = cv2.GaussianBlur(keep.astype(np.float32) / 255.0, (0, 0), sigmaX=sigma)
        keep = (soft >= 0.5).astype(np.uint8) * 255
    return keep, info


def harmonise_generated(
    generated_rgb: Any,
    warped_original: Any,
    soft_mask: Any,
    reference: Any,
    face_width_px: float,
    config: FaceLockConfig,
) -> tuple[Any, dict[str, Any]]:
    """Tone-match the generated pixels to the photograph, outside the core.

    The styling model returns a brighter, cooler face than the input photo, so
    the restored original face lands next to generated skin that no longer
    matches it. This fits a smooth *additive* correction field on the pixels
    that are skin in both images (the only place "what should this pixel have
    been" is well defined), then fades it out with distance from the protection
    boundary.

    The field is confined to the render's *skin*: darkening the new hairstyle
    towards the photograph's lighting would both undo the designed hair colour
    and print a fresh step at the hairline, where the untouched pasted skin
    meets the darkened hair.

    It is purely tonal -- no geometry, no texture synthesis -- and it is applied
    to the generated image *before* compositing. Protected core pixels are
    written at alpha 1, so they never change.
    """
    stats: dict[str, Any] = {"applied": False}
    if not config.harmonise or reference is None:
        stats["reason"] = "disabled" if not config.harmonise else "no skin reference"
        return generated_rgb, stats

    generated = np.asarray(generated_rgb, dtype=np.float32)
    warped = np.asarray(warped_original, dtype=np.float32)
    distance = cv2.distanceTransform((soft_mask < 0.5).astype(np.uint8), cv2.DIST_L2, 5)
    near = distance <= config.harmonise_band_ratio * face_width_px
    target_skin = generated_skin_weight(
        generated_rgb, reference, face_width_px, config
    )
    both_skin = (
        (adaptive_skin_mask(warped_original, reference, face_width_px, config) > 0)
        & (adaptive_skin_mask(generated_rgb, reference, face_width_px, config) > 0)
    )
    valid = (near & both_skin).astype(np.float32)
    count = int(valid.sum())
    stats["samplePixels"] = count
    if count < 50:
        stats["reason"] = "not enough matching skin samples"
        return generated_rgb, stats

    difference = warped - generated
    median = np.median(difference[valid > 0], axis=0)
    # The median *is* the tone step this exists to remove: the render's skin is
    # brighter/cooler than the photograph's by exactly that much, and pasting
    # the photo next to it prints the step. Keep the median and clip the
    # excursions around it (specular highlights, the model's new jewellery).
    limited = np.clip(difference, median - 18.0, median + 18.0) * valid[..., None]
    sigma = max(config.harmonise_sigma_ratio * face_width_px, 0.5)
    blurred = cv2.GaussianBlur(limited, (0, 0), sigmaX=sigma)
    weight = cv2.GaussianBlur(valid, (0, 0), sigmaX=sigma)[..., None]
    correction = blurred / np.maximum(weight, 1e-3)
    taper = np.exp(-((distance / max(config.harmonise_band_ratio * face_width_px, 1.0)) ** 2))
    limited_field = np.clip(
        correction, -config.harmonise_clamp, config.harmonise_clamp
    )
    applied = limited_field * taper[..., None] * target_skin[..., None]
    stats.update(
        {
            "applied": True,
            "medianDifference": [round(float(value), 2) for value in median],
            "appliedAbsMean": round(float(np.abs(applied).mean()), 3),
            "appliedAbsMeanNearFace": round(
                float(np.abs(applied)[near].mean()), 3
            )
            if near.any()
            else 0.0,
            "skinPixelsNearFace": int((near & (target_skin > 0.5)).sum()),
            "clamp": config.harmonise_clamp,
        }
    )
    return np.clip(generated + applied, 0.0, 255.0), stats


def gaussian_mask(mask: Any, sigma_px: float) -> Any:
    if sigma_px <= 0:
        return mask.astype(np.float32) / 255.0
    blurred = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(sigma_px, 0.5))
    return (blurred.astype(np.float32) / 255.0).clip(0.0, 1.0)


def warp_image(image: Any, matrix: Any, size: tuple[int, int]) -> Any:
    return cv2.warpAffine(
        image,
        np.asarray(matrix, dtype=np.float64),
        (int(size[0]), int(size[1])),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REPLICATE,
    )


def warp_mask(mask: Any, matrix: Any, size: tuple[int, int]) -> Any:
    return cv2.warpAffine(
        mask,
        np.asarray(matrix, dtype=np.float64),
        (int(size[0]), int(size[1])),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def edge_magnitude(image_rgb: Any) -> Any:
    """Sobel gradient magnitude of an RGB image, in 8-bit grey levels."""
    gray = cv2.cvtColor(np.asarray(image_rgb, dtype=np.uint8), cv2.COLOR_RGB2GRAY)
    gray = gray.astype(np.float32)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(grad_x * grad_x + grad_y * grad_y)


# ---------------------------------------------------------------- measurement


def _region_centroid(landmarks: Any, indices: Sequence[int]) -> Any:
    return np.mean(np.asarray(landmarks)[list(indices)], axis=0)


def landmark_metrics(landmarks: Any) -> dict[str, float]:
    """Scale-free-ish measures used to detect non-rigid face rewriting."""
    mouth_width = float(
        np.linalg.norm(landmarks[MOUTH_LEFT] - landmarks[MOUTH_RIGHT])
    )
    nose_to_chin = float(np.linalg.norm(landmarks[NOSE_TIP] - landmarks[CHIN]))
    eye_center = _region_centroid(landmarks, EYE_CORNERS)
    interocular = float(np.linalg.norm(landmarks[33] - landmarks[362]))
    return {
        "mouth_width": mouth_width,
        "nose_to_chin": nose_to_chin,
        "interocular": interocular,
        "eye_center_x": float(eye_center[0]),
        "eye_center_y": float(eye_center[1]),
        "face_width": face_width(landmarks),
    }


def mouth_openness(landmarks: Any) -> float:
    """Vertical inner-lip gap: ~0 for a closed mouth, >0 for a visible smile."""
    upper = np.mean(landmarks[[13, 312, 311, 310, 415, 308]], axis=0)
    lower = np.mean(landmarks[[14, 82, 81, 80, 191, 78]], axis=0)
    return float(abs(upper[1] - lower[1]))


# ------------------------------------------------------------------ the lock


def _detect_single_face(detector: LandmarkDetector, image: Any) -> tuple[Any | None, int]:
    faces = detector.detect(image)
    count = len(faces)
    if count == 0:
        return None, 0
    # Prefer the largest face (the subject) but keep the real count for the gate.
    faces = sorted(faces, key=lambda pts: float(face_width(pts)), reverse=True)
    return faces[0], count


def read_manual_mask(path: Path | str, shape: tuple[int, int]) -> Any:
    """Load a hand-painted protection mask (white = must come from the original).

    The mask is read in the *original* photo's coordinate frame; if the human
    painted it on a differently sized copy it is resized (non-generative,
    nearest-neighbour) and the mismatch is reported to the caller. This is the
    escape hatch for the cases automatic landmarks get wrong -- for example hair
    or glasses the detector places badly.
    """
    from PIL import Image

    with Image.open(path) as handle:
        mask_image = handle.convert("L")
    if mask_image.size != (int(shape[1]), int(shape[0])):
        mask_image = mask_image.resize(
            (int(shape[1]), int(shape[0])), Image.Resampling.NEAREST
        )
    mask = np.asarray(mask_image)
    return (mask > 127).astype(np.uint8) * 255


def apply_face_lock(
    original_path: Path | str,
    generated_path: Path | str,
    output_path: Path | str,
    *,
    detector: LandmarkDetector | None = None,
    model_path: Path | str | None = None,
    config: FaceLockConfig | None = None,
    manual_mask_path: Path | str | None = None,
) -> FaceLockReport:
    """Restore the original face into ``generated_path`` and write ``output_path``.

    ``output_path`` is only written when every gate passes; on a failed gate the
    generated image is left untouched so a human can inspect it.

    ``manual_mask_path`` optionally adds a hand-painted protection region on top
    of the landmark-derived one.
    """
    config = config or FaceLockConfig()
    report = _apply_face_lock_inner(
        Path(original_path),
        Path(generated_path),
        Path(output_path),
        detector,
        model_path,
        config,
        Path(manual_mask_path) if manual_mask_path is not None else None,
    )
    return report


def _failure(
    status: FaceLockStatus,
    message: str,
    diagnostics: dict[str, Any] | None = None,
) -> FaceLockReport:
    return FaceLockReport(
        status=status,
        applied=False,
        message=message,
        diagnostics=diagnostics or {},
    )


def _apply_face_lock_inner(
    original_path: Path,
    generated_path: Path,
    output_path: Path,
    detector: LandmarkDetector | None,
    model_path: Path | str | None,
    config: FaceLockConfig,
    manual_mask_path: Path | None = None,
) -> FaceLockReport:
    from PIL import Image  # local import keeps the module import light

    if config.mode != FACE_LOCK_STRICT:
        return _failure(FaceLockStatus.DISABLED, f"Face lock mode is '{config.mode}'.")

    try:
        _require_stack()
    except FaceLockUnavailable as exc:
        return _failure(FaceLockStatus.UNAVAILABLE, str(exc))

    if detector is None:
        if model_path is None:
            return _failure(
                FaceLockStatus.UNAVAILABLE,
                "No face landmark detector or model path was provided.",
            )
        try:
            detector = default_detector(model_path)
        except FaceLockUnavailable as exc:
            return _failure(FaceLockStatus.UNAVAILABLE, str(exc))

    try:
        original_rgb = np.asarray(Image.open(original_path).convert("RGB"))
        generated_rgb = np.asarray(Image.open(generated_path).convert("RGB"))
    except Exception as exc:  # pragma: no cover - defensive
        return _failure(FaceLockStatus.RENDER_ERROR, f"Cannot read images: {exc}")

    diagnostics: dict[str, Any] = {
        "originalSize": [int(original_rgb.shape[1]), int(original_rgb.shape[0])],
        "generatedSize": [int(generated_rgb.shape[1]), int(generated_rgb.shape[0])],
        "mode": config.mode,
    }

    orig_face, orig_count = _detect_single_face(detector, original_rgb)
    gen_face, gen_count = _detect_single_face(detector, generated_rgb)
    diagnostics["originalFaceCount"] = orig_count
    diagnostics["generatedFaceCount"] = gen_count

    if orig_count == 0:
        return _failure(FaceLockStatus.NO_FACE, "No face found in the original photo.", diagnostics)
    if gen_count == 0:
        return _failure(
            FaceLockStatus.NO_FACE, "No face found in the generated image.", diagnostics
        )
    if orig_count > 1 or gen_count > 1:
        # Person count changed (or was never a single subject) -> refuse.
        return _failure(
            FaceLockStatus.MULTIPLE_FACES,
            f"Face count changed or is ambiguous (original={orig_count}, generated={gen_count}).",
            diagnostics,
        )

    return _lock_and_render(
        original_rgb,
        generated_rgb,
        orig_face,
        gen_face,
        output_path,
        config,
        diagnostics,
        manual_mask_path,
    )


def _lock_and_render(
    original_rgb: Any,
    generated_rgb: Any,
    orig_face: Any,
    gen_face: Any,
    output_path: Path,
    config: FaceLockConfig,
    diagnostics: dict[str, Any],
    manual_mask_path: Path | None = None,
) -> FaceLockReport:
    from PIL import Image

    width_orig = int(original_rgb.shape[1])
    height_orig = int(original_rgb.shape[0])
    width_gen = int(generated_rgb.shape[1])
    height_gen = int(generated_rgb.shape[0])

    orig_metrics = landmark_metrics(orig_face)
    gen_metrics = landmark_metrics(gen_face)
    orig_face_width = orig_metrics["face_width"] or 1.0

    # --- 1. registration on the rigid subset only -------------------------
    matrix = similarity_transform(orig_face[list(RIGID_FIT)], gen_face[list(RIGID_FIT)])
    scale, rotation = transform_decomposition(matrix)
    scale_deviation = abs(scale - 1.0)

    projected = apply_transform(matrix, orig_face)
    residuals = np.linalg.norm(projected - gen_face, axis=1)
    rigid_residual_p90 = float(np.percentile(residuals[list(RIGID_FIT)], 90))
    rigid_residual_ratio = rigid_residual_p90 / orig_face_width

    inverse = invert_similarity(matrix)
    gen_oval_in_orig = apply_transform(inverse, gen_face[list(FACE_OVAL)])

    coverage_extra_px = 0.0
    orig_oval_ring = orig_face[list(FACE_OVAL)]
    for point in gen_oval_in_orig:
        coverage_extra_px = max(
            coverage_extra_px, polygon_distance_outside(point, orig_oval_ring)
        )
    coverage_ratio = coverage_extra_px / orig_face_width

    diagnostics.update(
        {
            "registration": {
                "scale": round(scale, 5),
                "rotationDeg": round(rotation, 3),
                "scaleDeviation": round(scale_deviation, 5),
                "rigidResidualP90Px": round(rigid_residual_p90, 3),
                "rigidResidualRatio": round(rigid_residual_ratio, 5),
                "coverageExtraPx": round(coverage_extra_px, 3),
                "coverageRatio": round(coverage_ratio, 5),
                "scaleDeviationLimit": config.max_scale_deviation,
                "rotationLimitDeg": config.max_rotation_deg,
                "rigidResidualRatioLimit": config.max_rigid_residual_ratio,
                "rigidResidualStrictRatio": config.strict_rigid_residual_ratio,
                "rigidResidualStrictPassed": bool(
                    rigid_residual_ratio <= config.strict_rigid_residual_ratio
                ),
                "coverageRatioLimit": config.max_coverage_ratio,
            },
            "originalMetrics": {key: round(value, 3) for key, value in orig_metrics.items()},
            "generatedMetrics": {key: round(value, 3) for key, value in gen_metrics.items()},
            "nonRigidDrift": {
                "mouthWidthRatio": round(
                    gen_metrics["mouth_width"] / (orig_metrics["mouth_width"] * scale or 1.0), 5
                ),
                "noseToChinRatio": round(
                    gen_metrics["nose_to_chin"] / (orig_metrics["nose_to_chin"] * scale or 1.0), 5
                ),
                "eyeCenterShiftPx": round(
                    float(
                        np.linalg.norm(
                            np.array([gen_metrics["eye_center_x"], gen_metrics["eye_center_y"]])
                            - apply_transform(
                                matrix,
                                np.array([orig_metrics["eye_center_x"], orig_metrics["eye_center_y"]]),
                            )
                        )
                    ),
                    3,
                ),
                "mouthOpennessPx": {
                    "original": round(mouth_openness(orig_face), 3),
                    "generated": round(mouth_openness(gen_face), 3),
                },
            },
        }
    )

    if scale_deviation > config.max_scale_deviation:
        return _failure(
            FaceLockStatus.ALIGNMENT_FAILED,
            f"Face scale differs by {scale_deviation * 100:.1f}% between the original and "
            f"the generated image (limit {config.max_scale_deviation * 100:.1f}%).",
            diagnostics,
        )
    if abs(rotation) > config.max_rotation_deg:
        return _failure(
            FaceLockStatus.ALIGNMENT_FAILED,
            f"Face rotation differs by {rotation:.1f}deg (limit {config.max_rotation_deg:.1f}deg).",
            diagnostics,
        )
    if rigid_residual_ratio > config.max_rigid_residual_ratio:
        return _failure(
            FaceLockStatus.ALIGNMENT_FAILED,
            "The generated face is not a rigid transform of the original "
            f"(residual {rigid_residual_ratio * 100:.1f}% of face width, "
            f"limit {config.max_rigid_residual_ratio * 100:.1f}%).",
            diagnostics,
        )
    if coverage_ratio > config.max_coverage_ratio:
        return _failure(
            FaceLockStatus.COVERAGE_FAILED,
            "The generated head extends "
            f"{coverage_extra_px:.1f}px beyond the original face outline "
            f"(limit {config.max_coverage_ratio * orig_face_width:.1f}px); locking would "
            "paste non-face pixels or leave generated facial features visible.",
            diagnostics,
        )

    # --- 2. build the protection mask in original space -------------------
    core_poly = _convex_hull(orig_face[list(FACE_OVAL)])
    coverage_poly = _convex_hull(np.vstack([orig_face[list(FACE_OVAL)], gen_oval_in_orig]))

    core_mask = fill_polygon((height_orig, width_orig), core_poly)
    coverage_mask = fill_polygon((height_orig, width_orig), coverage_poly)
    paste_mask = dilate_mask(
        coverage_mask, config.paste_dilate_ratio * orig_face_width
    )

    # The face outline also covers whatever the subject *had* there in the
    # original photo -- for a long-haired subject that is their old fringe and
    # temples. Pasting those pixels over the new hairstyle is what prints the
    # "sticker" band along the hairline; keep only the subject's own face.
    skin_reference, skin_samples = skin_reference_colour(
        original_rgb, orig_face, orig_face_width
    )
    keep_info: dict[str, Any] = {"samplePixels": int(skin_samples), "applied": False}
    if skin_reference is not None:
        keep_info["referenceColour"] = [int(round(float(value))) for value in skin_reference]
    if config.skin_refine and skin_reference is not None:
        keep_mask, keep_detail = protection_keep_mask(
            original_rgb, orig_face, orig_face_width, skin_reference, config
        )
        keep_info.update({k: v for k, v in keep_detail.items() if k != "applied"})
        if keep_mask is not None:
            refined_core = cv2.bitwise_and(core_mask, keep_mask)
            keep_ratio = float((refined_core > 0).sum()) / max(
                float((core_mask > 0).sum()), 1.0
            )
            keep_info["coreCoverage"] = round(keep_ratio, 4)
            keep_info["minCoreCoverage"] = config.min_core_coverage
            if keep_ratio >= config.min_core_coverage:
                keep_info["droppedPastePixels"] = int(
                    ((paste_mask > 0) & (keep_mask == 0)).sum()
                )
                paste_mask = cv2.bitwise_and(paste_mask, keep_mask)
                hard_core = refined_core
                if config.core_erode_ratio > 0:
                    hard_core = erode_mask(
                        refined_core, config.core_erode_ratio * orig_face_width
                    )
                    keep_info["coreErodePx"] = round(
                        config.core_erode_ratio * orig_face_width, 2
                    )
                    keep_info["hardCorePixels"] = int((hard_core > 0).sum())
                core_mask = hard_core
                keep_info["applied"] = True
            else:
                keep_info["reason"] = "refined core below the coverage floor"
    elif not config.skin_refine:
        keep_info["reason"] = "disabled"
    else:
        keep_info["reason"] = "no reliable skin reference"

    # A hand-painted mask (glasses frame, an awkward fringe, a landmark the
    # detector placed badly) is treated as *core*: the pixels there are the
    # original photograph at full opacity, never a feathered blend.
    manual_mask = None
    if manual_mask_path is not None:
        try:
            manual_mask = read_manual_mask(manual_mask_path, (height_orig, width_orig))
        except Exception as exc:
            return _failure(
                FaceLockStatus.RENDER_ERROR,
                f"Could not read the manual face-protection mask: {exc}",
                diagnostics,
            )
        paste_mask = np.maximum(paste_mask, manual_mask)
        core_mask = np.maximum(core_mask, manual_mask)

    soft_mask_orig = gaussian_mask(paste_mask, config.feather_ratio * orig_face_width)
    # Never let the feather touch the core: inside the core the content is the
    # original photo, full stop.
    soft_mask_orig = np.maximum(soft_mask_orig, core_mask.astype(np.float32) / 255.0)

    # --- 3. warp mask + original content into the generated frame ---------
    warped_original = warp_image(original_rgb, matrix, (width_gen, height_gen))
    soft_gen = warp_mask(
        (soft_mask_orig * 255.0).astype(np.uint8), matrix, (width_gen, height_gen)
    ).astype(np.float32) / 255.0
    core_gen = warp_mask(core_mask, matrix, (width_gen, height_gen)) > 127
    # The mask travels through an 8-bit buffer, so bilinear resampling leaves a
    # one-pixel ring just inside the protected core where alpha dips below 1 and
    # the model's pixels would bleed back in. Pin the core to the original at
    # full opacity -- the protected pixels are the original photograph, exactly.
    soft_gen = np.maximum(soft_gen, core_gen.astype(np.float32))

    # Tone-match the generated side of the boundary to the photograph. Applied
    # before compositing, so the alpha-1 core below is unaffected by design.
    generated_rgb, harmonise_stats = harmonise_generated(
        generated_rgb, warped_original, soft_gen, skin_reference, orig_face_width, config
    )

    alpha = soft_gen[..., None]
    composite = warped_original.astype(np.float32) * alpha + generated_rgb.astype(np.float32) * (
        1.0 - alpha
    )
    result = np.clip(composite, 0, 255).astype(np.uint8)

    # --- 4. verification ---------------------------------------------------
    # Two very different quantities live here and only one of them is a seam:
    #   * sourceDifference -- how far apart the two photographs are inside the
    #     feather band. For a real styling job this is *huge by design* (new
    #     hair, new clothes, new lighting). It is reported for transparency and
    #     must never be read as an artifact.
    #   * introducedEdge -- an edge that exists in the composited output but in
    #     NEITHER source. That is what a visible seam actually is: the blend
    #     printing a contour the photographs never had. A smooth ramp never
    #     produces one, a hard cut always does. This is what gates the job.
    band = (soft_gen > 0.02) & (soft_gen < 0.98)
    ring = (soft_gen >= 0.25) & (soft_gen <= 0.75)

    source_difference = np.abs(
        warped_original.astype(np.float32) - generated_rgb.astype(np.float32)
    ).mean(axis=2)
    introduced = np.clip(
        edge_magnitude(result)
        - np.maximum(edge_magnitude(warped_original), edge_magnitude(generated_rgb)),
        0.0,
        None,
    )

    if band.any():
        source_p95 = float(np.percentile(source_difference[band], 95))
        source_p99 = float(np.percentile(source_difference[band], 99))
    else:  # pragma: no cover - would mean an empty feather band
        source_p95 = 0.0
        source_p99 = 0.0
    if ring.any():
        seam_excess_p95 = float(np.percentile(introduced[ring], 95))
        seam_excess_p99 = float(np.percentile(introduced[ring], 99))
    else:  # pragma: no cover - would mean an empty feather ring
        seam_excess_p95 = 0.0
        seam_excess_p99 = 0.0

    if core_gen.any():
        core_delta = np.abs(
            result.astype(np.float32)[core_gen] - warped_original.astype(np.float32)[core_gen]
        )
        core_mean = float(core_delta.mean())
        core_max = float(core_delta.max())
    else:  # pragma: no cover - would mean the face fell outside the frame
        core_mean = 0.0
        core_max = 0.0

    teeth_indices = list(LIPS_INNER)
    teeth_mask_orig = fill_polygon(
        (height_orig, width_orig),
        _convex_hull(orig_face[teeth_indices]),
    )
    teeth_mask_gen = warp_mask(teeth_mask_orig, matrix, (width_gen, height_gen)) > 127
    if teeth_mask_gen.any():
        teeth_region = teeth_mask_gen
        teeth_delta = float(
            np.abs(
                result.astype(np.float32)[teeth_region]
                - warped_original.astype(np.float32)[teeth_region]
            ).mean()
        )
    else:  # pragma: no cover - would mean the mouth fell outside the frame
        teeth_delta = 0.0

    projected_eye_center = np.mean(
        apply_transform(matrix, orig_face[list(EYE_CORNERS)]), axis=0
    )
    generated_identity = abs(gen_metrics["eye_center_y"] - float(projected_eye_center[1]))

    diagnostics.update(
        {
            "mask": {
                "pastePixels": int((paste_mask > 0).sum()),
                "corePixels": int((core_mask > 0).sum()),
                "coverageExtraPx": round(coverage_extra_px, 3),
                "manualMask": {
                    "provided": manual_mask is not None,
                    "path": str(manual_mask_path) if manual_mask_path else None,
                    "pixels": int((manual_mask > 0).sum()) if manual_mask is not None else 0,
                },
                "skinRefinement": keep_info,
                "harmonisation": harmonise_stats,
            },
            "seam": {
                "bandPixels": int(band.sum()),
                "ringPixels": int(ring.sum()),
                "introducedEdgeP95": round(seam_excess_p95, 3),
                "introducedEdgeP99": round(seam_excess_p99, 3),
                "introducedEdgeLimit": config.max_seam_excess_p95,
                "sourceDifferenceP95": round(source_p95, 3),
                "sourceDifferenceP99": round(source_p99, 3),
                "note": (
                    "introducedEdge* gates the job: an edge the blend printed that "
                    "neither source had. sourceDifference* is informational only -- "
                    "new hair/clothes/lighting make the two sources differ strongly "
                    "inside the band by design."
                ),
            },
            "coreIdentity": {
                "meanAbsDiff": round(core_mean, 5),
                "maxAbsDiff": round(core_max, 5),
                "meanLimit": config.core_identity_mean_tol,
                "maxLimit": config.core_identity_max_tol,
            },
            "teethRegionMeanAbsDiff": round(teeth_delta, 5),
            "lockedRegions": {
                name: {
                    "landmarks": len(indices),
                    "centroidXy": [
                        round(float(np.mean(orig_face[list(indices)], axis=0)[0]), 1),
                        round(float(np.mean(orig_face[list(indices)], axis=0)[1]), 1),
                    ],
                }
                for name, indices in LOCKED_REGIONS.items()
            },
        }
    )

    if seam_excess_p95 > config.max_seam_excess_p95:
        return _failure(
            FaceLockStatus.SEAM_UNSAFE,
            "Blending the original face introduced a visible contour "
            f"(introduced edge p95={seam_excess_p95:.1f}/255, limit "
            f"{config.max_seam_excess_p95:.1f}); the original face cannot be restored "
            "without a seam.",
            diagnostics,
        )
    if core_mean > config.core_identity_mean_tol or core_max > config.core_identity_max_tol:
        return _failure(
            FaceLockStatus.IDENTITY_CHECK_FAILED,
            "The protected facial core does not match the original photo "
            f"(mean={core_mean:.2f}, max={core_max:.2f}).",
            diagnostics,
        )

    # --- 5. write the locked image ----------------------------------------
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        suffix = output_path.suffix.lower()
        image = Image.fromarray(result)
        if suffix in (".jpg", ".jpeg"):
            image.save(output_path, format="JPEG", quality=95, subsampling=0)
        elif suffix == ".webp":
            image.save(output_path, format="WEBP", quality=95)
        else:
            image.save(output_path, format="PNG")
    except Exception as exc:  # pragma: no cover - defensive
        return _failure(FaceLockStatus.RENDER_ERROR, f"Could not write output: {exc}", diagnostics)

    diagnostics["outputSize"] = [int(result.shape[1]), int(result.shape[0])]
    diagnostics["generatedEyeCenterOffsetPx"] = round(float(generated_identity), 3)

    return FaceLockReport(
        status=FaceLockStatus.APPLIED,
        applied=True,
        message=(
            "Original face restored: eyes, nose, mouth, teeth, cheeks, chin and the "
            "outer face contour come from the original photograph."
        ),
        diagnostics=diagnostics,
    )


# ------------------------------------------------- non-generative makeup pass


@dataclass(frozen=True)
class MakeupSpec:
    """Colour-only makeup. Geometry is never touched (luminance is preserved)."""

    lip_color: tuple[int, int, int] | None = None
    brow_color: tuple[int, int, int] | None = None
    eyeshadow_color: tuple[int, int, int] | None = None
    strength: float = 0.55
    label: str = ""


def _rgb_to_lab(rgb: Any) -> Any:
    from PIL import Image

    return np.asarray(Image.fromarray(np.ascontiguousarray(rgb), mode="RGB").convert("LAB")).astype(
        np.float32
    )


def _lab_to_rgb(lab: Any) -> Any:
    from PIL import Image

    clipped = np.clip(lab, 0, 255).astype(np.uint8)
    return np.asarray(Image.fromarray(clipped, mode="LAB").convert("RGB"))


def _lab_of_color(color: tuple[int, int, int]) -> Any:
    swatch = np.full((1, 1, 3), color, dtype=np.uint8)
    return _rgb_to_lab(swatch).reshape(3)


#: The chroma-blend pass never writes the luminance channel; the only residual
#: drift is 8-bit sRGB gamut clipping of a saturated colour at an extreme L.
LUMINANCE_GAMUT_TOLERANCE = 16.0


def apply_non_generative_makeup(
    image_rgb: Any,
    landmarks: Any,
    spec: MakeupSpec,
) -> tuple[Any, dict[str, Any]]:
    """Recolour lips / brows / eyelids on the *original* face pixels.

    Only the chroma channels (a, b) are blended toward the target colour; the
    luminance channel is copied verbatim from the original. The eye, nose and
    mouth contours are never moved, so this cannot enlarge an eye, redraw a nose
    bridge or reshape a jaw. It is the only kind of makeup the strict face lock
    allows while the face is pinned to the original pixels.
    """
    _require_stack()

    height, width = image_rgb.shape[:2]
    source = np.asarray(image_rgb, dtype=np.uint8)
    lab = _rgb_to_lab(source)
    lab_l = lab[..., 0].copy()
    strength = max(0.0, min(1.0, float(spec.strength)))
    applied: dict[str, Any] = {"label": spec.label, "strength": strength, "regions": {}}

    #: Per-pixel makeup coverage. Everything outside it must come back from the
    #: original photograph byte for byte; only the covered pixels are recoloured.
    coverage = np.zeros((height, width), dtype=np.float32)

    def _blend(mask: Any, color: tuple[int, int, int]) -> None:
        target = _lab_of_color(color)[1:]
        weight = (np.asarray(mask, dtype=np.float32) * strength).clip(0.0, 1.0)
        np.maximum(coverage, weight, out=coverage)
        blended = lab[..., 1:] * (1.0 - weight[..., None]) + target.reshape(1, 1, 2) * weight[..., None]
        lab[..., 1:] = blended

    def _soft_polygon(indices: Sequence[int], grow_px: float) -> Any:
        ring = fill_polygon((height, width), _convex_hull(landmarks[list(indices)]))
        if grow_px:
            ring = dilate_mask(ring, grow_px)
        return gaussian_mask(ring, max(1.0, grow_px * 0.6))

    face_w = face_width(landmarks)

    if spec.lip_color is not None:
        mouth = dilate_mask(
            fill_polygon((height, width), _convex_hull(landmarks[list(LIPS_OUTER)])),
            face_w * 0.010,
        )
        soft = gaussian_mask(mouth, face_w * 0.012)
        _blend(soft, spec.lip_color)
        applied["regions"]["lips"] = int((soft > 0.5).sum())

    if spec.brow_color is not None:
        brows = dilate_mask(
            fill_polygon(
                (height, width), _convex_hull(landmarks[list(BROW_A) + list(BROW_B)])
            ),
            face_w * 0.012,
        )
        soft = gaussian_mask(brows, face_w * 0.015)
        _blend(soft, spec.brow_color)
        applied["regions"]["brows"] = int((soft > 0.5).sum())

    if spec.eyeshadow_color is not None:
        eye_poly = _convex_hull(landmarks[list(EYE_A) + list(EYE_B)])
        ring = dilate_mask(fill_polygon((height, width), eye_poly), face_w * 0.045)
        opening = fill_polygon((height, width), eye_poly)
        ring[opening > 0] = 0
        # The convex hull of *both* eyes bridges the eye line straight across the
        # nose, so the lid ring would paint eyeshadow onto the nose bridge. The
        # nose is the one region a makeup pass must never reach, so it is cut out
        # of the ring (and out of the blurred result, which would otherwise creep
        # a pixel or two back in).
        nose_guard = fill_polygon((height, width), _convex_hull(landmarks[list(NOSE)]))
        ring[nose_guard > 0] = 0
        soft = gaussian_mask(ring, face_w * 0.020)
        soft[nose_guard > 0] = 0.0
        _blend(soft, spec.eyeshadow_color)
        applied["regions"]["eyeshadow"] = int((soft > 0.5).sum())

    recoloured = _lab_to_rgb(lab)
    alpha = coverage[..., None]
    merged = source.astype(np.float32) * (1.0 - alpha) + recoloured.astype(np.float32) * alpha
    out = np.clip(np.rint(merged), 0.0, 255.0).astype(np.uint8)

    untouched = coverage <= 0.0
    applied["chromaOnly"] = True
    applied["unmaskedPixelsUnchanged"] = bool(
        np.array_equal(out[untouched], source[untouched])
    )
    out_lab = _rgb_to_lab(out)
    delta = np.abs(out_lab[..., 0] - lab_l)
    applied["maxLuminanceDelta"] = float(delta.max())
    # Only the chroma (a, b) channels of the working buffer are ever blended, so
    # luminance can only drift where sRGB gamut clipping pulls a saturated colour
    # back into range -- never because the L channel was drawn on.
    applied["luminanceUnchanged"] = bool(delta.max() <= LUMINANCE_GAMUT_TOLERANCE)
    applied["maskedRegions"] = int((~untouched).sum())
    return out, applied
