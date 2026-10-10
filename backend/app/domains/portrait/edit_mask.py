"""Local edit masks that protect the face and open only hair and clothing.

The styling model repaints the *whole* person -- a prompt cannot keep a smile --
so the pipeline needs a mask expressed in the original photo's own coordinates
that says which pixels a generator may touch and which must come back unchanged.

The mask semantics are the ones every mature inpainting stack uses (diffusers
``AutoPipelineForInpainting``, BrushNet, the hosted fal.ai endpoints):

    white (255) = repaint          black (0) = keep the original pixels

``editable`` is therefore the generator input, and :meth:`EditMaskSet.preserve`
is its exact complement: identity plus every pixel the module did not select.
"Unselected stays original" is what that complement means in practice.

Two mask builders live here
---------------------------
:func:`build_edit_masks` (P2.6) is the original *geometric* builder. It needs
nothing but MediaPipe landmarks, and its hair/clothing regions are an ellipse
around the head and a rectangle under the chin -- documented stand-ins, not
measurements. It is kept for compatibility and for the offline review that uses
it, and it is never silently substituted for a failed semantic run.

:func:`build_semantic_edit_masks` (P2.7) is the real one. It takes the ATR
18-class label map produced by :mod:`app.domains.portrait.human_parsing`
(pretrained SCHP, ONNX, CPU) and builds the regions out of *parsed* hair,
clothing and face instead of guessed geometry. It is the path the pipeline uses;
if the parser is unavailable it raises rather than falling back to the ellipse.

What is real and what is a stand-in
-----------------------------------
* **Identity is real geometry.** The eyes, brows, nose and lips come from the
  pretrained MediaPipe face mesh, and the mouth hull covers the teeth of an open
  smile, so no generator can rewrite the expression.
* **Hair, clothing and the face/neck are parsed, not guessed.** The ATR model
  labels them per pixel, which is what replaces the P2.6 ellipse: the hair mask
  follows this subject's actual silhouette, and a new hairstyle may grow a
  bounded margin *outside* it instead of being pinned inside it.
* **The parsed face island is protected whole -- minus a collar it mistook for
  anatomy.** ATR has no neck class, so the neck comes back as ``Face``; the
  guard is therefore the full parsed face+neck island unioned with the MediaPipe
  oval, not a skin-tone-filtered subset. A shadowed neck is skin that fails a
  skin-tone test, and a throat that became editable would be repainted by a
  garment swap. The single carve-out is the region *below the chin row* that is
  **not** the subject's own skin: a white collar is a garment, not a face, and
  freezing it with the face is what previously stopped a new outfit from ever
  reaching the neckline. The neck skin below the chin is measured and stays
  protected. The subject's own skin tone is sampled both for that split and as a
  quality signal on the parse (``faceSkinShare``).
* **Speckle parses are dropped, not shipped.** The net sometimes tags a sliver
  of background at the frame edge as hair or as a garment. Grown by a rim, that
  sliver opens a patch of background to the generator, so islands below a
  fraction of the largest one are removed before the regions are grown, and the
  removal is reported in ``stats`` and ``gaps``.
* **Glasses are geometry, not classification.** ATR only has a ``Sunglasses``
  class; clear prescription glasses come back as ``Face``/``Hair``. The lens
  band is therefore guarded from the MediaPipe eye rings and face sides.
* **The parsing boundary is soft.** ATR is a 512x512 network upsampled to the
  photo, so the hairline and the collar are approximate to a few pixels; that
  residual is reported in ``gaps`` for human correction, not hidden.

Nothing here generates pixels, trains anything or calls a paid provider.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from app.domains.portrait.face_lock import (
    CHIN,
    EYE_A,
    EYE_B,
    FACE_OVAL,
    LIPS_INNER,
    FaceLockConfig,
    adaptive_skin_mask,
    dilate_mask,
    erode_mask,
    face_width,
    fill_feature_guard,
    fill_polygon,
    gaussian_mask,
    skin_reference_colour,
)
from app.domains.portrait.human_parsing import (
    ATR_CLASSES,
    CLASS_COUNT,
    CLOTHING_CLASSES,
    FACE_CLASSES,
    HAIR,
)

#: Photo indices whose bounding box defines the face rectangle handed to humans.
FACE_BOX_SOURCES = FACE_OVAL

#: Landmarks that bracket a pair of spectacles: both eye rings, the nose bridge
#: and the face sides at eye height. ATR only knows ``Sunglasses``, and clear
#: prescription lenses come back as ``Face``/``Hair``, so the lens band has to
#: come from the face mesh instead of from the classifier.
GLASSES_BAND = (
    EYE_A + EYE_B + (168, 6, 197, 195, 5, 4, 127, 356, 234, 454, 93, 323)
)


@dataclass(frozen=True)
class EditMaskConfig:
    """Geometry knobs. All ratios are relative to the detected face width."""

    #: Grow the identity guard so a feathered composite cannot eat an eyelash.
    feature_grow_ratio: float = 0.030
    #: Grow the protected face so its boundary sits just outside the jaw/cheek.
    face_grow_ratio: float = 0.020
    #: Head ellipse: how far the hair band reaches past the face, sideways/up.
    hair_side_ratio: float = 0.85
    hair_up_ratio: float = 0.55
    #: Clothing band: gap below the chin, and how wide past the face it runs.
    clothing_gap_ratio: float = 0.10
    clothing_side_ratio: float = 1.60
    #: Soft transition allowed where an editable region meets a kept region.
    feather_ratio: float = 0.020
    #: Below this a "region" is speckle; it is dropped and reported instead.
    min_region_px: int = 256


@dataclass
class EditMaskSet:
    """The three masks a masked edit needs, plus what a human must verify."""

    #: Identity and face pixels that must be the original photograph.
    protected: Any
    #: Pixels a generator may repaint. On the geometric path these are the hair
    #: and clothing bands; on the parsed path, the real hair and garment regions.
    editable: Any
    face_width_px: float
    feather_px: float
    stats: dict[str, Any] = field(default_factory=dict)
    gaps: tuple[str, ...] = ()
    #: Semantic regions, present only on the parsed (P2.7) path. They are
    #: ``None`` on the geometric (P2.6) path, where hair and clothing are bands
    #: rather than parsed regions, so a caller can always tell the two apart.
    hair: Any = None
    clothing: Any = None
    background_preserve: Any = None
    label_map: Any = None
    #: The parsed face/neck island *before* the skin and identity guards are
    #: unioned in: what the parser said, kept for review and for the gap report.
    parsed_face: Any = None
    #: The anatomy half of the parse: the parsed face/neck island with the
    #: garment the parser mistook for the jaw carved out. This is what the
    #: protected guard is built from.
    face_core: Any = None
    #: The piece carved out of the parsed face: a collar the parser labelled
    #: ``Face``. It joins the clothing region. Empty on the geometric path and
    #: whenever the chin split did not apply.
    collar: Any = None

    @property
    def semantic(self) -> bool:
        """True when this set came from a parsed label map, not from geometry."""
        return self.label_map is not None

    @property
    def preserve(self) -> Any:
        """Complement of ``editable``: identity plus everything unselected."""
        return cv2.bitwise_not(self.editable)

    def editable_weight(self) -> Any:
        """Soft 0-1 editable weight with the boundary feather applied."""
        return gaussian_mask(self.editable, self.feather_px)

    def protected_weight(self) -> Any:
        """Soft 0-1 protected weight with the boundary feather applied."""
        return gaussian_mask(self.protected, self.feather_px)


Graph = dict[str, Any]


def build_edit_masks(
    image_rgb: Any,
    landmarks: Any,
    *,
    config: EditMaskConfig | None = None,
    face_lock_config: FaceLockConfig | None = None,
) -> EditMaskSet:
    """Mask one face in one photo. Pure geometry -- no model call, no download."""
    cfg = config or EditMaskConfig()
    lock_cfg = face_lock_config or FaceLockConfig()

    image = np.asarray(image_rgb)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("build_edit_masks expects an HxWx3 RGB image")
    height, width = int(image.shape[0]), int(image.shape[1])
    shape = (height, width)

    lm = np.asarray(landmarks, dtype=np.float64)
    if lm.ndim != 2 or lm.shape[1] != 2:
        raise ValueError("landmarks must be an (N, 2) array of pixel coordinates")
    face_w = face_width(lm)
    if not np.isfinite(face_w) or face_w <= 0:
        raise ValueError("degenerate landmarks: face width is not positive")

    gaps: list[str] = []

    # 1. Identity guard. Teeth of an open smile live inside the convex hull of
    #    the outer lip ring, so protecting that hull protects the expression.
    features = fill_feature_guard(shape, lm)
    features = np.maximum(features, fill_polygon(shape, _hull(lm[list(LIPS_INNER)])))
    features = dilate_mask(features, cfg.feature_grow_ratio * face_w)

    # 2. The face itself, judged by the subject's own skin tone. A dark fringe
    #    over the forehead is not skin, so it stays in the editable hair band.
    oval = fill_polygon(shape, _hull(lm[list(FACE_OVAL)]))
    reference, sample_px = skin_reference_colour(image, lm, face_w)
    if reference is None:
        face_skin = oval
        gaps.append(
            "skin_reference_unavailable: the face was too small or too stylised to "
            "sample its own skin tone, so the whole face oval is protected and no "
            "fringe overlapping it can be restyled"
        )
    else:
        face_skin = cv2.bitwise_and(
            oval, adaptive_skin_mask(image, reference, face_w, lock_cfg)
        )
        # A thin occluder crossing the skin (glasses wire, lip line, specular
        # highlight) must not punch a hole that a generator could paint through.
        face_skin = dilate_mask(face_skin, cfg.face_grow_ratio * face_w)

    protected = np.maximum(features, face_skin)

    # 3. Hair band: an ellipse around the head minus everything protected. The
    #    part of the ellipse that lands on the face disappears with the
    #    subtraction, which is exactly the "hair may change, face may not" rule.
    oval_pts = lm[list(FACE_OVAL)]
    x0, x1 = float(oval_pts[:, 0].min()), float(oval_pts[:, 0].max())
    y0, y1 = float(oval_pts[:, 1].min()), float(oval_pts[:, 1].max())
    face_h = max(y1 - y0, 1.0)
    face_cx = 0.5 * (x0 + x1)
    head = _ellipse_mask(
        shape,
        center=(face_cx, y0 + 0.42 * face_h),
        axes=(
            0.5 * (x1 - x0) + cfg.hair_side_ratio * face_w,
            0.55 * face_h + cfg.hair_up_ratio * face_w,
        ),
    )
    hair = cv2.bitwise_and(head, cv2.bitwise_not(protected))

    # 4. Clothing band: below the chin, out towards the shoulders. It cannot be
    #    separated from the neck, the arms or the background without a human
    #    parsing model, and that limit is reported rather than papered over.
    chin_y = float(lm[CHIN][1]) if len(lm) > CHIN else y1
    top = int(round(max(chin_y, y1) + cfg.clothing_gap_ratio * face_h))
    left = int(round(max(0.0, x0 - cfg.clothing_side_ratio * face_w)))
    right = int(round(min(float(width), x1 + cfg.clothing_side_ratio * face_w)))
    clothing = np.zeros(shape, np.uint8)
    if top < height and right > left:
        clothing[top:height, left:right] = 255
    clothing = cv2.bitwise_and(clothing, cv2.bitwise_not(protected))

    # 5. Drop speckle regions instead of shipping a mask that looks broken.
    for name, region in (("hair", hair), ("clothing", clothing)):
        pixels = int((region > 0).sum())
        if 0 < pixels < cfg.min_region_px:
            region[:] = 0
            gaps.append(
                f"{name}_region_below_floor: only {pixels} pixels survived the "
                f"face guard (< {cfg.min_region_px}); the region was dropped"
            )

    editable = np.maximum(hair, clothing)
    feather_px = cfg.feather_ratio * face_w

    gaps.append(
        "hair_band_is_geometric: no hair-segmentation model is available in this "
        "environment, so the hair region is an ellipse around the head and not a "
        "hair matte; its boundary is a heuristic, not a measured hairline"
    )
    gaps.append(
        "clothing_band_is_geometric: no human-parsing/segmentation model is "
        "available here, so the clothing rectangle also covers neck, arms and "
        "background pixels that share the band"
    )

    stats: Graph = {
        "imageWidth": width,
        "imageHeight": height,
        "faceBox": [round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2)],
        "faceWidthPx": round(face_w, 2),
        "faceHeightPx": round(face_h, 2),
        "featherPx": round(feather_px, 2),
        "skinSamplePixels": int(sample_px),
        "protectedPixels": int((protected > 0).sum()),
        "hairPixels": int((hair > 0).sum()),
        "clothingPixels": int((clothing > 0).sum()),
        "editablePixels": int((editable > 0).sum()),
        "protectedShare": round(float((protected > 0).mean()), 4),
        "editableShare": round(float((editable > 0).mean()), 4),
        "identityGuardPixels": int((features > 0).sum()),
        "faceSkinPixels": int((face_skin > 0).sum()),
        "identityCoversEditable": bool(
            cv2.bitwise_and(features, editable).sum() == 0
        ),
    }

    return EditMaskSet(
        protected=protected,
        editable=editable,
        face_width_px=float(face_w),
        feather_px=float(feather_px),
        stats=stats,
        gaps=tuple(gaps),
    )


@dataclass(frozen=True)
class SemanticMaskConfig:
    """Geometry knobs for the parsed (ATR) mask path.

    Every ratio is relative to the detected face width, exactly like
    :class:`EditMaskConfig`, so the two paths can be compared knob for knob.
    """

    #: Grow the identity guard so a feathered composite cannot eat an eyelash.
    feature_grow_ratio: float = 0.030
    #: Grow the protected face/neck island so its boundary sits just outside
    #: the jaw.
    face_grow_ratio: float = 0.020
    #: Spectacle guard, grown from the eye-ring hull.
    glasses_grow_ratio: float = 0.020
    #: How far a *new* hairstyle may reach outside the parsed hair. This is the
    #: "not locked inside the old hair" allowance: a bounded rim around the real
    #: silhouette, not a head-sized ellipse.
    hair_grow_ratio: float = 0.12
    #: How far that rim may reach *over a parsed garment*. The wide rim above is
    #: right on the background side of the silhouette, but on the clothes side it
    #: let the hair pass repaint a collar's worth of pixels and then hand the
    #: same pixels to the clothing pass -- two passes over one seam, which is
    #: what printed the pale ghost below the chin. Over a garment the rim is
    #: tightened to this ratio, so only a hair's-breadth of overhang can
    #: repaint cloth and the hair/clothing boundary stays a single clean line.
    hair_rim_over_clothing_ratio: float = 0.02
    #: How far a *new* garment may reach outside the parsed one, so a collar or
    #: a shoulder line can move without opening the background or the arms.
    clothing_grow_ratio: float = 0.06
    #: Clear air the garment keeps from the parsed hair. The garment is grown
    #: for a new collar/shoulder line and then this band is taken back out, so
    #: at zero the clothing mask stops exactly on the parse's own hair boundary
    #: and the hair lying on the clothes is owned by the hair pass alone instead
    #: of being repainted by both passes.
    clothing_hair_clearance_ratio: float = 0.0
    #: Freed from the face guard before it is shipped: the parsed hair is eroded
    #: by this ratio and subtracted from the protected island, so the subject's
    #: own fringe is repainted with the new hairstyle instead of being frozen as
    #: a dark band under it. At zero the whole parsed fringe is released, which
    #: is safe here because a class map never puts one pixel in both ``Hair``
    #: and ``Face``: no parsed skin pixel can be freed, only parsed hair. Raise
    #: it only if a parse is seen spilling a few pixels of hair onto bare brow.
    hair_guard_release_erode_ratio: float = 0.0
    #: Soft transition allowed where an editable region meets a kept region.
    feather_ratio: float = 0.020
    #: Below this a "region" is speckle; it is dropped and reported instead.
    min_region_px: int = 256
    #: A parsed island smaller than this share of the largest island is speckle
    #: (a background sliver mislabelled as hair or garment) and is dropped.
    component_floor_ratio: float = 0.05
    #: Split the parsed face at the chin so a garment the parser filed as
    #: ``Face`` -- a high collar that rises to the jaw -- becomes editable while
    #: the throat stays protected. Needs the subject's own skin tone; if no
    #: sample can be taken the island is protected whole and the gap is reported
    #: rather than guessed at.
    collar_split_enabled: bool = True
    #: Parsed-``Face`` rows below ``CHIN + this * face width`` are candidates for
    #: the split, so the jaw itself can never be handed to a garment.
    collar_margin_ratio: float = 0.02
    #: Below this many pixels the non-skin mass below the chin is treated as
    #: noise rather than as a collar, and the parsed island is kept whole.
    collar_min_pixels: int = 64
    #: A below-chin ``Face`` pixel that is not skin *and* is darker than this
    #: share of the subject's own skin luminance is a shadowed throat, not a
    #: collar, and is protected rather than carved. Without it the darkest
    #: shadow under the jaw was handed to the garment and repainted, which is
    #: what left the grey cast on the underside of the chin.
    collar_skin_luma_ratio: float = 0.35
    #: The split above only reaches rows at ``CHIN + collar_margin_ratio * face
    #: width``, so a garment that rises to touch the jaw keeps a margin-height
    #: band frozen with the face. On the reference photo that band is the top
    #: edge of a white knit collar -- ~805 px the garment pass could never
    #: repaint, and the last paid run printed it as a pale ghost under the chin.
    #: When this is on, ``protected`` pixels below the chin row that pass the
    #: *same* tests as the split (not the subject's skin, not the shadow under
    #: the jaw) and carry independent evidence of being a garment are handed to
    #: the clothing region as well. It can only ever release pixels the split
    #: would already have released at a zero margin; the dark throat, the
    #: identity guard, the spectacle band and every skin pixel stay protected.
    collar_margin_recovery_enabled: bool = True
    #: Independent evidence #1: the proximity ring around the collar and the
    #: parsed garment within which recovered pixels must sit. Bounds the release
    #: to the collar's own neighbourhood, as a fraction of the face width.
    collar_margin_recovery_ring_ratio: float = 0.02
    #: Independent evidence #2: a pixel brighter than this share of the
    #: subject's *own* skin luminance is brighter than skin that was sampled at
    #: all, so it cannot be a shadowed throat. Paired with the chroma ceiling
    #: below it isolates a fabric highlight instead of lit skin.
    collar_margin_recovery_luma_ratio: float = 0.90
    #: Chroma ceiling (max channel - min channel) for "neutral fabric": skin
    #: keeps a high red-minus-blue spread even in shadow, a white knit does not.
    collar_margin_recovery_chroma: float = 45.0


def build_semantic_edit_masks(
    image_rgb: Any,
    landmarks: Any,
    label_map: Any,
    *,
    config: SemanticMaskConfig | None = None,
    face_lock_config: FaceLockConfig | None = None,
) -> EditMaskSet:
    """Mask one face in one photo from a *parsed* ATR human-parsing label map.

    ``label_map`` is the ``uint8`` class map :class:`~app.domains.portrait.
    human_parsing.SchpOnnxSegmenter` returns for the *same* photo, at the same
    HxW. Hair, clothing and the face/neck island are taken from the parse; the
    identity features and the glasses band come from the MediaPipe face mesh;
    the subject's own skin tone separates face skin from a fringe.

    The result is disjoint by construction: ``protected`` and ``editable`` are
    built so that no pixel is in both, which is what lets a downstream
    inpainting pipeline trust ``preserve`` as its keep-mask.
    """
    cfg = config or SemanticMaskConfig()
    lock_cfg = face_lock_config or FaceLockConfig()

    image = np.asarray(image_rgb)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("build_semantic_edit_masks expects an HxWx3 RGB image")
    height, width = int(image.shape[0]), int(image.shape[1])
    shape = (height, width)

    labels = np.asarray(label_map)
    if labels.ndim != 2:
        raise ValueError("label_map must be a 2-D class map")
    if labels.shape != shape:
        raise ValueError(
            f"label_map must be in the photo's own coordinates: got "
            f"{labels.shape}, expected {shape}"
        )
    if labels.dtype.kind not in "iu":
        raise ValueError("label_map must hold integer ATR class ids")
    top = int(labels.max()) if labels.size else 0
    if top >= CLASS_COUNT:
        raise ValueError(
            f"label_map holds class id {top}, outside the {CLASS_COUNT}-class "
            "ATR taxonomy"
        )

    lm = np.asarray(landmarks, dtype=np.float64)
    if lm.ndim != 2 or lm.shape[1] != 2:
        raise ValueError("landmarks must be an (N, 2) array of pixel coordinates")
    face_w = face_width(lm)
    if not np.isfinite(face_w) or face_w <= 0:
        raise ValueError("degenerate landmarks: face width is not positive")

    oval_pts = lm[list(FACE_OVAL)]
    x0, x1 = float(oval_pts[:, 0].min()), float(oval_pts[:, 0].max())
    y0, y1 = float(oval_pts[:, 1].min()), float(oval_pts[:, 1].max())
    face_h = max(y1 - y0, 1.0)

    gaps: list[str] = []

    # 1. Identity guard: eyes, brows, nose, lips and the teeth of an open smile.
    features = fill_feature_guard(shape, lm)
    features = np.maximum(features, fill_polygon(shape, _hull(lm[list(LIPS_INNER)])))
    features = dilate_mask(features, cfg.feature_grow_ratio * face_w)

    # 2. Spectacle guard. ATR reports "Sunglasses", and clear prescription
    #    glasses are parsed as Face/Hair, so this band is the only thing that
    #    keeps a lens and its wire from being repainted with the hairstyle.
    glasses = fill_polygon(shape, _hull(lm[list(GLASSES_BAND)]))
    glasses = dilate_mask(glasses, cfg.glasses_grow_ratio * face_w)

    # 3. Parsed face/neck island. ATR has no neck class, so the throat arrives
    #    as Face; the island is protected whole and grown a little, so a parse
    #    that stops a pixel short of the jaw cannot leave a rim of cheek
    #    editable, and a garment swap cannot repaint the throat. The face oval
    #    is unioned in so a sloppy parse edge cannot leave a sliver of cheek
    #    unprotected.
    parsed_face = _class_mask(labels, FACE_CLASSES)

    # 3a. The subject's own skin tone, sampled once. It is needed *before* the
    #     island is built, because one thing filed as Face on this photo is not
    #     anatomy: a high collar rises to the jaw and ATR has no collar class to
    #     put it in. Below the chin row that sample is what separates a garment
    #     from the throat; above it nothing is ever carved out.
    reference, sample_px = skin_reference_colour(image, lm, face_w)
    skin_mask = (
        adaptive_skin_mask(image, reference, face_w, lock_cfg)
        if reference is not None
        else None
    )

    # 3b. Split the parse at the chin. Parsed-Face pixels below the chin row
    #     that are *not* the subject's skin are a collar, not anatomy, and are
    #     handed to the clothing region so a new outfit can replace them.
    #     Below-chin pixels that *are* skin are the throat and stay protected.
    #     When the two cannot be told apart the island is kept whole and the gap
    #     is reported -- never silently called good.
    collar = np.zeros(shape, np.uint8)
    collar_pixels = 0
    neck_skin_pixels = 0
    dark_throat_pixels = 0
    collar_split_applied = False
    if cfg.collar_split_enabled:
        chin_y = float(lm[CHIN][1]) + cfg.collar_margin_ratio * face_w
        below_chin = np.zeros(shape, np.uint8)
        below_chin[max(int(round(chin_y)), 0) :, :] = 255
        below_face = cv2.bitwise_and(parsed_face, below_chin)
        if int((below_face > 0).sum()) > 0:
            if skin_mask is None:
                gaps.append(
                    "collar_split_unresolved: the parser filed pixels below the "
                    "chin as Face and no skin tone could be sampled to separate a "
                    "collar from the throat, so the island was protected whole -- "
                    "a garment at the neckline would not be editable. Fix the skin "
                    "sample or correct the mask by hand before a paid generation"
                )
            else:
                neck_skin_pixels = int(
                    (cv2.bitwise_and(below_face, skin_mask) > 0).sum()
                )
                candidate = cv2.bitwise_and(
                    below_face, cv2.bitwise_not(skin_mask)
                )
                # A shadowed throat is skin that the chroma tolerance rejects;
                # the darkest pixels under the jaw are that shadow, not cloth.
                # Handing them to the garment is what let a pass repaint the
                # underside of the chin, so they go back to the guard and are
                # counted as neck skin, not as a collar.
                if cfg.collar_skin_luma_ratio > 0:
                    luma = _luma(image)
                    ref_luma = float(
                        _luma(
                            np.asarray(reference, dtype=np.float32).reshape(1, 1, 3)
                        )[0, 0]
                    )
                    dark = np.where(
                        luma < cfg.collar_skin_luma_ratio * ref_luma, 255, 0
                    ).astype(np.uint8)
                    dark = cv2.bitwise_and(dark, below_face)
                    dark_throat_pixels = int(
                        (cv2.bitwise_and(candidate, dark) > 0).sum()
                    )
                    candidate = cv2.bitwise_and(candidate, cv2.bitwise_not(dark))
                    neck_skin_pixels += dark_throat_pixels
                candidate_pixels = int((candidate > 0).sum())
                if candidate_pixels >= cfg.collar_min_pixels:
                    collar = candidate
                    collar_pixels = candidate_pixels
                    collar_split_applied = True
                else:
                    gaps.append(
                        "collar_split_found_nothing: the parse reached below the "
                        f"chin but the non-skin mass there was {candidate_pixels} px "
                        f"(< {cfg.collar_min_pixels}), too small to be a collar; the "
                        "parsed face/neck island was protected whole"
                    )

    # The parsed hair is needed before the guard is closed: the subject's own
    # fringe lies *inside* the face oval, and unioning the oval in without
    # carving the hair back out froze that fringe as a dark band under the new
    # hairline -- the double edge a paid run printed on the forehead.
    parsed_hair = _class_mask(labels, (HAIR,))
    parsed_hair, hair_cleanup = _drop_speckle(parsed_hair, cfg)

    face_core = cv2.bitwise_and(parsed_face, cv2.bitwise_not(collar))
    oval = fill_polygon(shape, _hull(lm[list(FACE_OVAL)]))
    # The oval is unioned in last and the collar subtracted again, so the
    # carve-out cannot be undone by a chin landmark that sits a hair low.
    island = cv2.bitwise_and(
        np.maximum(face_core, oval), cv2.bitwise_not(collar)
    )
    # The guard is grown so a parse that stops a pixel short of the jaw cannot
    # leave a rim of cheek editable, and then the collar is subtracted again:
    # the rim must not grow *over* the garment we just measured, or the top of
    # the collar would be frozen right back onto the face. ``collar`` is all
    # zeros when the split did not apply, so this is a no-op on the old path.
    protected_face = cv2.bitwise_and(
        dilate_mask(island, cfg.face_grow_ratio * face_w),
        cv2.bitwise_not(collar),
    )
    # The fringe, eroded a hair's breadth. Eroding keeps a parse that spills a
    # few pixels onto the brow from opening bare forehead skin to the
    # generator, while still releasing the bulk of the fringe so the hair pass
    # can lay a new hairline over it instead of printing the old one.
    hair_guard_release = erode_mask(
        parsed_hair, cfg.hair_guard_release_erode_ratio * face_w
    )
    guard_release_pixels = int(
        (cv2.bitwise_and(protected_face, hair_guard_release) > 0).sum()
    )
    protected_face = cv2.bitwise_and(
        protected_face, cv2.bitwise_not(hair_guard_release)
    )

    # The subject's own skin tone is also kept as a quality signal on the parse
    # (faceSkinShare); it is not allowed to punch holes in the guard, because the
    # shadowed neck is skin that fails the test.
    if skin_mask is None:
        face_skin = island
        gaps.append(
            "skin_reference_unavailable: the face was too small or too stylised "
            "to sample its own skin tone, so the parsed face island was "
            "protected without a skin-tone cross-check"
        )
    else:
        face_skin = cv2.bitwise_and(island, skin_mask)
    island_pixels = max(int((island > 0).sum()), 1)
    face_skin_pixels = int((face_skin > 0).sum())

    protected = np.maximum(np.maximum(features, glasses), protected_face)

    # 3c. Recover the band the chin margin leaves behind. The split above starts
    #     at CHIN + collar_margin_ratio * face_w, so a garment that rises to the
    #     jaw keeps a margin-height strip frozen with the face. Measured against
    #     the reference photo that strip is the top edge of the old white knit
    #     collar: the last paid run could not reach it and printed it as a pale
    #     ghost. A pixel is released only when it is below the chin row, is not
    #     the subject's skin, is not the dark shadow under the jaw, and carries
    #     independent evidence of being a garment -- the parser's own
    #     Upper-clothes label, or a bright neutral highlight sitting on the
    #     collar's rim. Nothing else moves: the identity guard, the spectacle
    #     band and the skin tone are subtracted back out and the count that had
    #     to be dropped is reported.
    collar_recovery = np.zeros(shape, np.uint8)
    recovery_pixels = 0
    recovery_bright_pixels = 0
    recovery_parsed_garment_pixels = 0
    recovery_guard_refused_pixels = 0
    if (
        cfg.collar_split_enabled
        and cfg.collar_margin_recovery_enabled
        and skin_mask is not None
        and reference is not None
        and int((collar > 0).sum()) > 0
    ):
        chin_row = float(lm[CHIN][1])
        below_row = np.zeros(shape, np.uint8)
        below_row[max(int(round(chin_row)), 0) :, :] = 255
        ref_luma = float(
            _luma(np.asarray(reference, dtype=np.float32).reshape(1, 1, 3))[0, 0]
        )
        luma = _luma(image)
        chroma = _chroma(image)
        luma_floor = np.where(
            luma >= cfg.collar_skin_luma_ratio * ref_luma, 255, 0
        ).astype(np.uint8)
        parsed_garment = _class_mask(labels, CLOTHING_CLASSES)
        ring = dilate_mask(
            np.maximum(collar, parsed_garment),
            cfg.collar_margin_recovery_ring_ratio * face_w,
        )
        bright = np.where(
            (luma > cfg.collar_margin_recovery_luma_ratio * ref_luma)
            & (chroma < cfg.collar_margin_recovery_chroma),
            255,
            0,
        ).astype(np.uint8)
        evidence = np.maximum(
            cv2.bitwise_and(parsed_garment, ring),
            cv2.bitwise_and(bright, ring),
        )
        candidate = cv2.bitwise_and(protected, below_row)
        candidate = cv2.bitwise_and(candidate, cv2.bitwise_not(skin_mask))
        candidate = cv2.bitwise_and(candidate, luma_floor)
        candidate = cv2.bitwise_and(candidate, evidence)
        refused = cv2.bitwise_and(
            candidate, np.maximum(np.maximum(features, glasses), parsed_hair)
        )
        recovery_guard_refused_pixels = int((refused > 0).sum())
        candidate = cv2.bitwise_and(candidate, cv2.bitwise_not(refused))
        collar_recovery = candidate
        recovery_pixels = int((collar_recovery > 0).sum())
        if recovery_pixels > 0:
            recovery_bright_pixels = int(
                (cv2.bitwise_and(collar_recovery, bright) > 0).sum()
            )
            recovery_parsed_garment_pixels = int(
                (cv2.bitwise_and(collar_recovery, parsed_garment) > 0).sum()
            )
            gaps.append(
                "collar_margin_recovery_is_tone_based: the "
                f"{recovery_pixels} px the chin margin had frozen with the face "
                "were released to the clothing region on the parser's garment "
                "label and on a bright-neutral-fabric test; a skin-toned collar "
                "or a white sleeve in identical light is the one case that "
                "still needs an eye before a paid generation"
            )
            protected = cv2.bitwise_and(
                protected, cv2.bitwise_not(collar_recovery)
            )

    # 4. Hair: the parsed silhouette with its speckle dropped, grown by a
    #    bounded rim so a longer, fuller or differently shaped hairstyle is
    #    allowed, then cut back by the face guard so the forehead and jaw
    #    cannot be repainted.
    #    The rim is asymmetric: over the background it stays wide (a new
    #    silhouette must be free to grow), but over a parsed garment it is
    #    tightened, because a wide rim there handed the hair pass a band of
    #    collar to repaint and then the clothing pass repainted it again.
    parsed_clothing = _class_mask(labels, CLOTHING_CLASSES)
    parsed_clothing, clothing_cleanup = _drop_speckle(parsed_clothing, cfg)
    if collar_split_applied:
        parsed_clothing = np.maximum(parsed_clothing, collar)
    hair_over_background_rim = cv2.bitwise_and(
        dilate_mask(parsed_hair, cfg.hair_grow_ratio * face_w),
        cv2.bitwise_not(parsed_clothing),
    )
    hair_over_clothing_rim = cv2.bitwise_and(
        dilate_mask(parsed_hair, cfg.hair_rim_over_clothing_ratio * face_w),
        parsed_clothing,
    )
    hair = cv2.bitwise_and(
        np.maximum(hair_over_background_rim, hair_over_clothing_rim),
        cv2.bitwise_not(protected),
    )

    # 5. Clothing: the parsed garment(s) with their speckle dropped, grown a
    #    little for a new collar and shoulder line, minus everything protected
    #    (so skin, the neck and the face survive a garment swap).
    #    The collar carved out of the parsed face joined the garment region
    #    above, which is the whole point of the split: on the old mask it was
    #    frozen with the face, so a new outfit could never reach the neckline.
    #    Hair lying *on* the clothes is subtracted back out, so the two passes
    #    tile the seam instead of both repainting it: clothing owns the cloth,
    #    the hair pass -- which runs second and therefore has the final say --
    #    owns the strands and the thin overhang they cast on it.
    clothing_owner = cv2.bitwise_and(
        dilate_mask(parsed_clothing, cfg.clothing_grow_ratio * face_w),
        cv2.bitwise_not(
            dilate_mask(parsed_hair, cfg.clothing_hair_clearance_ratio * face_w)
        ),
    )
    clothing = cv2.bitwise_and(
        clothing_owner, cv2.bitwise_not(protected)
    )

    # 6. Drop speckle regions instead of shipping a mask that looks broken.
    for name, region in (("hair", hair), ("clothing", clothing)):
        pixels = int((region > 0).sum())
        if 0 < pixels < cfg.min_region_px:
            region[:] = 0
            gaps.append(
                f"{name}_region_below_floor: only {pixels} pixels survived the "
                f"face guard (< {cfg.min_region_px}); the region was dropped"
            )

    editable = np.maximum(hair, clothing)
    feather_px = cfg.feather_ratio * face_w

    background = _class_mask(labels, (0,))
    background_preserve = cv2.bitwise_and(background, cv2.bitwise_not(editable))

    editable_pixels = int((editable > 0).sum())
    overlap = int(cv2.bitwise_and(protected, editable).sum())
    background_editable = int(
        (cv2.bitwise_and(background, editable) > 0).sum()
    )

    for name, cleanup in (("hair", hair_cleanup), ("garment", clothing_cleanup)):
        if cleanup["droppedIslands"]:
            gaps.append(
                f"speckle_{name}_islands_dropped: the parser tagged "
                f"{cleanup['droppedIslands']} {name} island(s) below "
                f"{cleanup['floorPixels']} px (background slivers at the frame "
                f"edge, {cleanup['droppedPixels']} px in total); they were "
                "removed before the region was grown"
            )

    parsed_share = _class_shares(labels)
    if parsed_share.get(ATR_CLASSES[3], 0.0) <= 0.0:
        gaps.append(
            "glasses_guard_is_geometric: the parser reported no Sunglasses pixels "
            "(clear prescription glasses are not in the ATR taxonomy), so the "
            "lens band comes from the MediaPipe eye rings and is reported as "
            "geometry rather than as a classified region"
        )
    if len(CLOTHING_CLASSES) > 1 and parsed_share.get(ATR_CLASSES[4], 0.0) <= 0.0:
        gaps.append(
            "no_upper_clothing_parsed: the parser found no Upper-clothes region, "
            "so the clothing mask is empty and no garment may be restyled"
        )
    gaps.append(
        "parsing_boundary_is_approximate: the ATR network runs at 512x512 and is "
        "upsampled to the photo, so the hairline and the collar are accurate to "
        "roughly a few pixels -- they still need human correction before a paid "
        "generation is attempted"
    )
    gaps.append(
        "hairline_is_not_a_hair_matte: the hair mask is the parsed hair silhouette "
        "grown by a rim, not a per-strand matte; wispy strands against the "
        "background may fall outside it"
    )
    if collar_split_applied:
        gaps.append(
            "collar_split_is_tone_based: the pixels below the chin that were moved "
            "out of the face island and into the clothing region were separated by "
            "the subject's own skin tone, not by a garment class -- a shadowed "
            "throat or a skin-toned collar is the one case that needs an eye "
            "before a paid generation"
        )

    stats: Graph = {
        "imageWidth": width,
        "imageHeight": height,
        "faceBox": [round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2)],
        "faceWidthPx": round(face_w, 2),
        "faceHeightPx": round(face_h, 2),
        "featherPx": round(feather_px, 2),
        "skinSamplePixels": int(sample_px),
        "protectedPixels": int((protected > 0).sum()),
        "hairPixels": int((hair > 0).sum()),
        "clothingPixels": int((clothing > 0).sum()),
        "editablePixels": editable_pixels,
        "protectedShare": round(float((protected > 0).mean()), 4),
        "editableShare": round(float((editable > 0).mean()), 4),
        "identityGuardPixels": int((features > 0).sum()),
        "glassesGuardPixels": int((glasses > 0).sum()),
        "faceSkinPixels": face_skin_pixels,
        "parsedFacePixels": int((parsed_face > 0).sum()),
        "parsedFaceIslandPixels": island_pixels,
        "faceCorePixels": int((face_core > 0).sum()),
        "collarSplitEnabled": bool(cfg.collar_split_enabled),
        "collarSplitApplied": bool(collar_split_applied),
        "collarReclassifiedPixels": int(collar_pixels),
        "collarDarkThroatPixels": int(dark_throat_pixels),
        "collarEditablePixels": int(
            (cv2.bitwise_and(collar, editable) > 0).sum()
        ),
        "collarMarginRecoveryEnabled": bool(cfg.collar_margin_recovery_enabled),
        "collarMarginRecoveredPixels": recovery_pixels,
        "collarMarginRecoveredBrightPixels": recovery_bright_pixels,
        "collarMarginRecoveredParsedGarmentPixels": recovery_parsed_garment_pixels,
        "collarMarginRecoveryRefusedByGuardPixels": recovery_guard_refused_pixels,
        "collarMarginRecoveredEditablePixels": int(
            (cv2.bitwise_and(collar_recovery, editable) > 0).sum()
        ),
        "collarMarginRecoveredStillProtectedPixels": int(
            (cv2.bitwise_and(collar_recovery, protected) > 0).sum()
        ),
        "neckSkinProtectedPixels": int(neck_skin_pixels),
        "hairGuardReleasePixels": int(guard_release_pixels),
        "hairFrozenByGuardPixels": int(
            (cv2.bitwise_and(protected, parsed_hair) > 0).sum()
        ),
        "hairOverClothingRimPixels": int(
            (cv2.bitwise_and(hair, parsed_clothing) > 0).sum()
        ),
        "clothingOverParsedHairPixels": int(
            (cv2.bitwise_and(clothing, parsed_hair) > 0).sum()
        ),
        "faceSkinShare": round(face_skin_pixels / island_pixels, 4),
        "backgroundPreservePixels": int((background_preserve > 0).sum()),
        "backgroundEditablePixels": background_editable,
        "backgroundEditableShare": round(
            float(background_editable) / float(max(editable_pixels, 1)), 4
        ),
        "hairIslandsDropped": hair_cleanup["droppedIslands"],
        "clothingIslandsDropped": clothing_cleanup["droppedIslands"],
        "specklePixelsDropped": int(
            hair_cleanup["droppedPixels"] + clothing_cleanup["droppedPixels"]
        ),
        "protectedCoversEditable": bool(overlap == 0),
        "identityCoversEditable": bool(
            cv2.bitwise_and(features, editable).sum() == 0
        ),
        "hairContactWithProtectedPixels": int(
            cv2.bitwise_and(dilate_mask(hair, 1.0), protected).sum() // 255
        ),
        "parsedClassShares": parsed_share,
        "semantic": True,
    }
    stats.update(_legacy_ellipse_comparison(hair, lm, shape))

    return EditMaskSet(
        protected=protected,
        editable=editable,
        face_width_px=float(face_w),
        feather_px=float(feather_px),
        stats=stats,
        gaps=tuple(gaps),
        hair=hair,
        clothing=clothing,
        background_preserve=background_preserve,
        label_map=labels.astype(np.uint8, copy=False),
        parsed_face=parsed_face,
        face_core=face_core,
        collar=collar,
    )


def _class_mask(labels: Any, classes: tuple[int, ...]) -> Any:
    """Binary mask of the pixels whose parsed class is one of *classes*."""
    mask = np.zeros(labels.shape, np.uint8)
    for class_id in classes:
        mask[labels == class_id] = 255
    return mask


def _luma(image_rgb: Any) -> Any:
    """Rec.601 luminance of an RGB image, in the 8-bit coding range.

    The same coefficients :func:`~app.domains.portrait.face_lock._ycc` uses, so
    the collar's shadow test agrees with the skin test that rejected the pixel.
    """
    f = np.asarray(image_rgb, dtype=np.float32)
    return 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]


def _chroma(image_rgb: Any) -> Any:
    """Max-minus-min channel of an RGB image: saturation on an 8-bit scale.

    Skin keeps a wide red-to-blue spread even in shadow, so a bright *and*
    narrow-spread pixel is a fabric highlight rather than lit skin.
    """
    f = np.asarray(image_rgb, dtype=np.float32)
    return f.max(axis=-1) - f.min(axis=-1)


def _drop_speckle(
    mask: Any, cfg: SemanticMaskConfig
) -> tuple[Any, dict[str, int]]:
    """Keep the significant islands of one parsed class mask, drop the speckle.

    ATR occasionally tags a sliver of background at the frame edge as hair or
    as a garment. Grown by a rim it would open a patch of background to the
    generator, so an island smaller than ``cfg.component_floor_ratio`` of the
    largest island (and smaller than ``cfg.min_region_px`` in absolute terms)
    is removed. A single subject always has one dominant island, so this cannot
    delete a real limb; what it removes is reported to the caller.
    """
    blank = {"droppedIslands": 0, "droppedPixels": 0, "floorPixels": 0}
    if not mask.any():
        return mask, blank
    count, components, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count <= 2:  # background + a single island: nothing to prune
        return mask, {**blank, "floorPixels": int(stats[1, cv2.CC_STAT_AREA])}

    areas = stats[1:, cv2.CC_STAT_AREA]
    floor = max(int(cfg.min_region_px), int(round(cfg.component_floor_ratio * int(areas.max()))))
    kept = np.zeros_like(mask)
    dropped_islands = 0
    dropped_pixels = 0
    for index, area in enumerate(areas, start=1):
        if int(area) >= floor:
            kept[components == index] = 255
        else:
            dropped_islands += 1
            dropped_pixels += int(area)
    return kept, {
        "droppedIslands": dropped_islands,
        "droppedPixels": dropped_pixels,
        "floorPixels": floor,
    }


def _class_shares(labels: Any) -> dict[str, float]:
    """Named share of the frame per ATR class that actually occurs."""
    values, counts = np.unique(np.asarray(labels), return_counts=True)
    total = float(max(labels.size, 1))
    shares: dict[str, float] = {}
    for value, count in zip(values.tolist(), counts.tolist()):
        if 0 <= int(value) < CLASS_COUNT:
            shares[ATR_CLASSES[int(value)]] = round(float(count) / total, 4)
    return shares


def _legacy_ellipse_comparison(
    hair: Any, landmarks: Any, shape: tuple[int, int]
) -> dict[str, float]:
    """Measure the parsed hair against the P2.6 head ellipse, for the report.

    "The hair region is no longer a big circle" is a claim, so it is quantified:
    the circle the P2.6 builder would have drawn is reconstructed from the same
    landmarks and its overlap with the parsed hair is reported.
    """
    legacy = EditMaskConfig()
    oval_pts = np.asarray(landmarks)[list(FACE_OVAL)]
    x0, x1 = float(oval_pts[:, 0].min()), float(oval_pts[:, 0].max())
    y0, y1 = float(oval_pts[:, 1].min()), float(oval_pts[:, 1].max())
    face_h = max(y1 - y0, 1.0)
    face_w = face_width(landmarks)
    ellipse = _ellipse_mask(
        shape,
        center=(0.5 * (x0 + x1), y0 + 0.42 * face_h),
        axes=(
            0.5 * (x1 - x0) + legacy.hair_side_ratio * face_w,
            0.55 * face_h + legacy.hair_up_ratio * face_w,
        ),
    )
    hair_bin = hair > 0
    ellipse_bin = ellipse > 0
    intersection = int(np.logical_and(hair_bin, ellipse_bin).sum())
    union = int(np.logical_or(hair_bin, ellipse_bin).sum())
    outside = int(np.logical_and(hair_bin, np.logical_not(ellipse_bin)).sum())
    hair_px = max(int(hair_bin.sum()), 1)
    return {
        "legacyEllipseStyle": "P2.6 head ellipse reconstructed from the same landmarks",
        "legacyEllipsePixels": int(ellipse_bin.sum()),
        "hairIoUWithLegacyEllipse": round(intersection / max(union, 1), 4),
        "hairOutsideLegacyEllipseShare": round(outside / hair_px, 4),
    }


def _hull(points: Any) -> Any:
    return cv2.convexHull(np.asarray(points, dtype=np.float32))


def _ellipse_mask(
    shape: tuple[int, int], center: tuple[float, float], axes: tuple[float, float]
) -> Any:
    mask = np.zeros(shape, np.uint8)
    cv2.ellipse(
        mask,
        (int(round(center[0])), int(round(center[1]))),
        (max(int(round(axes[0])), 1), max(int(round(axes[1])), 1)),
        0.0,
        0.0,
        360.0,
        255,
        thickness=-1,
    )
    return mask
