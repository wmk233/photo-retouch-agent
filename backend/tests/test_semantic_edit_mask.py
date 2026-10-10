"""The parsed (P2.7) mask path: real ATR label maps, no geometry stand-ins.

These tests never touch onnxruntime: the label map is synthesised the way the
parser would emit one, so the *mask builder* can be exercised on its own. The
point of the suite is that the builder is seam-correct on a parsed input --
regions disjoint, identity guarded, speckle pruned -- and that it refuses a
label map that is not in the photo's own coordinates instead of quietly
drawing the P2.6 ellipse.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.domains.portrait.edit_mask import (
    GLASSES_BAND,
    EditMaskConfig,
    SemanticMaskConfig,
    build_edit_masks,
    build_semantic_edit_masks,
)
from app.domains.portrait.face_lock import (
    BROW_A,
    BROW_B,
    CHIN,
    EYE_A,
    EYE_B,
    FACE_OVAL,
    FACE_SIDES,
    LIPS_INNER,
    NOSE,
)
from app.domains.portrait.human_parsing import (
    BACKGROUND,
    FACE,
    HAIR,
    UPPER_CLOTHES,
)

IMAGE_HEIGHT = 480
IMAGE_WIDTH = 360
FACE_CX = 180.0
FACE_CY = 200.0
FACE_RX = 60.0
FACE_RY = 80.0
SKIN = np.array([200, 170, 150], dtype=np.uint8)

#: Where the synthetic parse puts each region. The hair band sits above the
#: face oval's top edge, the face/neck island over the oval, and the garment
#: well below the chin, so a mix-up between classes cannot hide.
HAIR_ROWS = (40, 118)
HAIR_COLS = (110, 250)
FACE_ROWS = (120, 280)
FACE_COLS = (120, 240)
CLOTHING_ROWS = (300, 470)
CLOTHING_COLS = (70, 290)

#: A high collar rises to the jaw and ATR has no collar class for it, so the
#: parser files it as Face. Here the band below the chin is added to the Face
#: island: a cool garment white (not this subject's skin tone) on the outside
#: and a skin-toned throat strip down the middle.
COLLAR_ROWS = (280, 330)
COLLAR_COLS = (120, 240)
NECK_COLS = (170, 200)
COLLAR_COLOUR = np.array([232, 238, 248], dtype=np.uint8)

#: The real mesh keeps these on the bridge of the nose and at the outer corners
#: of the eyes. The synthetic ``FACE_OVAL`` ring above reuses the very same
#: indices, so they are pinned back to eye height afterwards: without that the
#: glasses hull is the convex hull of the whole oval and the "lens band" guard
#: swallows the forehead, which is a fixture artefact and not the real guard.
GLASSES_BRIDGE = (5, 4, 6, 168, 195, 197)
GLASSES_SIDES_LEFT = (127, 234, 93)
GLASSES_SIDES_RIGHT = (356, 454, 323)


def _ring(points: np.ndarray, indices, center, rx: float, ry: float) -> None:
    count = len(indices)
    for position, index in enumerate(indices):
        angle = 2.0 * np.pi * position / count
        points[index] = [
            center[0] + rx * np.cos(angle),
            center[1] + ry * np.sin(angle),
        ]


def _landmarks() -> np.ndarray:
    """A 478-point mesh shaped like a face -- no photo, no model call."""
    points = np.tile([FACE_CX, FACE_CY], (478, 1)).astype(np.float64)
    _ring(points, FACE_OVAL, (FACE_CX, FACE_CY), FACE_RX, FACE_RY)
    points[FACE_SIDES[0]] = [FACE_CX - FACE_RX, FACE_CY]
    points[FACE_SIDES[1]] = [FACE_CX + FACE_RX, FACE_CY]
    _ring(points, EYE_A, (FACE_CX - 26.0, FACE_CY - 22.0), 14.0, 7.0)
    _ring(points, EYE_B, (FACE_CX + 26.0, FACE_CY - 22.0), 14.0, 7.0)
    _ring(points, BROW_A, (FACE_CX - 26.0, FACE_CY - 34.0), 15.0, 4.0)
    _ring(points, BROW_B, (FACE_CX + 26.0, FACE_CY - 34.0), 15.0, 4.0)
    _ring(points, NOSE, (FACE_CX, FACE_CY + 8.0), 8.0, 14.0)
    _ring(points, LIPS_INNER, (FACE_CX, FACE_CY + 44.0), 12.0, 6.0)
    points[CHIN] = [FACE_CX, FACE_CY + FACE_RY]
    for index in GLASSES_BRIDGE:
        points[index] = [FACE_CX, FACE_CY - 26.0]
    for index in GLASSES_SIDES_LEFT:
        points[index] = [FACE_CX - 40.0, FACE_CY - 22.0]
    for index in GLASSES_SIDES_RIGHT:
        points[index] = [FACE_CX + 40.0, FACE_CY - 22.0]
    return points


def _image() -> np.ndarray:
    image = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
    image[:, :] = SKIN
    return image


def _labels() -> np.ndarray:
    labels = np.full((IMAGE_HEIGHT, IMAGE_WIDTH), BACKGROUND, dtype=np.uint8)
    labels[HAIR_ROWS[0] : HAIR_ROWS[1], HAIR_COLS[0] : HAIR_COLS[1]] = HAIR
    labels[FACE_ROWS[0] : FACE_ROWS[1], FACE_COLS[0] : FACE_COLS[1]] = FACE
    labels[
        CLOTHING_ROWS[0] : CLOTHING_ROWS[1], CLOTHING_COLS[0] : CLOTHING_COLS[1]
    ] = UPPER_CLOTHES
    return labels


def _class_mask(labels: np.ndarray, class_id: int) -> np.ndarray:
    mask = np.zeros(labels.shape, np.uint8)
    mask[labels == class_id] = 255
    return mask


@pytest.fixture()
def mask_set():
    return build_semantic_edit_masks(_image(), _landmarks(), _labels())


# ---------------------------------------------------------------- provenance


def test_semantic_flag_marks_a_parsed_mask(mask_set):
    assert mask_set.semantic is True
    assert mask_set.label_map is not None
    assert mask_set.label_map.dtype == np.uint8
    assert np.array_equal(mask_set.label_map, _labels())
    assert mask_set.stats["semantic"] is True
    assert mask_set.parsed_face is not None


def test_geometric_path_stays_non_semantic():
    """The P2.6 builder is untouched: no label map, no semantic regions."""
    legacy = build_edit_masks(_image(), _landmarks())
    assert legacy.semantic is False
    assert legacy.label_map is None
    assert legacy.hair is None and legacy.clothing is None
    assert "hair_band_is_geometric" in " ".join(legacy.gaps)


# ------------------------------------------------------------------- regions


def test_parsed_regions_are_open(mask_set):
    assert mask_set.stats["hairPixels"] > 0
    assert mask_set.stats["clothingPixels"] > 0
    assert mask_set.stats["editablePixels"] > 0
    assert mask_set.editable.max() == 255
    assert mask_set.stats["parsedFacePixels"] > 0
    assert mask_set.stats["parsedFaceIslandPixels"] >= mask_set.stats["parsedFacePixels"]


def test_hair_follows_the_parse_and_not_a_circle(mask_set):
    """Hair must live on the parsed hair, and must not be the P2.6 ellipse."""
    hair = mask_set.hair
    parsed = _class_mask(_labels(), HAIR)
    inside_parse = cv2.bitwise_and(hair, parsed)
    assert int((inside_parse > 0).sum()) > 0
    assert mask_set.stats["hairIoUWithLegacyEllipse"] < 0.95
    assert mask_set.stats["hairOutsideLegacyEllipseShare"] > 0.05


def test_clothing_stays_below_the_face(mask_set):
    parsed_face = _class_mask(_labels(), FACE)
    assert int(cv2.bitwise_and(mask_set.clothing, parsed_face).sum()) == 0
    centre_row = (CLOTHING_ROWS[0] + CLOTHING_ROWS[1]) // 2
    centre_col = (CLOTHING_COLS[0] + CLOTHING_COLS[1]) // 2
    assert mask_set.clothing[centre_row, centre_col] == 255


def test_background_preserve_excludes_the_edit_area(mask_set):
    background = _class_mask(_labels(), BACKGROUND)
    overlap = cv2.bitwise_and(mask_set.background_preserve, mask_set.editable)
    assert int(overlap.sum()) == 0
    assert int(cv2.bitwise_and(mask_set.background_preserve, background)[470, 5]) == 255
    assert 0.0 <= mask_set.stats["backgroundEditableShare"] < 0.35
    assert mask_set.stats["backgroundEditablePixels"] >= 0


def test_editable_and_protected_are_disjoint(mask_set):
    overlap = cv2.bitwise_and(mask_set.editable, mask_set.protected)
    assert int(overlap.sum()) == 0
    assert mask_set.stats["protectedCoversEditable"] is True
    assert np.array_equal(mask_set.preserve, cv2.bitwise_not(mask_set.editable))


def test_face_guard_covers_the_lock_landmarks(mask_set):
    landmarks = _landmarks()
    protected = mask_set.protected
    for group in (EYE_A, EYE_B, BROW_A, BROW_B, NOSE, LIPS_INNER):
        for index in group:
            x, y = landmarks[index]
            assert protected[int(round(y)), int(round(x))] > 0, index
    centre = landmarks[list(LIPS_INNER)].mean(axis=0)
    assert protected[int(round(centre[1])), int(round(centre[0]))] > 0


def test_glasses_band_is_protected_from_geometry(mask_set):
    """ATR has no clear-glasses class, so the lens band comes from the mesh."""
    landmarks = _landmarks()
    for index in GLASSES_BAND:
        x, y = landmarks[index]
        assert mask_set.protected[int(round(y)), int(round(x))] > 0, index
    assert mask_set.stats["glassesGuardPixels"] > 0
    assert "glasses_guard_is_geometric" in " ".join(mask_set.gaps)


def test_the_synthetic_lens_band_is_a_band_and_not_the_whole_oval(mask_set):
    """The pinned indices keep the fixture honest about the real guard."""
    pinned = set(GLASSES_BRIDGE + GLASSES_SIDES_LEFT + GLASSES_SIDES_RIGHT)
    assert pinned <= set(GLASSES_BAND)
    # A degenerate hull would make the oval itself the guard and freeze the
    # whole forehead, so the band must stay a small share of the parsed face.
    assert mask_set.stats["glassesGuardPixels"] < 0.35 * mask_set.stats[
        "parsedFacePixels"
    ]


def test_a_new_hairstyle_may_reach_outside_the_parsed_hair(mask_set):
    parsed = _class_mask(_labels(), HAIR)
    rim = cv2.bitwise_and(mask_set.hair, cv2.bitwise_not(parsed))
    assert int((rim > 0).sum()) > 0


def _hair_lying_on_a_garment():
    """Hair with a garment directly beneath it: the seam that printed the ghost.

    The base layout keeps hair and garment far apart, so a stray rim cannot be
    seen. Here the parsed garment starts on the row the hair ends, which is the
    geometry the pale band under the chin came from: a wide rim handed the hair
    pass a band of collar and the clothing pass repainted it afterwards.
    """
    hair_rows, hair_cols = (40, 118), (270, 350)
    garment_rows, garment_cols = (118, 320), (270, 350)
    labels = np.full((IMAGE_HEIGHT, IMAGE_WIDTH), BACKGROUND, dtype=np.uint8)
    labels[hair_rows[0] : hair_rows[1], hair_cols[0] : hair_cols[1]] = HAIR
    labels[garment_rows[0] : garment_rows[1], garment_cols[0] : garment_cols[1]] = UPPER_CLOTHES
    labels[FACE_ROWS[0] : FACE_ROWS[1], FACE_COLS[0] : FACE_COLS[1]] = FACE
    return _image(), labels


def test_the_hair_rim_is_tightened_where_it_overlaps_a_garment():
    """The wide rim is for the background. Over cloth it is pulled right in.

    Both variants are the same scene and the same builder; the only difference
    is the rim allowed over a parsed garment, so the pixels *taken* from the
    garment are attributable to that one knob.
    """
    image, labels = _hair_lying_on_a_garment()
    tight = build_semantic_edit_masks(image, _landmarks(), labels)
    wide = build_semantic_edit_masks(
        image,
        _landmarks(),
        labels,
        config=SemanticMaskConfig(
            hair_rim_over_clothing_ratio=SemanticMaskConfig().hair_grow_ratio
        ),
    )

    assert tight.stats["hairOverClothingRimPixels"] > 0
    assert tight.stats["hairOverClothingRimPixels"] < wide.stats["hairOverClothingRimPixels"]
    # A hair's breadth of overhang is still allowed, so the seam is not a hard cut.
    assert tight.stats["hairOverClothingRimPixels"] < 0.25 * wide.stats[
        "hairOverClothingRimPixels"
    ]


def test_clothing_never_reaches_back_over_the_parsed_hair():
    """One pass owns the seam: the garment stops at the parse, the hair laps it.

    The hair pass runs second, so it -- not the garment -- has the final say on
    the strand-on-cloth seam. The garment is carved back to the parsed hair
    boundary and never repaints a strand. To keep that zero from being vacuous
    (two masks that simply never touch) the same scene is asserted to have the
    hair pass reaching over the cloth, and a wider clearance is checked to only
    ever take pixels *away* from the garment.
    """
    image, labels = _hair_lying_on_a_garment()
    parsed_hair = _class_mask(labels, HAIR)
    parsed_clothing = _class_mask(labels, UPPER_CLOTHES)
    default = build_semantic_edit_masks(image, _landmarks(), labels)

    assert default.stats["clothingOverParsedHairPixels"] == 0
    assert int((cv2.bitwise_and(default.clothing, parsed_hair) > 0).sum()) == 0
    # The seam is real and owned by exactly one pass: hair paints over cloth.
    assert int((cv2.bitwise_and(default.hair, parsed_clothing) > 0).sum()) > 0

    wider = build_semantic_edit_masks(
        image,
        _landmarks(),
        labels,
        config=SemanticMaskConfig(clothing_hair_clearance_ratio=0.06),
    )
    assert wider.stats["clothingOverParsedHairPixels"] == 0
    assert int((cv2.bitwise_and(wider.clothing, parsed_hair) > 0).sum()) == 0
    assert wider.stats["clothingPixels"] <= default.stats["clothingPixels"]


def _fringe_scene():
    """The subject's own dark fringe lying inside the face oval.

    This is the real photo's geometry: MediaPipe's oval reaches over the hair,
    so unioning the oval in without carving the parse back out froze the fringe
    and printed a second hairline under the new one.
    """
    labels = np.full((IMAGE_HEIGHT, IMAGE_WIDTH), BACKGROUND, dtype=np.uint8)
    labels[FACE_ROWS[0] : FACE_ROWS[1], FACE_COLS[0] : FACE_COLS[1]] = FACE
    labels[105:155, 140:220] = HAIR
    return _image(), labels


def test_the_parsed_fringe_is_released_so_a_new_hairline_has_one_edge():
    image, labels = _fringe_scene()
    released = build_semantic_edit_masks(image, _landmarks(), labels)
    frozen = build_semantic_edit_masks(
        image,
        _landmarks(),
        labels,
        config=SemanticMaskConfig(hair_guard_release_erode_ratio=0.30),
    )

    # By default the parsed fringe comes out of the face guard, so the hair pass
    # paints a new hairline over it instead of leaving the old one underneath.
    assert released.stats["hairGuardReleasePixels"] > 0
    assert released.stats["hairFrozenByGuardPixels"] == 0
    # Only the parsed hair can be freed, never parsed skin, so the guard still
    # cannot be punched open.
    assert released.stats["protectedCoversEditable"] is True
    assert released.stats["identityCoversEditable"] is True
    # Turning the release down to a large erode is the old behaviour -- the
    # fringe stays inside the guard -- and it shows up as frozen hair under the
    # new hairline: the double edge. The knob has to move it, so the release is
    # strictly smaller and the frozen fringe comes back.
    assert frozen.stats["hairGuardReleasePixels"] < released.stats[
        "hairGuardReleasePixels"
    ]
    assert frozen.stats["hairFrozenByGuardPixels"] > 0


def test_face_skin_share_is_reported_as_a_quality_signal(mask_set):
    assert mask_set.stats["faceSkinShare"] > 0.5


def test_gaps_flag_the_approximate_boundary(mask_set):
    joined = " ".join(mask_set.gaps)
    assert "parsing_boundary_is_approximate" in joined
    assert "hairline_is_not_a_hair_matte" in joined


# ------------------------------------------------------------------ speckle


def _labels_with_corner_speckle() -> np.ndarray:
    labels = _labels()
    labels[0:3, 0:3] = HAIR
    return labels


def test_speckle_island_is_dropped_and_reported():
    result = build_semantic_edit_masks(
        _image(), _landmarks(), _labels_with_corner_speckle()
    )
    assert result.hair[0:20, 0:20].max() == 0
    assert result.stats["hairIslandsDropped"] >= 1
    assert result.stats["specklePixelsDropped"] >= 9
    assert "speckle_hair_islands_dropped" in " ".join(result.gaps)


def test_speckle_survives_when_the_floor_is_disabled():
    """Control: with the floor off the same sliver *does* open the corner."""
    config = SemanticMaskConfig(min_region_px=1, component_floor_ratio=0.0)
    result = build_semantic_edit_masks(
        _image(), _landmarks(), _labels_with_corner_speckle(), config=config
    )
    assert result.hair[0:20, 0:20].max() == 255
    assert "speckle_hair_islands_dropped" not in " ".join(result.gaps)


def test_region_below_floor_is_dropped_and_reported():
    config = SemanticMaskConfig(min_region_px=10 ** 9)
    result = build_semantic_edit_masks(_image(), _landmarks(), _labels(), config=config)
    assert result.stats["hairPixels"] == 0
    assert result.stats["clothingPixels"] == 0
    joined = " ".join(result.gaps)
    assert "hair_region_below_floor" in joined
    assert "clothing_region_below_floor" in joined


# -------------------------------------------------------------------- errors


def test_rejects_a_label_map_of_the_wrong_size():
    with pytest.raises(ValueError, match="photo's own coordinates"):
        build_semantic_edit_masks(_image(), _landmarks(), np.zeros((10, 10), np.uint8))


def test_rejects_a_class_id_outside_the_taxonomy():
    labels = _labels()
    labels[400, 5] = 25
    with pytest.raises(ValueError, match="ATR taxonomy"):
        build_semantic_edit_masks(_image(), _landmarks(), labels)


def test_rejects_a_non_integer_label_map():
    labels = _labels().astype(np.float32)
    with pytest.raises(ValueError, match="integer ATR class ids"):
        build_semantic_edit_masks(_image(), _landmarks(), labels)


def test_rejects_a_non_2d_label_map():
    with pytest.raises(ValueError, match="2-D class map"):
        build_semantic_edit_masks(_image(), _landmarks(), np.zeros((4, 4, 3), np.uint8))


def test_rejects_non_photo_input():
    with pytest.raises(ValueError, match="HxWx3"):
        build_semantic_edit_masks(
            np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), np.uint8), _landmarks(), _labels()
        )


def test_rejects_degenerate_landmarks():
    flat = np.zeros((478, 2), dtype=np.float64)
    with pytest.raises(ValueError, match="degenerate"):
        build_semantic_edit_masks(_image(), flat, _labels())


def test_legacy_config_is_still_available():
    """P2.6 compatibility: the geometric knobs still exist and build a set."""
    legacy = build_edit_masks(_image(), _landmarks(), config=EditMaskConfig())
    assert legacy.stats["hairPixels"] > 0


# ---------------------------------------------------------------- collar split


def _collar_image() -> np.ndarray:
    """A white collar the parser mislabels as Face, with a real throat strip."""
    image = _image().copy()
    image[
        COLLAR_ROWS[0] : COLLAR_ROWS[1], COLLAR_COLS[0] : COLLAR_COLS[1]
    ] = COLLAR_COLOUR
    image[COLLAR_ROWS[0] : COLLAR_ROWS[1], NECK_COLS[0] : NECK_COLS[1]] = SKIN
    return image


def _collar_labels() -> np.ndarray:
    labels = _labels().copy()
    labels[
        COLLAR_ROWS[0] : COLLAR_ROWS[1], COLLAR_COLS[0] : COLLAR_COLS[1]
    ] = FACE
    return labels


def test_white_collar_below_the_chin_becomes_editable_clothing():
    """The old collar must be repaintable, and the throat beside it must not."""
    result = build_semantic_edit_masks(_collar_image(), _landmarks(), _collar_labels())

    assert result.stats["collarSplitEnabled"] is True
    assert result.stats["collarSplitApplied"] is True
    assert result.stats["collarReclassifiedPixels"] > 0
    # Everything carved out of the face island lands in the edit area.
    assert result.stats["collarEditablePixels"] == result.stats["collarReclassifiedPixels"]
    # The skin-toned throat inside the same band stays protected.
    assert result.stats["neckSkinProtectedPixels"] > 0
    assert result.stats["identityCoversEditable"] is True
    assert result.stats["protectedCoversEditable"] is True
    assert 0.0 <= result.stats["backgroundEditableShare"] < 0.35

    # A collar pixel (left of the throat) is white in the edit mask...
    collar_row = COLLAR_ROWS[0] + 5
    collar_col = COLLAR_COLS[0] + 10
    assert result.editable[collar_row, collar_col] == 255
    assert result.protected[collar_row, collar_col] == 0
    # ...while the throat strip in the very same band is kept pixel-for-pixel.
    neck_col = (NECK_COLS[0] + NECK_COLS[1]) // 2
    assert result.editable[collar_row + 5, neck_col] == 0
    assert result.protected[collar_row + 5, neck_col] == 255

    # The separation is tone-based, and the builder says so instead of claiming
    # a garment-class certainty it does not have.
    assert "collar_split_is_tone_based" in " ".join(result.gaps)


def test_collar_that_cannot_be_told_from_the_throat_stays_protected():
    """No silent pass: an unseparated collar is protected whole and flagged."""
    config = SemanticMaskConfig(collar_min_pixels=10 ** 9)
    result = build_semantic_edit_masks(
        _collar_image(), _landmarks(), _collar_labels(), config=config
    )

    assert result.stats["collarSplitApplied"] is False
    assert result.stats["collarReclassifiedPixels"] == 0
    assert "collar_split_found_nothing" in " ".join(result.gaps)
    # The disputed pixels stay on the keep side rather than being guessed at.
    collar_row = COLLAR_ROWS[0] + 5
    collar_col = COLLAR_COLS[0] + 10
    assert result.editable[collar_row, collar_col] == 0
    assert result.protected[collar_row, collar_col] == 255


def test_collar_split_reports_when_no_skin_tone_can_be_sampled():
    """No sample of the subject's skin -> no split, and an explicit gap."""
    tiny = np.array([FACE_CX, FACE_CY])
    landmarks = tiny + (_landmarks() - tiny) * 0.05
    result = build_semantic_edit_masks(_collar_image(), landmarks, _collar_labels())

    assert result.stats["collarSplitApplied"] is False
    assert "collar_split_unresolved" in " ".join(result.gaps)
    collar_row = COLLAR_ROWS[0] + 5
    collar_col = COLLAR_COLS[0] + 10
    assert result.editable[collar_row, collar_col] == 0
    assert result.protected[collar_row, collar_col] == 255


def _shadowed_throat_image() -> np.ndarray:
    """The same band, but the garment is in deep shadow under the jaw.

    The shadow fails the chroma skin test the same way a dark garment does, so
    without a luminance floor it was carved out as cloth and repainted -- the
    grey cast on the underside of the chin.
    """
    image = _collar_image().copy()
    image[
        COLLAR_ROWS[0] : COLLAR_ROWS[1], COLLAR_COLS[0] : COLLAR_COLS[1]
    ] = np.array([40, 36, 42], dtype=np.uint8)
    image[COLLAR_ROWS[0] : COLLAR_ROWS[1], NECK_COLS[0] : NECK_COLS[1]] = SKIN
    return image


def test_a_shadowed_throat_stays_protected_instead_of_becoming_a_collar():
    bright = build_semantic_edit_masks(_collar_image(), _landmarks(), _collar_labels())
    shadow = build_semantic_edit_masks(
        _shadowed_throat_image(), _landmarks(), _collar_labels()
    )

    # A genuinely white collar has no dark pixels to reclaim.
    assert bright.stats["collarDarkThroatPixels"] == 0
    assert bright.stats["collarEditablePixels"] > 0

    # A shadowed band is claimed back as neck skin, so nothing is carved and the
    # island is protected whole rather than handed to the garment.
    assert shadow.stats["collarDarkThroatPixels"] > 0
    assert shadow.stats["collarReclassifiedPixels"] == 0
    assert shadow.stats["collarSplitApplied"] is False
    collar_row = COLLAR_ROWS[0] + 5
    collar_col = COLLAR_COLS[0] + 10
    assert shadow.editable[collar_row, collar_col] == 0
    assert shadow.protected[collar_row, collar_col] == 255


# ------------------------------------------------- the chin-margin collar strip


def test_a_collar_that_rises_to_the_jaw_is_not_frozen_by_the_split_margin():
    """The split starts a margin below the chin, so the strip it skips stayed frozen.

    On the reference photo that strip is the top edge of the old white knit
    collar: ~805 px the last paid run could never repaint, printed as a pale
    ghost under the chin. The recovery knob releases the well-evidenced part of
    it -- a bright, neutral fabric beside the collar -- into the clothing region.
    """
    wide_margin = 0.06
    frozen = build_semantic_edit_masks(
        _collar_image(),
        _landmarks(),
        _collar_labels(),
        config=SemanticMaskConfig(
            collar_margin_ratio=wide_margin, collar_margin_recovery_enabled=False
        ),
    )
    released = build_semantic_edit_masks(
        _collar_image(),
        _landmarks(),
        _collar_labels(),
        config=SemanticMaskConfig(collar_margin_ratio=wide_margin),
    )

    chin_row = int(round(FACE_CY + FACE_RY))
    # The rows the chin margin freezes: from the chin down to where the split
    # line begins. The release is evidence-gated, so it lifts only the rows that
    # carry independent garment evidence -- the rest of the strip stays frozen.
    strip = slice(chin_row, chin_row + 5)
    collar_col = COLLAR_COLS[0] + 10

    frozen_strip = frozen.protected[strip, collar_col]
    released_strip = released.protected[strip, collar_col]

    # Without the knob the whole strip between the chin and the split line is
    # frozen and uneditable.
    assert frozen_strip.min() == 255
    assert frozen.editable[strip, collar_col].max() == 0

    # With it, the well-evidenced rows of that same strip leave the guard and
    # land in the clothing region...
    freed = (frozen_strip == 255) & (released_strip == 0)
    assert freed.any()
    assert released.editable[strip, collar_col][freed].min() == 255
    assert released.clothing[strip, collar_col][freed].min() == 255
    # ...while the rows with no such evidence keep the guard, so the carve stays
    # measured instead of dropping the whole chin margin on the floor.
    assert not freed.all()

    recovered = released.stats["collarMarginRecoveredPixels"]
    assert recovered > 0
    assert recovered == int((frozen.protected > 0).sum()) - int(
        (released.protected > 0).sum()
    )
    assert released.stats["collarMarginRecoveredEditablePixels"] == recovered
    assert released.stats["collarMarginRecoveredStillProtectedPixels"] == 0
    assert "collar_margin_recovery_is_tone_based" in " ".join(released.gaps)


def test_collar_margin_recovery_never_frees_skin_the_face_or_the_glasses():
    """The release is narrow by construction: it cannot touch a guarded pixel."""
    frozen = build_semantic_edit_masks(
        _collar_image(),
        _landmarks(),
        _collar_labels(),
        config=SemanticMaskConfig(
            collar_margin_ratio=0.06, collar_margin_recovery_enabled=False
        ),
    )
    released = build_semantic_edit_masks(
        _collar_image(),
        _landmarks(),
        _collar_labels(),
        config=SemanticMaskConfig(collar_margin_ratio=0.06),
    )

    freed = (frozen.protected > 0) & (released.protected == 0)
    assert freed.any()

    # The throat strip inside the very same band is skin and stays kept.
    neck_col = (NECK_COLS[0] + NECK_COLS[1]) // 2
    assert not freed[:, neck_col].any()
    assert released.protected[300, neck_col] == 255

    # Neither guard loses a single pixel, and the total protected area only
    # shrinks by exactly what was handed to the garment.
    assert released.stats["neckSkinProtectedPixels"] == frozen.stats["neckSkinProtectedPixels"]
    assert released.stats["identityGuardPixels"] == frozen.stats["identityGuardPixels"]
    assert released.stats["glassesGuardPixels"] == frozen.stats["glassesGuardPixels"]
    assert (
        frozen.stats["protectedPixels"] - released.stats["protectedPixels"]
        == int(freed.sum())
    )
    # Every freed pixel is a garment pixel now, not an uneditable gap.
    assert int((released.clothing[freed] > 0).sum()) == int(freed.sum())
    # The hair/clothing seam is unchanged: the release happens below the hair.
    assert released.stats["hairOverClothingRimPixels"] == frozen.stats["hairOverClothingRimPixels"]


def test_collar_margin_recovery_off_reproduces_the_last_accepted_mask():
    """The knob is additive: off is byte-for-byte the mask that was accepted."""
    off = build_semantic_edit_masks(
        _collar_image(),
        _landmarks(),
        _collar_labels(),
        config=SemanticMaskConfig(collar_margin_recovery_enabled=False),
    )
    assert off.stats["collarMarginRecoveredPixels"] == 0
    assert off.stats["collarMarginRecoveryEnabled"] is False
    assert off.stats["protectedCoversEditable"] is True
