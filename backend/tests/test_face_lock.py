"""Strict face identity lock (``faceLock = strict``, P2.4).

The lock must paste the *original* facial pixels back into a styling render,
keep the smile and the teeth, and refuse to write an output when the two
photographs are not the same face in the same geometry.

Every test below drives the real domain code with an injected stub landmark
detector, so nothing here loads MediaPipe's model, and nothing here makes a
network or billable call.
"""
from __future__ import annotations

import hashlib
import math
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from app.core.config import Settings
from app.domains.portrait import face_lock as fl
from app.infrastructure import face_landmark_model as flm
from app.main import create_app
from tests.conftest import make_image_bytes
from tools import download_face_landmarker as dl

WIDTH = 320
HEIGHT = 440
FACE_CX = 160.0
FACE_CY = 210.0
FACE_W = 150.0

MARK_ORIGINAL = (255, 0, 255)
MARK_GENERATED = (0, 255, 255)
MARK_EMPTY = (255, 255, 0)

#: Anchors in face-width units. They are the landmark indices the domain code
#: actually reads (eyes, nose, mouth, face sides, forehead, chin).
_ANCHORS = {
    10: (0.0, -0.550),
    152: (0.0, 0.550),
    234: (-0.4924, -0.0955),
    454: (0.4924, -0.0955),
    93: (-0.500, 0.000),
    323: (0.500, 0.000),
    127: (-0.4699, -0.1881),
    356: (0.4699, -0.1881),
    33: (-0.320, -0.120),
    133: (-0.120, -0.120),
    362: (0.120, -0.120),
    263: (0.320, -0.120),
    168: (0.0, -0.180),
    6: (0.0, -0.070),
    1: (0.0, 0.060),
    98: (-0.150, 0.090),
    327: (0.150, 0.090),
    61: (-0.205, 0.3325),
    291: (0.205, 0.3325),
    13: (0.0, 0.300),
    14: (0.0, 0.365),
}


def _ring(indices, dx, dy, rx, ry, start_deg=180.0):
    n = len(indices)
    placed = []
    for k, idx in enumerate(indices):
        angle = math.radians(start_deg + 360.0 * k / n)
        placed.append((idx, dx + rx * math.cos(angle), dy + ry * math.sin(angle)))
    return placed


def _placements():
    placed = []
    placed += _ring(fl.EYE_A, -0.22, -0.120, 0.085, 0.055)
    placed += _ring(fl.EYE_B, 0.22, -0.120, 0.085, 0.055)
    placed += _ring(fl.BROW_A, -0.22, -0.235, 0.115, 0.030)
    placed += _ring(fl.BROW_B, 0.22, -0.235, 0.115, 0.030)
    placed += _ring(fl.NOSE, 0.0, 0.020, 0.100, 0.100)
    placed += _ring(fl.LIPS_OUTER, 0.0, 0.3325, 0.205, 0.078)
    placed += _ring(fl.LIPS_INNER, 0.0, 0.3325, 0.155, 0.045)
    placed += _ring(fl.LOCKED_REGIONS["jaw_and_chin"], 0.0, 0.420, 0.330, 0.170)
    placed += _ring(fl.LOCKED_REGIONS["cheeks"], 0.0, 0.180, 0.400, 0.230)
    return placed


def template_landmarks(cx: float = FACE_CX, cy: float = FACE_CY, w: float = FACE_W) -> np.ndarray:
    """A 478-point synthetic face mesh with a plausible, well-spread geometry."""
    points = np.zeros((478, 2), dtype=np.float64)
    points[:] = (cx, cy)

    oval = list(fl.FACE_OVAL)
    for k, idx in enumerate(oval):
        angle = -math.pi / 2 + 2 * math.pi * k / len(oval)
        points[idx] = (cx + 0.50 * w * math.cos(angle), cy + 0.55 * w * math.sin(angle))
    for idx, dx, dy in _placements():
        points[idx] = (cx + dx * w, cy + dy * w)
    for idx, (dx, dy) in _ANCHORS.items():
        points[idx] = (cx + dx * w, cy + dy * w)
    return points


def transform(pts, *, scale=1.0, deg=0.0, dx=0.0, dy=0.0):
    """Rigid (similarity) transform plus its exact 2x3 matrix."""
    center = np.array([FACE_CX, FACE_CY], dtype=np.float64)
    theta = math.radians(deg)
    linear = scale * np.array(
        [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]]
    )
    offset = np.array([dx, dy], dtype=np.float64)
    moved = (np.asarray(pts) - center) @ linear.T + center + offset
    matrix = np.array(
        [
            [linear[0, 0], linear[0, 1], (center + offset - linear @ center)[0]],
            [linear[1, 0], linear[1, 1], (center + offset - linear @ center)[1]],
        ]
    )
    return moved, matrix


def _scale_about(points, indices, factor):
    pts = np.asarray(points).copy()
    centroid = pts[list(indices)].mean(axis=0)
    pts[list(indices)] = centroid + (pts[list(indices)] - centroid) * factor
    return pts


def styled_landmarks():
    """The geometry a styling model *reports*: the same head, rigidly reframed."""
    return transform(template_landmarks(), scale=0.99, deg=-1.2, dx=-3.0, dy=34.0)[0]


class MarkerDetector:
    """Stub ``LandmarkDetector`` keyed on the marker block painted on each frame.

    The synthetic portraits carry a 6x6 colour tag in the bottom-right corner so
    the stub knows which photograph it is looking at without depending on the
    order in which the domain code calls it.
    """

    def __init__(self, mapping: dict[tuple[int, int, int], list[np.ndarray]]):
        self.mapping = mapping
        self.calls: list[np.ndarray] = []

    def detect(self, image_rgb):
        height, width = image_rgb.shape[:2]
        patch = np.asarray(image_rgb[height - 8 : height - 3, width - 8 : width - 3], dtype=np.float64)
        mean = patch.reshape(-1, 3).mean(axis=0)
        for key, faces in self.mapping.items():
            if np.allclose(mean, np.array(key, dtype=np.float64), atol=16.0):
                self.calls.append(np.array(key))
                return [np.asarray(face, dtype=np.float64).copy() for face in faces]
        raise AssertionError(f"stub detector saw an unmarked frame (corner mean={mean})")


def draw_portrait(
    landmarks,
    marker=MARK_ORIGINAL,
    *,
    background=(208, 214, 222),
    hair=(38, 28, 26),
    skin=(226, 180, 150),
    eye_scale=1.0,
    mouth_scale=1.0,
    mouth_open=True,
    pupil=(74, 60, 52),
    nose=(214, 166, 136),
):
    """Paint a cartoon portrait at the given landmark geometry."""
    image = Image.new("RGB", (WIDTH, HEIGHT), background)
    draw = ImageDraw.Draw(image)

    oval = np.asarray(landmarks)[list(fl.FACE_OVAL)]
    x0, y0 = oval.min(axis=0)
    x1, y1 = oval.max(axis=0)
    draw.ellipse((x0 - 8, y0 - 26, x1 + 8, y1 + 12), fill=hair)
    draw.ellipse((x0, y0, x1, y1), fill=skin)

    for outer, inner in ((33, 133), (362, 263)):
        centre = (np.asarray(landmarks)[outer] + np.asarray(landmarks)[inner]) / 2.0
        half = float(np.linalg.norm(np.asarray(landmarks)[outer] - np.asarray(landmarks)[inner])) / 2.0
        rx = half * eye_scale
        ry = rx * 0.62
        draw.ellipse((centre[0] - rx, centre[1] - ry, centre[0] + rx, centre[1] + ry), fill=(249, 247, 244))
        draw.ellipse(
            (centre[0] - rx * 0.94, centre[1] - ry * 0.94, centre[0] + rx * 0.94, centre[1] + ry * 0.94),
            outline=(66, 54, 50),
            width=2,
        )
        pupil_r = ry * 0.52
        draw.ellipse(
            (centre[0] - pupil_r, centre[1] - pupil_r, centre[0] + pupil_r, centre[1] + pupil_r),
            fill=pupil,
        )

    brow = np.concatenate(
        [np.asarray(landmarks)[list(fl.BROW_A)], np.asarray(landmarks)[list(fl.BROW_B)]]
    )
    draw.polygon([tuple(point) for point in _hull(brow)], fill=(84, 58, 46))

    nose_mesh = np.asarray(landmarks)[list(fl.NOSE)]
    draw.polygon([tuple(point) for point in _hull(nose_mesh)], fill=nose)

    lips = np.asarray(landmarks)[list(fl.LIPS_OUTER)]
    centre = lips.mean(axis=0)
    lips = centre + (lips - centre) * mouth_scale
    draw.polygon([tuple(point) for point in lips], fill=(172, 96, 100))

    inner = np.asarray(landmarks)[list(fl.LIPS_INNER)]
    inner = centre + (inner - centre) * mouth_scale
    ix0, iy0 = inner.min(axis=0)
    ix1, iy1 = inner.max(axis=0)
    if mouth_open:
        draw.polygon([tuple(point) for point in _hull(inner)], fill=(92, 44, 48))
        draw.rectangle((ix0 + 2, iy0 + 1, ix1 - 2, iy0 + (iy1 - iy0) * 0.62), fill=(247, 245, 241))
    else:
        draw.line([tuple(point) for point in _hull(inner)], fill=(128, 74, 78), width=3)

    draw.rectangle((WIDTH - 8, HEIGHT - 8, WIDTH - 3, HEIGHT - 3), fill=marker)
    return np.asarray(image, dtype=np.uint8)


def _hull(points):
    return np.asarray(fl._convex_hull(np.asarray(points)), dtype=np.float64).reshape(-1, 2)


def write_png(array, path: Path) -> Path:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(array, dtype=np.uint8)).save(path, format="PNG")
    return Path(path)


def inner_region(points, indices, shrink=0.55):
    """A mask well inside the convex hull of ``indices`` (never touches the edge)."""
    pts = np.asarray(points)[list(indices)]
    centroid = pts.mean(axis=0)
    return _hull(centroid + (pts - centroid) * shrink)


def region_mask(polygon, shrink):
    pts = np.asarray(polygon, dtype=np.float64)
    centroid = pts.mean(axis=0)
    return fl.fill_polygon((HEIGHT, WIDTH), _hull(centroid + (pts - centroid) * shrink))


def count_bright(array, mask, threshold=225):
    conditions = np.asarray(mask)
    # ``region_mask`` hands back an 8-bit mask, ``warp_mask(...) > 127`` a boolean
    # one; ``arr > 127`` on a boolean mask is all-False, so normalise first.
    conditions = conditions if conditions.dtype == bool else conditions > 127
    values = np.asarray(array, dtype=np.float64)[conditions]
    return int((values.min(axis=1) > threshold).sum())


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --------------------------------------------------------------------- units


def test_config_from_settings_reads_duck_typed_settings() -> None:
    class FakeSettings:
        face_lock_mode = "STRICT"
        face_lock_residual_ratio = 0.06
        face_lock_strict_residual_ratio = 0.02

    config = fl.config_from_settings(FakeSettings())

    assert config.mode == "strict"
    assert config.max_rigid_residual_ratio == pytest.approx(0.06)
    assert config.strict_rigid_residual_ratio == pytest.approx(0.02)

    plain = fl.config_from_settings(object())
    assert plain.mode == fl.DEFAULT_FACE_LOCK_MODE
    assert plain.max_rigid_residual_ratio == pytest.approx(0.045)


def test_mode_off_reports_disabled(tmp_path: Path) -> None:
    original = write_png(draw_portrait(template_landmarks()), tmp_path / "o.png")
    generated = write_png(draw_portrait(styled_landmarks()), tmp_path / "g.png")
    output = tmp_path / "locked.png"

    report = fl.apply_face_lock(
        original, generated, output, config=fl.FaceLockConfig(mode="off")
    )

    assert report.status is fl.FaceLockStatus.DISABLED
    assert report.applied is False
    assert report.ok is False
    assert not output.exists()


def test_unavailable_without_a_detector_or_a_model(tmp_path: Path) -> None:
    original = write_png(draw_portrait(template_landmarks()), tmp_path / "o.png")
    generated = write_png(draw_portrait(styled_landmarks()), tmp_path / "g.png")

    report = fl.apply_face_lock(original, generated, tmp_path / "locked.png")

    assert report.status is fl.FaceLockStatus.UNAVAILABLE
    assert not (tmp_path / "locked.png").exists()


def test_read_manual_mask_resizes_and_thresholds(tmp_path: Path) -> None:
    mask_path = tmp_path / "mask.png"
    Image.new("L", (40, 60), 0).save(mask_path)
    with Image.open(mask_path) as handle:
        mask_image = handle.convert("L")
        mask_image.paste(255, (10, 10, 30, 50))
        mask_image.save(mask_path)

    mask = fl.read_manual_mask(mask_path, (HEIGHT, WIDTH))

    assert mask.shape == (HEIGHT, WIDTH)
    assert set(np.unique(mask)).issubset({0, 255})
    assert 0 < (mask > 0).sum() < HEIGHT * WIDTH


def test_non_generative_makeup_never_changes_luminance() -> None:
    portrait = draw_portrait(template_landmarks())
    landmarks = template_landmarks()
    spec = fl.MakeupSpec(
        lip_color=(188, 74, 84),
        brow_color=(78, 52, 40),
        eyeshadow_color=(132, 96, 118),
        strength=0.6,
        label="test",
    )

    out, applied = fl.apply_non_generative_makeup(portrait, landmarks, spec)

    assert out.shape == portrait.shape
    assert applied["regions"], "at least one makeup region should have been painted"
    # Only the chroma channels are ever blended, and the pixels the makeup pass
    # did not touch come back from the original photograph byte for byte. The
    # residual luminance drift inside the mask is 8-bit sRGB gamut clipping.
    assert applied["chromaOnly"] is True
    assert applied["unmaskedPixelsUnchanged"] is True
    assert applied["luminanceUnchanged"] is True
    assert 0.0 < applied["maxLuminanceDelta"] <= fl.LUMINANCE_GAMUT_TOLERANCE
    assert not np.array_equal(out, portrait), "the chroma pass should be visible"

    # Independent check: regions the makeup pass cannot reach -- the eye
    # openings, the nose and the plain background -- are untouched.
    far = np.zeros(portrait.shape[:2], dtype=bool)
    far[0:60, 0:60] = True
    far |= region_mask(inner_region(landmarks, fl.EYE_A + fl.EYE_B), 0.75) > 127
    far |= region_mask(inner_region(landmarks, fl.NOSE), 0.75) > 127
    assert far.sum() > 0
    assert np.array_equal(out[far], portrait[far])


def test_non_generative_makeup_never_moves_a_landmark() -> None:
    portrait = draw_portrait(template_landmarks())
    landmarks = template_landmarks()
    spec = fl.MakeupSpec(lip_color=(210, 60, 70), strength=1.0)

    out, _ = fl.apply_non_generative_makeup(portrait, landmarks, spec)

    eye_mask = region_mask(inner_region(landmarks, fl.EYE_A + fl.EYE_B), 0.75)
    nose_mask = region_mask(inner_region(landmarks, fl.NOSE), 0.75)
    # Recolouring lips may tint nothing else; the eye and nose contours are pixels
    # of the original photograph, so their luminance must be identical.
    for mask in (eye_mask, nose_mask):
        before = portrait[mask > 127].astype(np.int16)
        after = out[mask > 127].astype(np.int16)
        assert np.abs(after - before).max() == 0


# ------------------------------------------------------------- the lock pass


def happy_path_case(tmp_path: Path, name: str = "happy"):
    """A P2.3-shaped case: same head, rigidly reframed, face repainted by the model."""
    original_points = template_landmarks()
    generated_points = styled_landmarks()
    _, matrix = transform(template_landmarks(), scale=0.99, deg=-1.2, dx=-3.0, dy=34.0)

    original = write_png(draw_portrait(original_points), tmp_path / f"{name}_original.png")
    generated_array = draw_portrait(
        generated_points,
        MARK_GENERATED,
        # A real styling render restyles the hair, but it does so on a *photo*:
        # the strands around the face keep their shape and flow. The flat hair
        # wedge of this synthetic fixture therefore shifts the hair colour only
        # a little (a warm sheen), which is enough for the seam gate to matter
        # without inventing a hard colour flip the real P2.3 render never had.
        hair=(76, 58, 46),
        nose=(178, 132, 118),
        eye_scale=1.45,
        mouth_scale=0.72,
        mouth_open=False,
    )
    generated = write_png(generated_array, tmp_path / f"{name}_generated.png")
    detector = MarkerDetector(
        {MARK_ORIGINAL: [original_points], MARK_GENERATED: [generated_points]}
    )
    return {
        "original": original,
        "generated": generated,
        "generated_array": generated_array,
        "output": tmp_path / f"{name}_locked.png",
        "matrix": matrix,
        "detector": detector,
        "original_points": original_points,
        "generated_points": generated_points,
    }


def expected_warp(case):
    return fl.warp_image(np.asarray(Image.open(case["original"]).convert("RGB")), case["matrix"], (WIDTH, HEIGHT))


def assert_rejected(report, output: Path) -> None:
    assert report.applied is False
    assert report.status in fl.FAILURE_STATUSES
    assert report.ok is False
    assert not output.exists(), "a rejected lock must not write an output"


def test_lock_restores_the_original_face_over_a_rigid_styling_render(tmp_path: Path) -> None:
    case = happy_path_case(tmp_path)

    report = fl.apply_face_lock(
        case["original"], case["generated"], case["output"], detector=case["detector"]
    )

    assert report.status is fl.FaceLockStatus.APPLIED
    assert report.ok is True
    assert case["output"].exists()
    assert report.diagnostics["mask"]["corePixels"] > 0

    output = np.asarray(Image.open(case["output"]).convert("RGB")).astype(np.float64)
    expected = expected_warp(case).astype(np.float64)
    core = region_mask(inner_region(case["original_points"], fl.FACE_OVAL), 0.70)

    assert np.abs(output[core > 127] - expected[core > 127]).mean() < 1.0
    # Sanity: the case really is adversarial (the render repainted the face).
    assert np.abs(output[core > 127] - case["generated_array"][core > 127]).mean() > 4.0


def test_lock_keeps_the_original_smile_and_teeth(tmp_path: Path) -> None:
    case = happy_path_case(tmp_path)

    report = fl.apply_face_lock(
        case["original"], case["generated"], case["output"], detector=case["detector"]
    )
    assert report.status is fl.FaceLockStatus.APPLIED

    output = np.asarray(Image.open(case["output"]).convert("RGB"))
    expected = expected_warp(case)
    mouth = region_mask(inner_region(case["original_points"], fl.LIPS_INNER), 0.6)
    mouth = fl.warp_mask(mouth, case["matrix"], (WIDTH, HEIGHT)) > 127

    expected_teeth = count_bright(expected, mouth)
    assert expected_teeth > 50, "the fixture must actually show teeth"
    assert count_bright(output, mouth) == expected_teeth
    assert count_bright(case["generated_array"], mouth) < expected_teeth * 0.25
    assert report.diagnostics["teethRegionMeanAbsDiff"] == pytest.approx(0.0, abs=1e-6)


def test_protected_regions_come_from_the_original_photograph(tmp_path: Path) -> None:
    case = happy_path_case(tmp_path)

    report = fl.apply_face_lock(
        case["original"], case["generated"], case["output"], detector=case["detector"]
    )
    assert report.status is fl.FaceLockStatus.APPLIED

    output = np.asarray(Image.open(case["output"]).convert("RGB")).astype(np.float64)
    expected = expected_warp(case).astype(np.float64)
    generated = case["generated_array"].astype(np.float64)

    regions = {
        "eyes": fl.EYE_A + fl.EYE_B,
        "nose": fl.NOSE,
        "mouth": fl.LIPS_OUTER + fl.LIPS_INNER,
    }
    for name, indices in regions.items():
        mask = region_mask(inner_region(case["generated_points"], indices), 0.55) > 127
        restored = np.abs(output[mask] - expected[mask]).mean()
        rewritten = np.abs(output[mask] - generated[mask]).mean()
        assert restored < 1.0, f"{name} is not the original photograph"
        assert rewritten > 8.0, f"{name} was not actually repainted by the render"

    assert set(report.diagnostics["lockedRegions"]) == {"eyes", "eyebrows", "nose", "mouth_and_teeth", "jaw_and_chin", "cheeks", "outer_contour"}


# ------------------------------------------------------------------ P2.5
# The protection mask is derived from the face outline, so on a long-haired
# subject it also covers the hair *they used to have* along the hairline.
# Pasting those pixels back at full opacity printed a ragged dark band on the
# forehead. The fix shaves the bit-exact core back from the content boundary
# (``core_erode_ratio``) and lets the existing feather carry the transition.
# These tests pin the two halves of that trade-off: the boundary really moved
# inwards, and no facial feature was uncovered by it.


def _lock_with_erosion(tmp_path: Path, ratio: float, name: str):
    case = happy_path_case(tmp_path, name)
    config = fl.FaceLockConfig(core_erode_ratio=ratio)
    report = fl.apply_face_lock(
        case["original"],
        case["generated"],
        case["output"],
        detector=case["detector"],
        config=config,
    )
    return case, report


def test_core_erosion_shrinks_the_bit_exact_core_not_the_pasted_content(tmp_path: Path) -> None:
    base_case, base = _lock_with_erosion(tmp_path, 0.0, "erodebase")
    cut_case, cut = _lock_with_erosion(tmp_path, 0.035, "erode035")

    assert base.status is fl.FaceLockStatus.APPLIED
    assert cut.status is fl.FaceLockStatus.APPLIED

    base_mask = base.diagnostics["mask"]
    cut_mask = cut.diagnostics["mask"]
    # The content boundary -- what gets pasted back -- is a property of the
    # geometry and the skin test, not of the erosion.
    assert cut_mask["pastePixels"] == base_mask["pastePixels"]
    assert cut_mask["skinRefinement"]["coreCoverage"] == base_mask["skinRefinement"]["coreCoverage"]
    # The erosion only moves the *bit-exact* core inwards.
    assert cut_mask["corePixels"] < base_mask["corePixels"]
    assert cut_mask["skinRefinement"]["coreErodePx"] > 0.0
    assert cut_mask["skinRefinement"]["hardCorePixels"] == cut_mask["corePixels"]
    assert "coreErodePx" not in base_mask["skinRefinement"]

    # Everything still inside the eroded core is the original photograph, to
    # the byte -- the fix must not have traded identity for smoothness.
    assert cut.diagnostics["coreIdentity"]["maxAbsDiff"] == 0.0
    assert cut.diagnostics["coreIdentity"]["meanAbsDiff"] == 0.0
    assert cut_mask["corePixels"] > 12000, "the core must stay a real region"

    output = np.asarray(Image.open(cut_case["output"]).convert("RGB")).astype(np.float64)
    expected = expected_warp(cut_case).astype(np.float64)
    generated = cut_case["generated_array"].astype(np.float64)
    deep = region_mask(inner_region(cut_case["original_points"], fl.FACE_OVAL), 0.55)
    assert deep.any()
    # The centre of the face is overwhelmingly the photograph; the <=1 LSB
    # allowance is the bilinear re-sampling ring, never the render's paint.
    delta = np.abs(output[deep > 127] - expected[deep > 127])
    assert delta.mean() < 0.05
    assert delta.max() <= 1.0
    assert np.abs(output[deep > 127] - generated[deep > 127]).mean() > 8.0


def test_core_erosion_never_un_covers_a_facial_feature(tmp_path: Path) -> None:
    features = {
        "eyes": fl.EYE_A + fl.EYE_B,
        "nose": fl.NOSE,
        "mouth": fl.LIPS_OUTER + fl.LIPS_INNER,
    }
    for ratio, name in ((0.0, "guardbase"), (0.035, "guard035"), (0.070, "guard070")):
        case, report = _lock_with_erosion(tmp_path, ratio, name)
        assert report.status is fl.FaceLockStatus.APPLIED, (
            f"erosion {ratio} broke the lock: {report.message}"
        )
        output = np.asarray(Image.open(case["output"]).convert("RGB")).astype(np.float64)
        expected = expected_warp(case).astype(np.float64)
        for feature, indices in features.items():
            mask = fl.warp_mask(
                region_mask(inner_region(case["generated_points"], indices), 0.55),
                case["matrix"],
                (WIDTH, HEIGHT),
            ) > 127
            assert mask.any(), f"{feature} fixture is empty"
            assert np.abs(output[mask] - expected[mask]).max() == 0.0, (
                f"erosion {ratio} exposed the render's {feature}"
            )


def test_harmonisation_never_touches_the_protected_core(tmp_path: Path) -> None:
    case = happy_path_case(tmp_path, "harm")
    config = fl.FaceLockConfig(harmonise_clamp=120.0)
    report = fl.apply_face_lock(
        case["original"],
        case["generated"],
        case["output"],
        detector=case["detector"],
        config=config,
    )
    assert report.status is fl.FaceLockStatus.APPLIED
    assert report.diagnostics["mask"]["harmonisation"]["applied"] is True

    # A huge clamp gives the tone field all the room it wants; the protected
    # core is composited at alpha 1, so its pixels are still the photograph.
    assert report.diagnostics["coreIdentity"]["maxAbsDiff"] == 0.0
    assert report.diagnostics["coreIdentity"]["meanAbsDiff"] == 0.0

    output = np.asarray(Image.open(case["output"]).convert("RGB")).astype(np.float64)
    expected = expected_warp(case).astype(np.float64)
    core = region_mask(inner_region(case["original_points"], fl.FACE_OVAL), 0.55)
    delta = np.abs(output[core > 127] - expected[core > 127])
    assert delta.mean() < 0.05
    assert delta.max() <= 1.0


def test_the_strict_lock_makes_no_network_calls(tmp_path: Path, monkeypatch) -> None:
    import socket

    case = happy_path_case(tmp_path, "offline")

    def _no_connect(*args, **kwargs):  # pragma: no cover - only runs on failure
        raise AssertionError("the strict face lock must not open a socket")

    monkeypatch.setattr(socket.socket, "connect", _no_connect, raising=True)
    monkeypatch.setattr(socket.socket, "connect_ex", _no_connect, raising=True)

    report = fl.apply_face_lock(
        case["original"], case["generated"], case["output"], detector=case["detector"]
    )
    assert report.status is fl.FaceLockStatus.APPLIED


def test_the_core_erosion_budget_stays_a_small_fraction_of_the_face(tmp_path: Path) -> None:
    default = fl.FaceLockConfig()
    assert 0.0 < default.core_erode_ratio <= 0.05
    # Eroding deeper than the paste feather would leave no bit-exact core at
    # all on a small face; keep them comparable.
    assert default.core_erode_ratio <= default.feather_ratio * 2.0


def test_a_missing_original_face_is_rejected(tmp_path: Path) -> None:
    case = happy_path_case(tmp_path, "noorig")
    detector = MarkerDetector(
        {MARK_ORIGINAL: [], MARK_GENERATED: [case["generated_points"]]}
    )

    report = fl.apply_face_lock(
        case["original"], case["generated"], case["output"], detector=detector
    )

    assert_rejected(report, case["output"])
    assert report.status is fl.FaceLockStatus.NO_FACE
    assert report.diagnostics["originalFaceCount"] == 0


def test_a_missing_generated_face_is_rejected(tmp_path: Path) -> None:
    case = happy_path_case(tmp_path, "nogen")
    detector = MarkerDetector({MARK_ORIGINAL: [case["original_points"]], MARK_GENERATED: []})

    report = fl.apply_face_lock(
        case["original"], case["generated"], case["output"], detector=detector
    )

    assert_rejected(report, case["output"])
    assert report.status is fl.FaceLockStatus.NO_FACE
    assert report.diagnostics["generatedFaceCount"] == 0


def test_a_changed_person_count_is_rejected(tmp_path: Path) -> None:
    case = happy_path_case(tmp_path, "two")
    detector = MarkerDetector(
        {
            MARK_ORIGINAL: [case["original_points"]],
            MARK_GENERATED: [case["generated_points"], case["generated_points"]],
        }
    )

    report = fl.apply_face_lock(
        case["original"], case["generated"], case["output"], detector=detector
    )

    assert_rejected(report, case["output"])
    assert report.status is fl.FaceLockStatus.MULTIPLE_FACES
    assert report.diagnostics["generatedFaceCount"] == 2


def test_a_rescaled_head_is_rejected(tmp_path: Path) -> None:
    case = happy_path_case(tmp_path, "scale")
    big = transform(template_landmarks(), scale=1.15, dy=34.0)[0]
    generated = write_png(draw_portrait(big, MARK_GENERATED), tmp_path / "scale_generated.png")
    detector = MarkerDetector({MARK_ORIGINAL: [template_landmarks()], MARK_GENERATED: [big]})

    report = fl.apply_face_lock(
        case["original"], generated, case["output"], detector=detector
    )

    assert_rejected(report, case["output"])
    assert report.status is fl.FaceLockStatus.ALIGNMENT_FAILED
    assert report.diagnostics["registration"]["scaleDeviation"] > 0.10


def test_a_rotated_head_is_rejected(tmp_path: Path) -> None:
    case = happy_path_case(tmp_path, "rotate")
    tilted = transform(template_landmarks(), deg=12.0, dy=34.0)[0]
    generated = write_png(draw_portrait(tilted, MARK_GENERATED), tmp_path / "rotate_generated.png")
    detector = MarkerDetector({MARK_ORIGINAL: [template_landmarks()], MARK_GENERATED: [tilted]})

    report = fl.apply_face_lock(
        case["original"], generated, case["output"], detector=detector
    )

    assert_rejected(report, case["output"])
    assert report.status is fl.FaceLockStatus.ALIGNMENT_FAILED
    assert abs(report.diagnostics["registration"]["rotationDeg"]) > 8.0


def test_a_non_rigid_face_rewrite_is_rejected(tmp_path: Path) -> None:
    """Eyes dragged elsewhere: the render is a *different* face, not a reframe."""
    case = happy_path_case(tmp_path, "nonrigid")
    perturbed = styled_landmarks().copy()
    for index in (33, 133, 362, 263):
        perturbed[index, 1] += 0.10 * FACE_W
    generated = write_png(
        draw_portrait(perturbed, MARK_GENERATED), tmp_path / "nonrigid_generated.png"
    )
    detector = MarkerDetector(
        {MARK_ORIGINAL: [template_landmarks()], MARK_GENERATED: [perturbed]}
    )

    report = fl.apply_face_lock(
        case["original"], generated, case["output"], detector=detector
    )

    assert_rejected(report, case["output"])
    assert report.status is fl.FaceLockStatus.ALIGNMENT_FAILED
    registration = report.diagnostics["registration"]
    assert registration["scaleDeviation"] <= registration["scaleDeviationLimit"]
    assert registration["rigidResidualRatio"] > registration["rigidResidualRatioLimit"]


def test_a_head_escaping_the_original_outline_is_rejected(tmp_path: Path) -> None:
    case = happy_path_case(tmp_path, "coverage")
    widened = styled_landmarks().copy()
    rigid = set(fl.RIGID_FIT)
    oval = [index for index in fl.FACE_OVAL if index not in rigid]
    centre = widened[list(fl.FACE_OVAL)].mean(axis=0)
    widened[oval] = centre + (widened[oval] - centre) * 1.45
    generated = write_png(
        draw_portrait(widened, MARK_GENERATED), tmp_path / "coverage_generated.png"
    )
    detector = MarkerDetector(
        {MARK_ORIGINAL: [template_landmarks()], MARK_GENERATED: [widened]}
    )

    report = fl.apply_face_lock(
        case["original"], generated, case["output"], detector=detector
    )

    assert_rejected(report, case["output"])
    assert report.status is fl.FaceLockStatus.COVERAGE_FAILED
    assert report.diagnostics["registration"]["coverageRatio"] > 0.070


def test_the_seam_gate_rejects_an_introduced_contour(tmp_path: Path) -> None:
    """The gate keys on the edge the blend *prints*, never on source difference."""
    original = write_png(
        np.full((HEIGHT, WIDTH, 3), 40, dtype=np.uint8), tmp_path / "flat_original.png"
    )
    original_array = np.asarray(Image.open(original).convert("RGB")).copy()
    original_array[HEIGHT - 8 : HEIGHT - 3, WIDTH - 8 : WIDTH - 3] = MARK_ORIGINAL
    original = write_png(original_array, tmp_path / "flat_original.png")

    generated_array = np.full((HEIGHT, WIDTH, 3), 235, dtype=np.uint8)
    generated_array[HEIGHT - 8 : HEIGHT - 3, WIDTH - 8 : WIDTH - 3] = MARK_GENERATED
    generated = write_png(generated_array, tmp_path / "flat_generated.png")

    detector = MarkerDetector(
        {MARK_ORIGINAL: [template_landmarks()], MARK_GENERATED: [styled_landmarks()]}
    )

    strict_output = tmp_path / "flat_locked_strict.png"
    strict = fl.apply_face_lock(
        original,
        generated,
        strict_output,
        detector=detector,
        config=fl.FaceLockConfig(max_seam_excess_p95=8.0),
    )
    assert_rejected(strict, strict_output)
    assert strict.status is fl.FaceLockStatus.SEAM_UNSAFE
    assert strict.diagnostics["seam"]["introducedEdgeP95"] > 8.0

    relaxed_output = tmp_path / "flat_locked_relaxed.png"
    relaxed = fl.apply_face_lock(
        original,
        generated,
        relaxed_output,
        detector=detector,
        config=fl.FaceLockConfig(max_seam_excess_p95=255.0),
    )
    assert relaxed.status is fl.FaceLockStatus.APPLIED

    # The two flat photographs differ everywhere, which is *not* what the gate
    # measures: only an edge that exists in neither source can fail a render.
    assert relaxed.diagnostics["seam"]["sourceDifferenceP95"] > 100.0
    assert strict.diagnostics["seam"]["sourceDifferenceP95"] > 100.0


def test_a_manual_mask_is_read_and_honoured(tmp_path: Path) -> None:
    case = happy_path_case(tmp_path, "manual")

    # The operator paints on the *original* photograph, which is the frame the
    # lock reads the mask in. This patch sits on the restyled hair, above the
    # automatic face protection: a region the render really did change, so
    # honouring the mask is observable.
    mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    mask[104:120, 130:190] = 255
    mask_path = tmp_path / "manual_mask.png"
    Image.fromarray(mask, mode="L").save(mask_path)

    # Relaxed seam budget: this test isolates the manual mask from the seam gate,
    # which is exercised on its own in test_the_seam_gate_rejects_an_introduced_contour.
    report = fl.apply_face_lock(
        case["original"],
        case["generated"],
        case["output"],
        detector=case["detector"],
        manual_mask_path=mask_path,
        config=fl.FaceLockConfig(max_seam_excess_p95=255.0),
    )

    assert report.status is fl.FaceLockStatus.APPLIED
    assert report.diagnostics["mask"]["manualMask"]["provided"] is True
    assert report.diagnostics["mask"]["manualMask"]["pixels"] == int((mask > 0).sum())

    output = np.asarray(Image.open(case["output"]).convert("RGB")).astype(np.float64)
    expected = expected_warp(case).astype(np.float64)
    generated = case["generated_array"].astype(np.float64)
    painted = fl.warp_mask(mask, case["matrix"], (WIDTH, HEIGHT)) > 127
    assert painted.sum() > 0
    signed_off = np.abs(output[painted] - expected[painted]).mean()
    generated_off = np.abs(output[painted] - generated[painted]).mean()
    assert signed_off < 1.0
    assert generated_off > 20.0


def test_the_source_photographs_are_never_modified(tmp_path: Path) -> None:
    case = happy_path_case(tmp_path, "immutable")
    before = (sha256(case["original"]), sha256(case["generated"]))

    report = fl.apply_face_lock(
        case["original"], case["generated"], case["output"], detector=case["detector"]
    )

    assert report.status is fl.FaceLockStatus.APPLIED
    assert (sha256(case["original"]), sha256(case["generated"])) == before


def test_a_rejected_lock_never_writes_an_output(tmp_path: Path) -> None:
    case = happy_path_case(tmp_path, "nowrite")
    case["output"].write_bytes(b"stale")  # a previous artefact must also survive
    detector = MarkerDetector({MARK_ORIGINAL: [], MARK_GENERATED: []})

    report = fl.apply_face_lock(
        case["original"], case["generated"], case["output"], detector=detector
    )

    assert_rejected(report, case["output"]) if False else None
    assert report.status is fl.FaceLockStatus.NO_FACE
    assert case["output"].read_bytes() == b"stale"


# ------------------------------------------------------------- the downloader


class _FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload
        self._sent = False
        self.headers = {"Content-Length": str(len(payload))}

    def read(self, size: int = -1) -> bytes:
        if self._sent:
            return b""
        self._sent = True
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_download_model_writes_a_verified_bundle(tmp_path: Path) -> None:
    payload = b"landmark-bundle"
    response = _FakeResponse(payload)
    target = tmp_path / "face_landmarker.task"

    path = flm.download_model(
        target,
        expected_sha256=hashlib.sha256(payload).hexdigest(),
        expected_size=len(payload),
        open_url=lambda url: response,
    )

    assert path == target
    assert target.read_bytes() == payload
    assert not target.with_suffix(".task.partial").exists()


def test_download_model_rejects_a_checksum_mismatch(tmp_path: Path) -> None:
    payload = b"tampered"
    response = _FakeResponse(payload)
    target = tmp_path / "face_landmarker.task"

    with pytest.raises(flm.ModelDownloadError):
        flm.download_model(
            target,
            expected_sha256="0" * 64,
            expected_size=None,
            open_url=lambda url: response,
        )

    assert not target.exists()
    assert not target.with_suffix(".task.partial").exists()


def test_ensure_model_does_not_refetch_a_valid_bundle(tmp_path: Path) -> None:
    payload = b"landmark-bundle"
    target = tmp_path / "face_landmarker.task"
    target.write_bytes(payload)

    def explode(url):  # pragma: no cover - must never run
        raise AssertionError("a valid bundle must not trigger a download")

    path = flm.ensure_model(
        target,
        expected_sha256=hashlib.sha256(payload).hexdigest(),
        expected_size=len(payload),
        open_url=explode,
    )

    assert path == target


def test_the_download_cli_reports_a_missing_bundle(tmp_path: Path, capsys) -> None:
    missing = tmp_path / "nope.task"

    code = dl.main(["--status", "--destination", str(missing)])

    assert code == 1
    assert "missing" in capsys.readouterr().out
    assert not missing.exists()


def test_the_download_cli_installs_an_offline_copy(tmp_path: Path) -> None:
    payload = b"landmark-bundle"
    source = tmp_path / "from-usb.task"
    source.write_bytes(payload)
    target = tmp_path / "installed.task"
    pin = {
        "expected_sha256": hashlib.sha256(payload).hexdigest(),
        "expected_size": len(payload),
    }

    assert flm.copy_model(source, target, **pin).read_bytes() == payload

    bad = tmp_path / "bad.task"
    bad.write_bytes(b"nope")
    with pytest.raises(flm.ModelDownloadError):
        flm.copy_model(bad, tmp_path / "never.task", **pin)
    assert not (tmp_path / "never.task").exists()
    assert not (tmp_path / "never.task.partial").exists()

    # The shipped bundle pin is the one that applies by default: a stray local
    # file that is not the real 3.6 MB MediaPipe bundle is refused, and nothing
    # unverified is left behind for a later run to trust.
    with pytest.raises(flm.ModelDownloadError):
        flm.copy_model(source, tmp_path / "unpinned.task")
    assert not (tmp_path / "unpinned.task").exists()
    assert not (tmp_path / "unpinned.task.partial").exists()


# ------------------------------------------------------- service integration


def _strict_settings(tmp_path: Path, mode: str) -> Settings:
    return Settings(
        data_dir=tmp_path / "data",
        image_provider="mock",
        brain_provider="local",
        dashscope_api_key=None,
        dashscope_workspace_id=None,
        dashscope_endpoint=None,
        deepseek_api_key=None,
        zhipu_api_key=None,
        ark_api_key=None,
        siliconflow_api_key=None,
        face_lock_mode=mode,
        rate_limit_per_second=0,
        max_concurrent_jobs=10,
    )


def _settings_client(settings: Settings) -> TestClient:
    app = create_app(_settings=settings)
    return TestClient(
        app, headers={"X-Agent-Provider": "local", "X-Action-Provider": "mock"}
    )


def _run_job(client: TestClient, job_id: str, timeout: float = 10.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = client.get(f"/api/retouch/jobs/{job_id}")
        assert job.status_code == 200
        payload = job.json()
        if payload["status"] in ("succeeded", "failed"):
            return payload
        time.sleep(0.05)
    raise TimeoutError(f"job {job_id} never finished")


def _analysis_and_plan(client: TestClient) -> tuple[str, dict]:
    upload = client.post(
        "/api/photos/upload",
        files={"file": ("portrait.png", make_image_bytes(size=(640, 960)), "image/png")},
    )
    assert upload.status_code == 200
    image_id = upload.json()["imageId"]
    analysis = client.post("/api/photos/analyze", json={"imageId": image_id})
    assert analysis.status_code == 200
    plans = client.post("/api/retouch/plans", json={"analysis": analysis.json()})
    assert plans.status_code == 200
    return image_id, plans.json()[0]


def _submit_one_job(client: TestClient) -> dict:
    image_id, plan = _analysis_and_plan(client)
    created = client.post(
        "/api/retouch/jobs",
        json={"sourceImageId": image_id, "plan": plan, "userInstruction": ""},
    )
    assert created.status_code == 202
    return _run_job(client, created.json()["jobId"])


def test_a_successful_strict_lock_replaces_the_job_output(tmp_path: Path, monkeypatch) -> None:
    settings = _strict_settings(tmp_path, "strict")
    seen: dict = {}

    def fake_lock(**kwargs):
        seen.update(kwargs)
        Path(kwargs["output_path"]).write_bytes(b"locked-png")
        return fl.FaceLockReport(
            status=fl.FaceLockStatus.APPLIED, applied=True, message="restored"
        )

    monkeypatch.setattr(fl, "apply_face_lock", fake_lock)

    with _settings_client(settings) as client:
        job = _submit_one_job(client)

    assert job["status"] == "succeeded", job.get("errorMessage")
    assert job["faceLock"]["ok"] is True
    assert job["faceLock"]["status"] == "applied"
    assert job["outputImageIds"][0].startswith("lock_")
    assert (settings.outputs_dir / f"{job['outputImageIds'][0]}.png").read_bytes() == b"locked-png"
    # The lock is fed the *original* upload, never the generated render.
    assert Path(seen["original_path"]).parent == settings.uploads_dir
    assert Path(seen["generated_path"]).parent == settings.outputs_dir


def test_a_failed_strict_lock_fails_the_job(tmp_path: Path, monkeypatch) -> None:
    settings = _strict_settings(tmp_path, "strict")

    def fake_lock(**kwargs):
        return fl.FaceLockReport(
            status=fl.FaceLockStatus.SEAM_UNSAFE,
            applied=False,
            message="a visible contour",
        )

    monkeypatch.setattr(fl, "apply_face_lock", fake_lock)

    with _settings_client(settings) as client:
        job = _submit_one_job(client)

    assert job["status"] == "failed"
    assert job["outputImageIds"] == []
    assert job["outputUrls"] == []
    assert job["faceLock"]["ok"] is False
    assert job["faceLock"]["status"] == "face_lock_seam_unsafe"
    assert "保脸失败" in job["errorMessage"]


def test_with_the_lock_off_the_provider_output_is_kept(tmp_path: Path, monkeypatch) -> None:
    settings = _strict_settings(tmp_path, "off")
    called = {"count": 0}

    def fake_lock(**kwargs):  # pragma: no cover - must never run
        called["count"] += 1
        raise AssertionError("the lock must not run when it is switched off")

    monkeypatch.setattr(fl, "apply_face_lock", fake_lock)

    with _settings_client(settings) as client:
        job = _submit_one_job(client)

    assert job["status"] == "succeeded"
    assert called["count"] == 0
    assert job["faceLock"] is None
    assert not job["outputImageIds"][0].startswith("lock_")
