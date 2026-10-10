"""Tests for :mod:`app.domains.portrait.inpaint_verify`.

These pin the promises the masked backend makes to the user:

* the pristine pixels stay pristine -- the final composite is the original
  wherever the mask is black, and nothing is ever warped to make that true;
* a result that cannot be registered (wrong size, systematic offset, model
  repaint over the kept region) is *refused*, never reported as a success;
* the three statuses -- ``MASK_VALID``, ``PIXEL_IDENTITY_PASSED`` and
  ``VISUAL_ACCEPTANCE_PENDING`` -- stay independent and are never conflated.

Everything here is pure numpy/Pillow; no provider, no network, no billing.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.domains.portrait.inpaint_verify import (
    InpaintVerificationError,
    best_alignment_shift,
    load_mask,
    restore_kept_pixels,
    review_mask,
    review_protected_region,
    verify_inpaint_output,
)

HEIGHT = 48
WIDTH = 64
#: The repaint box (white in the mask), parked in the "clothing" area.
EDIT_BOX = (slice(30, 40), slice(40, 50))
#: The protected facial features, entirely inside the kept (black) region.
FACE_GUARD = (slice(18, 28), slice(20, 32))


def _pattern(height: int = HEIGHT, width: int = WIDTH) -> np.ndarray:
    y = np.arange(height)[:, None]
    x = np.arange(width)[None, :]
    base = (y * 5 + x * 3) % 200 + 20
    return np.stack(
        [base, (base + 40) % 256, (base + 80) % 256], axis=-1
    ).astype(np.uint8)


def _flat(color: tuple[int, int, int] = (120, 130, 140)) -> np.ndarray:
    array = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    array[:, :] = color
    return array


def _mask_array(edit_box: tuple[slice, slice] = EDIT_BOX) -> np.ndarray:
    mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    mask[edit_box] = 255
    return mask


def _guard_array(box: tuple[slice, slice] = FACE_GUARD) -> np.ndarray:
    guard = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    guard[box] = 255
    return guard


def _save_rgb(path: Path, array: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array.astype(np.uint8)).save(path)
    return path


def _save_mask(path: Path, array: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array.astype(np.uint8), mode="L").save(path)
    return path


# ------------------------------------------------------------------ mask review


def test_review_mask_accepts_a_clean_black_white_mask() -> None:
    review = review_mask(_mask_array(), (WIDTH, HEIGHT))
    assert review.valid is True
    assert review.reasons == ()
    assert review.width == WIDTH and review.height == HEIGHT
    assert review.editable_pixels == 100
    assert review.protected_pixels == HEIGHT * WIDTH - 100
    assert 0.0 < review.editable_share < 1.0
    assert len(review.sha256) == 64
    assert review.to_dict()["semantics"] == "white=repaint black=keep original pixels"


def test_review_mask_reports_a_size_mismatch() -> None:
    review = review_mask(np.zeros((10, 10), dtype=np.uint8), (WIDTH, HEIGHT))
    assert review.valid is False
    assert any("own coordinates" in reason for reason in review.reasons)


def test_review_mask_rejects_grey_levels() -> None:
    mask = _mask_array()
    mask[0, 0] = 128
    review = review_mask(mask, (WIDTH, HEIGHT))
    assert review.valid is False
    assert any("strictly black/white" in reason for reason in review.reasons)


def test_review_mask_rejects_an_empty_mask() -> None:
    review = review_mask(np.zeros((HEIGHT, WIDTH), dtype=np.uint8), (WIDTH, HEIGHT))
    assert review.valid is False
    assert any("almost nothing" in reason for reason in review.reasons)


def test_review_mask_rejects_a_whole_frame_mask() -> None:
    mask = np.full((HEIGHT, WIDTH), 255, dtype=np.uint8)
    review = review_mask(mask, (WIDTH, HEIGHT))
    assert review.valid is False
    assert any("whole-image edit" in reason for reason in review.reasons)


def test_review_mask_rejects_editable_pixels_on_the_face() -> None:
    guard = _guard_array()
    mask = _mask_array()
    mask[22:26, 24:28] = 255  # a repaint box that reaches into the face guard
    review = review_mask(mask, (WIDTH, HEIGHT), identity_guard=guard)
    assert review.valid is False
    assert review.identity_guard_overlap == 16
    assert any("overlaps the protected facial features" in reason for reason in review.reasons)


def test_review_mask_accepts_when_the_guard_is_fully_kept() -> None:
    review = review_mask(_mask_array(), (WIDTH, HEIGHT), identity_guard=_guard_array())
    assert review.valid is True
    assert review.identity_guard_overlap == 0


def test_review_mask_rejects_a_mis_sized_identity_guard() -> None:
    with pytest.raises(InpaintVerificationError):
        review_mask(_mask_array(), (WIDTH, HEIGHT), identity_guard=np.zeros((5, 5), dtype=np.uint8))


def test_load_mask_accepts_a_file_and_an_array() -> None:
    array = _mask_array()
    from_array, array_digest = load_mask(array)
    assert array_digest == load_mask(np.stack([array] * 3, axis=-1))[1]
    assert from_array.shape == (HEIGHT, WIDTH)
    assert from_array.dtype == np.uint8


# ------------------------------------------------------------------- alignment


def test_best_alignment_shift_finds_an_exact_translation() -> None:
    original = _pattern()
    generated = np.roll(original, shift=2, axis=0)
    region = np.ones((HEIGHT, WIDTH), dtype=bool)
    shift, best, zero = best_alignment_shift(generated, original, region, max_shift=3)
    assert shift == (-2, 0)
    assert best == pytest.approx(0.0)
    assert zero > 0.0


def test_best_alignment_shift_is_zero_for_identical_images() -> None:
    original = _pattern()
    region = np.ones((HEIGHT, WIDTH), dtype=bool)
    shift, best, zero = best_alignment_shift(original, original, region)
    assert shift == (0, 0)
    assert best == pytest.approx(0.0)
    assert zero == pytest.approx(0.0)


def test_best_alignment_shift_handles_an_empty_region() -> None:
    original = _pattern()
    region = np.zeros((HEIGHT, WIDTH), dtype=bool)
    assert best_alignment_shift(original, original, region) == ((0, 0), 0.0, 0.0)


# ----------------------------------------------------------------- compositing


def test_restore_kept_pixels_copies_original_into_the_black_region() -> None:
    original = _pattern()
    generated = _flat((250, 20, 30))
    mask = _mask_array()

    final = restore_kept_pixels(original, generated, mask)

    assert np.array_equal(final[EDIT_BOX], generated[EDIT_BOX])
    kept = np.logical_not(mask == 255)
    assert np.array_equal(final[kept], original[kept])
    # The final image is byte-identical to the original everywhere the mask kept.
    assert final[0, 0].tolist() == original[0, 0].tolist()


def test_restore_kept_pixels_refuses_to_resize() -> None:
    original = _pattern()
    generated = _pattern(height=HEIGHT - 4, width=WIDTH - 4)
    with pytest.raises(InpaintVerificationError):
        restore_kept_pixels(original, generated, _mask_array())


def test_restore_kept_pixels_refuses_a_mask_in_other_coordinates() -> None:
    original = _pattern()
    generated = _pattern()
    with pytest.raises(InpaintVerificationError):
        restore_kept_pixels(original, generated, np.zeros((8, 8), dtype=np.uint8))


def test_review_protected_region_marks_the_black_pixels() -> None:
    kept = review_protected_region(_mask_array(), (HEIGHT, WIDTH))
    assert kept.dtype == np.bool_
    assert kept[0, 0]
    assert not kept[EDIT_BOX][0, 0]
    with pytest.raises(InpaintVerificationError):
        review_protected_region(_mask_array(), (8, 8))


# --------------------------------------------------------------- verification


def test_verify_inpaint_output_restores_and_passes(tmp_path: Path) -> None:
    original = _flat((120, 130, 140))
    generated = original.copy()
    generated[EDIT_BOX] = (30, 200, 90)

    original_path = _save_rgb(tmp_path / "original.png", original)
    generated_path = _save_rgb(tmp_path / "generated.png", generated)
    mask_path = _save_mask(tmp_path / "mask.png", _mask_array())
    restore_path = tmp_path / "restored" / "final.png"

    result = verify_inpaint_output(
        original_path,
        generated_path,
        mask_path,
        restore_path=restore_path,
        identity_guard=_guard_array(),
    )

    assert result.ok is True
    assert result.mask_valid is True
    assert result.pixel_identity_passed is True
    assert result.visual_acceptance_pending is True
    assert result.aligned is True
    assert result.best_shift == (0, 0)
    assert result.kept_pixels == HEIGHT * WIDTH - 100
    assert result.restored_path == restore_path
    assert restore_path.is_file()

    restored = np.asarray(Image.open(restore_path).convert("RGB"))
    # The repainted clothing is the model's; every kept pixel is the original's.
    assert restored[35, 45].tolist() == [30, 200, 90]
    assert np.array_equal(restored[0:20, 0:20], original[0:20, 0:20])
    assert np.array_equal(restored[EDIT_BOX][0, 0].tolist(), [30, 200, 90])


def test_verify_inpaint_output_rejects_a_different_size(tmp_path: Path) -> None:
    original_path = _save_rgb(tmp_path / "original.png", _pattern())
    generated_path = _save_rgb(tmp_path / "generated.png", _pattern(height=HEIGHT - 6, width=WIDTH - 6))
    mask_path = _save_mask(tmp_path / "mask.png", _mask_array())
    with pytest.raises(InpaintVerificationError):
        verify_inpaint_output(original_path, generated_path, mask_path)


def test_verify_inpaint_output_rejects_a_systematic_shift(tmp_path: Path) -> None:
    original = _pattern()
    generated = np.roll(original, shift=2, axis=0)
    original_path = _save_rgb(tmp_path / "original.png", original)
    generated_path = _save_rgb(tmp_path / "generated.png", generated)
    mask_path = _save_mask(tmp_path / "mask.png", _mask_array())
    restore_path = tmp_path / "restored.png"

    result = verify_inpaint_output(
        original_path, generated_path, mask_path, restore_path=restore_path
    )

    # The mask itself is fine; the *output* is what failed to register.
    assert result.mask_valid is True
    assert result.aligned is False
    assert result.pixel_identity_passed is False
    assert result.ok is False
    assert any("shifted" in reason for reason in result.reasons)
    # Nothing is written and nothing is presented as a final picture.
    assert result.restored_path is None
    assert not restore_path.exists()


def test_verify_inpaint_output_rejects_a_repainted_kept_region(tmp_path: Path) -> None:
    original = _flat((100, 100, 100))
    generated = _flat((250, 250, 250))  # the model repainted everything
    original_path = _save_rgb(tmp_path / "original.png", original)
    generated_path = _save_rgb(tmp_path / "generated.png", generated)
    mask_path = _save_mask(tmp_path / "mask.png", _mask_array())
    restore_path = tmp_path / "restored.png"

    result = verify_inpaint_output(
        original_path, generated_path, mask_path, restore_path=restore_path
    )

    assert result.aligned is False
    assert result.pixel_identity_passed is False
    assert any("repainted pixels the mask kept" in reason for reason in result.reasons)
    assert not restore_path.exists()


def test_verify_inpaint_output_rejects_an_edit_that_reaches_the_face(tmp_path: Path) -> None:
    original = _flat((120, 130, 140))
    generated = original.copy()
    mask = _mask_array()
    mask[22:26, 24:28] = 255  # the mask opens a hole over the face guard
    original_path = _save_rgb(tmp_path / "original.png", original)
    generated_path = _save_rgb(tmp_path / "generated.png", generated)
    mask_path = _save_mask(tmp_path / "mask.png", mask)
    restore_path = tmp_path / "restored.png"

    result = verify_inpaint_output(
        original_path,
        generated_path,
        mask_path,
        restore_path=restore_path,
        identity_guard=_guard_array(),
    )

    assert result.mask_valid is False
    assert result.pixel_identity_passed is False
    assert result.ok is False
    assert result.mask.identity_guard_overlap == 16
    assert not restore_path.exists()


def test_statuses_are_independent_and_never_conflated() -> None:
    ok_review = review_mask(_mask_array(), (WIDTH, HEIGHT))
    from app.domains.portrait.inpaint_verify import InpaintVerification

    passing = InpaintVerification(
        mask=ok_review,
        aligned=True,
        kept_pixels=1,
        kept_mean_abs_diff=0.0,
        best_shift=(0, 0),
        kept_mean_abs_diff_at_best_shift=0.0,
    )
    assert passing.statuses == {
        "MASK_VALID": True,
        "PIXEL_IDENTITY_PASSED": True,
        "VISUAL_ACCEPTANCE_PENDING": True,
    }
    assert passing.ok is True
    # Even a machine-perfect composite still awaits a human eye.
    assert passing.visual_acceptance_pending is True
    assert passing.to_dict()["statuses"]["VISUAL_ACCEPTANCE_PENDING"] is True

    failing = InpaintVerification(
        mask=ok_review,
        aligned=False,
        kept_pixels=1,
        kept_mean_abs_diff=99.0,
        best_shift=(0, 2),
        kept_mean_abs_diff_at_best_shift=99.0,
    )
    # The mask can be valid while the pixel-identity gate fails; the two statuses
    # must not collapse into one another.
    assert failing.statuses["MASK_VALID"] is True
    assert failing.statuses["PIXEL_IDENTITY_PASSED"] is False
    assert failing.statuses["VISUAL_ACCEPTANCE_PENDING"] is True
    assert failing.ok is False


def test_verify_inpaint_output_uses_a_narrowed_probe(tmp_path: Path) -> None:
    original = _pattern()
    generated = original.copy()
    original_path = _save_rgb(tmp_path / "original.png", original)
    generated_path = _save_rgb(tmp_path / "generated.png", generated)
    mask_path = _save_mask(tmp_path / "mask.png", _mask_array())

    result = verify_inpaint_output(
        original_path,
        generated_path,
        mask_path,
        region_of_interest=_guard_array(),
    )
    assert result.diagnostics["probeIsIdentityGuard"] is True
    assert result.diagnostics["probePixels"] == 10 * 12
    assert result.ok is True


def test_verify_inpaint_output_without_a_restore_path_writes_nothing(tmp_path: Path) -> None:
    original = _flat((120, 130, 140))
    generated = original.copy()
    generated[EDIT_BOX] = (10, 10, 10)
    result = verify_inpaint_output(
        _save_rgb(tmp_path / "original.png", original),
        _save_rgb(tmp_path / "generated.png", generated),
        _save_mask(tmp_path / "mask.png", _mask_array()),
    )
    assert result.ok is True
    assert result.restored_path is None
    assert sorted(p.name for p in tmp_path.iterdir()) == ["generated.png", "mask.png", "original.png"]
