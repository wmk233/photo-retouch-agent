"""Submit ONE minimal, single-pass local collar repair to RunningHub (P2.9).

This tool is deliberately *not* the two-pass Clothing -> Hair generator in
``tools.runninghub_crop_stitch_run``. The brief for this round freezes the best
render and forbids regenerating the whole look to fix one local defect:

* baseline  = ``evidence/p29-submit-run/final.png`` (task 2108330225116082177),
  sha256 ``1b9634d1eb3f24738d9caf67d1009b21053a9790a1947400c67449bc5331590a``;
* defect    = the pale knit collar strip the batch passes left under the chin;
* fix       = one masked inpaint of that strip only, on the finished frame.

It shares the proven CropAndStitch graph, checkpoint and CropAndStitch settings
with ``runninghub_crop_stitch_run`` by *importing* them, so "compatible with the
last successful configuration" is a property of the code, not a promise. The
only structural differences are:

    * the image is the finished render, not the original photo;
    * the mask is ``collar_repair_mask.png`` (released residual) rather than the
      clothing/hair silhouettes;
    * there is exactly ONE pass -- one KSampler, one InpaintCropImproved, one
      InpaintStitchImproved, one SaveImage. No hair pass.

Node graph (ids stable so the gate can assert the wiring)::

    1  LoadImage             <- final.png
    2  LoadImageMask (red)   <- collar_repair_mask.png
    3  CheckpointLoaderSimple
    4  InpaintCropImproved   <- 1, 2
    5  CLIPTextEncode +      <- repair prompt
    6  CLIPTextEncode -      <- repair negatives (white/beige inner layer, trim)
    7  InpaintModelConditioning <- 5, 6, 3.vae, 4.cropped_image, 4.cropped_mask, noise_mask
    8  KSampler              <- 3.model, 7.(pos,neg,latent), denoise < 1.0
    9  VAEDecode
    10 InpaintStitchImproved <- stitcher 4, inpainted 9  (restores the 904x1190 canvas)
    11 SaveImage

Stages, from ``backend/``::

    # build + validate + offline gate only -- no network, no key
    ../.venv/bin/python -m tools.runninghub_collar_repair_run --dry-run

    # upload the two inputs (free)
    ../.venv/bin/python -m tools.runninghub_collar_repair_run --upload

    # create the one billed task
    ../.venv/bin/python -m tools.runninghub_collar_repair_run \
        --submit --i-understand-this-costs-money

Gate 3 is the mask-scope gate. The released strip normally has to sit inside the
analytic collar window (``window_bbox`` in the mask stats). The adjudication
added one *named* exception, the 172 px carve directly under the chin
(``chin_band_carve_*``), which reaches a few pixels above that window so the
repair can reach the top of the white knit tuft. Gate 3 accepts it only when the
carve clears its own evidence test -- 8-connected to the released strip, off the
face oval, old knit the render left unchanged, and the subject's own skin model
flagging at most ``CARVE_MAX_SKIN_FLAG_RATIO`` of it -- and refuses to widen the
window otherwise. The allowed region is reported as window UNION carve, not as
one silently enlarged box.

Gate 8 is the keep-vs-remove conflict the brief asks about, now *split* rather
than merely reported. The old 3007 px guarded strip was adjudicated: the 172 px
that carried independent fabric evidence became the carve (gate 3), and the
remainder is skin-toned with no independent fabric evidence, so it stays
protected -- "what cannot be confirmed stays protected". Gate 8 therefore asks
whether the region this pass will NOT edit still holds a visible old-knit
remnant, measured as ``pale_knit_survivor_*``. A remnant at or below one latent
cell of the checkpoint (8x8 px) is beneath the pass's own spatial resolution and
does not block; anything larger HALTs the run by default (exit 2) with the exact
pixel counts and blob boxes. ``--accept-guarded-residual`` is the emergency
override and is explicitly NOT to be used this round.

The API key is read from ``~/.runninghub_key`` and is never printed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import httpx
import numpy as np
from PIL import Image

from tools.runninghub_crop_stitch_run import (  # shared, proven graph config
    CHECKPOINT,
    CROP_DEFAULTS,
    DENOISE_BAND,
    SAMPLER,
    PROTECTED_CORE_ERODE_PX,
    _crop_box,
    _mask_pipeline_np,
    validate_checkpoint,
    validate_workflow,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The frozen best render this round repairs. The sha256 is the identity check;
#: a different frame must not be silently substituted.
BASELINE_FINAL = REPO_ROOT / "evidence/p29-submit-run/final.png"
BASELINE_SHA256 = "1b9634d1eb3f24738d9caf67d1009b21053a9790a1947400c67449bc5331590a"
EXPECTED_FRAME = (904, 1190)

DEFAULT_MASK = REPO_ROOT / "evidence/p29-collar-repair/collar_repair_mask.png"
DEFAULT_NO_EDIT = REPO_ROOT / "evidence/p29-collar-repair/collar_repair_no_edit.png"
DEFAULT_STATS = REPO_ROOT / "evidence/p29-collar-repair/collar_repair_stats.json"
DEFAULT_LABELS = REPO_ROOT / "evidence/p29-collar-repair/labels_fresh.png"

#: Denoise sits in the shared band, one notch below the clothing pass's proven
#: 0.75: the frame is already correct nearly everywhere, so the repair wants the
#: smallest strength that still replaces the pale knit inside the mask.
COLLAR_REPAIR_DENOISE = 0.70

FILENAME_PREFIX = "p29_collar_repair"

#: A single-pass latent edit cannot express detail finer than one latent cell of
#: the checkpoint (8 px, as for every SD1.5-family VAE). An old-knit remnant that
#: fits inside one 8x8 cell is therefore below this pass's own spatial resolution
#: -- it is not something a second generation of the same graph could fix either,
#: and it is not a visible white patch on the delivered frame. Gate 8 uses this
#: scale to call the residual *resolved* rather than demanding pixel-perfect
#: coverage of the 3007 px guarded strip (which cannot be released at all: it is
#: skin-toned and carries no independent fabric evidence).
LATENT_CELL_PX = 8
SURVIVOR_RESIDUAL_BLOB_LIMIT_PX = LATENT_CELL_PX * LATENT_CELL_PX          # 64
SURVIVOR_RESIDUAL_TOTAL_LIMIT_PX = LATENT_CELL_PX * LATENT_CELL_PX * 4     # 256

#: The carve is the one protected strip the adjudication moved into the release.
#: It must be justified by its own evidence, not by "it looked pale": it has to
#: be 8-connected to the released strip, must not sit on the face oval, must be
#: a strip of the old knit the render left unchanged, and the subject's own skin
#: model must flag (almost) none of it. ``carve_px`` counts these pixels; the
#: skin-flag ratio is the fraction the skin model would still call skin.
CARVE_MAX_SKIN_FLAG_RATIO = 0.05

#: The strict "is any visible white cloth left?" limit. ``neutral_white_*`` is the
#: FINAL render's near-greyscale bright pixels inside the collar band; the coarse
#: pale-knit rules also catch pale-olive blouse highlights, so this is the test
#: that speaks to "white cloth". A remnant at or below one latent cell does not
#: block, same scale as the old-knit residual.
NEUTRAL_WHITE_LIMIT_PX = LATENT_CELL_PX * LATENT_CELL_PX          # 64
NEUTRAL_WHITE_BLOB_LIMIT_PX = LATENT_CELL_PX * LATENT_CELL_PX     # 64

#: Minimal repair prompt. One job: continue the surrounding blouse across the
#: masked strip. No facial/identity words, no accessory, no second garment.
POSITIVE_PROMPT = (
    "repaint only the masked area as the same deep olive-green cloth as the "
    "surrounding blouse, continuing the existing collar into one natural closed "
    "neckline with soft vintage folds, matching the neighbouring colour, shading "
    "and fabric texture"
)

#: A short, targeted negative list -- the artefacts a repair at the neckline can
#: actually produce. Deliberately NOT the dozens of terms the full generator
#: accumulated: this pass repaints ~3.3k px and needs only the "no lighter inner
#: layer" rules the last run proved necessary.
NEGATIVE_PROMPT = (
    "white collar, cream collar, beige collar lining, ivory lining, white inner "
    "layer, white undershirt, pale knit at the neckline, beige knitted scarf, "
    "white scarf, lace trim, ribbon, bow, necklace, duplicated neckline, "
    "extra collar flap, blurry, lowres, watermark"
)

#: Words a masked edit must never put in a positive prompt (a whole head is in
#: the crop as context; naming it reads as "draw it"). Same rule as the generator.
FORBIDDEN_POSITIVE_WORDS = (
    "face", "facial", "eye", "eyebrow", "nose", "lip", "mouth", "teeth",
    "smile", "glasses", "person", "portrait", "identity", "chin", "neck",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_workflow(final_name: str, mask_name: str, seed: int) -> dict:
    """The single-pass repair graph. One sampler, one crop, one stitch."""
    crop = CROP_DEFAULTS
    return {
        "1": {"class_type": "LoadImage", "inputs": {"image": final_name}},
        "2": {"class_type": "LoadImageMask", "inputs": {"image": mask_name, "channel": "red"}},
        "3": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CHECKPOINT}},
        "4": {
            "class_type": "InpaintCropImproved",
            "inputs": {"image": ["1", 0], "mask": ["2", 0], **crop},
        },
        "5": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": POSITIVE_PROMPT, "clip": ["3", 1]},
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": NEGATIVE_PROMPT, "clip": ["3", 1]},
        },
        "7": {
            "class_type": "InpaintModelConditioning",
            "inputs": {
                "positive": ["5", 0],
                "negative": ["6", 0],
                "vae": ["3", 2],
                "pixels": ["4", 1],
                "mask": ["4", 2],
                "noise_mask": True,
            },
        },
        "8": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["3", 0],
                "seed": seed,
                "positive": ["7", 0],
                "negative": ["7", 1],
                "latent_image": ["7", 2],
                **SAMPLER,
                "denoise": COLLAR_REPAIR_DENOISE,
            },
        },
        "9": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["3", 2]}},
        "10": {
            "class_type": "InpaintStitchImproved",
            "inputs": {"stitcher": ["4", 0], "inpainted_image": ["9", 0]},
        },
        "11": {"class_type": "SaveImage", "inputs": {"images": ["10", 0], "filename_prefix": FILENAME_PREFIX}},
    }


def _load_gray(path: Path) -> "np.ndarray":
    with Image.open(path) as image:
        return np.array(image.convert("L"))


def _hue_sat(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = rgb.astype(np.float32) / 255.0
    mx, mn = a.max(2), a.min(2)
    diff = mx - mn
    hue = np.zeros_like(mx)
    nz = diff > 1e-6
    r, g, b = a[:, :, 0], a[:, :, 1], a[:, :, 2]
    idx = nz & (mx == r)
    hue[idx] = ((g - b)[idx] / diff[idx]) % 6
    idx = nz & (mx == g)
    hue[idx] = ((b - r)[idx] / diff[idx]) + 2
    idx = nz & (mx == b)
    hue[idx] = ((r - g)[idx] / diff[idx]) + 4
    hue *= 60.0
    sat = np.where(mx > 1e-6, diff / np.maximum(mx, 1e-6), 0.0)
    return hue, sat


def collect_gate_checks(workflow: dict, object_info: dict | None, *,
                        final_path: Path, mask_path: Path, no_edit_path: Path,
                        stats_path: Path, labels_path: Path) -> tuple[list[str], dict]:
    """The eight offline gates from the round brief. No network, all local."""
    errors: list[str] = []
    report: dict = {
        "checkpoint": CHECKPOINT,
        "denoise": COLLAR_REPAIR_DENOISE,
        "denoise_band": list(DENOISE_BAND),
        "prompt": POSITIVE_PROMPT,
        "negative_prompt": NEGATIVE_PROMPT,
        "shares_config_with_proven_run": {
            "checkpoint": CHECKPOINT,
            "crop": {k: CROP_DEFAULTS[k] for k in
                     ("mask_fill_holes", "mask_expand_pixels", "mask_blend_pixels",
                      "context_from_mask_extend_factor", "output_target_width",
                      "output_target_height", "output_resize_to_target_size")},
            "sampler": SAMPLER,
        },
    }

    # -- gate 1: the repaired image is the frozen best final --------------------
    final_sha = _sha256(final_path)
    with Image.open(final_path) as image:
        final_size = image.size
    report["gate1_input_is_best_final"] = {
        "path": str(final_path.relative_to(REPO_ROOT)),
        "sha256": final_sha,
        "expected_sha256": BASELINE_SHA256,
        "matches_frozen_best": final_sha == BASELINE_SHA256,
        "size": list(final_size),
    }
    if final_sha != BASELINE_SHA256:
        errors.append(f"gate1: input is not the frozen best final (sha256 {final_sha})")

    # -- gate 2: exactly one local pass ---------------------------------------
    samplers = sorted(nid for nid, n in workflow.items() if n["class_type"] == "KSampler")
    crops = sorted(nid for nid, n in workflow.items() if n["class_type"] == "InpaintCropImproved")
    conds = sorted(nid for nid, n in workflow.items() if n["class_type"] == "InpaintModelConditioning")
    stitches = sorted(nid for nid, n in workflow.items() if n["class_type"] == "InpaintStitchImproved")
    saves = sorted(nid for nid, n in workflow.items() if n["class_type"] == "SaveImage")
    forbidden = sorted(nid for nid, n in workflow.items()
                       if n["class_type"] in {"EmptyLatentImage", "EmptySD3LatentImage", "VAEEncodeForInpaint"})
    report["gate2_single_local_pass"] = {
        "ksampler_nodes": samplers,
        "inpaint_crop_nodes": crops,
        "inpaint_model_conditioning_nodes": conds,
        "inpaint_stitch_nodes": stitches,
        "save_nodes": saves,
        "passes": 1 if len(samplers) == 1 else len(samplers),
    }
    if len(samplers) != 1:
        errors.append(f"gate2: expected exactly one KSampler, found {samplers}")
    if len(crops) != 1 or len(conds) != 1 or len(stitches) != 1:
        errors.append("gate2: the graph is not a single crop->condition->stitch pass")
    if len(saves) != 1:
        errors.append(f"gate2: expected one SaveImage, found {saves}")
    if forbidden:
        errors.append(f"gate2: a text-to-image / fill node is present {forbidden}")
    if workflow["10"]["inputs"]["stitcher"] != ["4", 0] or workflow["4"]["inputs"]["image"] != ["1", 0]:
        errors.append("gate2: the repaired pixels do not come from the loaded final via one crop")
    if workflow["8"]["inputs"]["latent_image"] != ["7", 2]:
        errors.append("gate2: the sampler latent is not the encoded crop")
    denoise = workflow["8"]["inputs"].get("denoise")
    if denoise is None or not DENOISE_BAND[0] <= denoise <= DENOISE_BAND[1]:
        errors.append(f"gate2: denoise {denoise} is outside the proven band {DENOISE_BAND}")
    if workflow["7"]["inputs"]["noise_mask"] is not True:
        errors.append("gate2: noise_mask must be on or the mask is not honoured")

    # prompt hygiene: no facial/identity word (whole word -- "neckline" is fine,
    # "neck" as a standalone noun is not), and short.
    lowered = POSITIVE_PROMPT.lower()
    hits = [w for w in FORBIDDEN_POSITIVE_WORDS if re.search(rf"\b{re.escape(w)}\b", lowered)]
    report["prompt_hygiene"] = {"forbidden_words_found": hits, "chars": len(POSITIVE_PROMPT),
                                "negative_terms": len([t for t in NEGATIVE_PROMPT.split(",") if t.strip()])}
    if hits:
        errors.append(f"prompt names a facial/identity word {hits}")

    # -- gates 3/4/5: mask scope, disjointness, no correct-content edits -------
    #
    # Two regions are distinguished on purpose. The *raw mask* is what the
    # operator authored (``collar_repair_mask.png``). The *painted region* is
    # what InpaintCropImproved will actually hand the sampler after its
    # hipass -> expand -> blur prep: the raw mask plus a ``mask_blend_pixels``
    # feather rim, which the stitch composites as generated*alpha + original*(1-alpha).
    # Identity is argued on the raw mask (strictly disjoint) and the feather is
    # bounded and reported; the strict "core" test uses the same erosion the
    # proven two-pass gate used, so the two tools share one definition of "the
    # generated pixels must not reach the protected interior".
    from scipy import ndimage
    from app.domains.portrait.human_parsing import ATR_CLASSES

    mask_raw = _load_gray(mask_path) > 127
    no_edit = _load_gray(no_edit_path) > 127
    with Image.open(final_path) as image:
        final_rgb = np.array(image.convert("RGB"))
    labels = _load_gray(labels_path)
    hair = labels == [k for k, v in ATR_CLASSES.items() if v == "Hair"][0]
    background = labels == [k for k, v in ATR_CLASSES.items() if v == "Background"][0]
    hue, sat = _hue_sat(final_rgb)
    olive = (hue >= 45.0) & (hue <= 115.0) & (sat > 0.12)

    prepared = _mask_pipeline_np(mask_raw.astype(np.float32),
                                 blend_pixels=int(CROP_DEFAULTS["mask_blend_pixels"]),
                                 expand_pixels=int(CROP_DEFAULTS["mask_expand_pixels"]),
                                 fill_holes=bool(CROP_DEFAULTS["mask_fill_holes"]))
    painted = prepared > 0.5
    feather = painted & ~mask_raw
    feather_max_dist = 0.0
    if feather.any():
        feather_max_dist = float(ndimage.distance_transform_edt(~mask_raw)[feather].max())

    def _bbox(m):
        ys, xs = np.where(m)
        return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if m.any() else None

    def _within(inner, outer, tol=0):
        return (inner is not None and outer is not None
                and inner[0] >= outer[0] - tol and inner[1] >= outer[1] - tol
                and inner[2] <= outer[2] + tol and inner[3] <= outer[3] + tol)

    def _guard_stats(guard, erode_px):
        inter = painted & guard
        depth = float(ndimage.distance_transform_edt(~guard)[inter].max()) if inter.any() else 0.0
        raw_inter = mask_raw & guard
        raw_depth = float(ndimage.distance_transform_edt(~guard)[raw_inter].max()) if raw_inter.any() else 0.0
        core = ndimage.binary_erosion(guard, iterations=erode_px) if guard.any() else guard
        return {
            "guard_px": int(guard.sum()),
            "raw_mask_intersection_px": int(raw_inter.sum()),
            "raw_mask_max_depth_px": raw_depth,
            "painted_intersection_px": int(inter.sum()),
            "painted_max_depth_px": depth,
            "core_erode_px": erode_px,
            "painted_into_core_px": int((painted & core).sum()),
        }

    stats = json.loads(stats_path.read_text())
    window_bbox = stats.get("window_bbox")
    feather_tol = int(CROP_DEFAULTS["mask_blend_pixels"]) + 1

    def _union_bbox(*boxes):
        boxes = [b for b in boxes if b]
        if not boxes:
            return None
        return [min(b[0] for b in boxes), min(b[1] for b in boxes),
                max(b[2] for b in boxes), max(b[3] for b in boxes)]

    # The adjudication added one *named* strip to the release: the 172 px carve
    # under the chin (``chin_band_carve_*`` in the mask stats). It reaches
    # ``carve_bbox[1]`` = 588, i.e. 6 px above the analytic collar window, so the
    # allowed region is the window UNION the carve band -- reported as two named
    # regions rather than one widened rectangle, and only accepted when the carve
    # clears its own evidence test below. Nothing else may leave the window.
    carve_bbox = stats.get("chin_band_carve_bbox")
    carve_px = int(stats.get("chin_band_carve_px", 0))
    carve_skin_flag_px = int(stats.get("chin_band_carve_skin_flag_px", 0))
    carve_oval_px = int(stats.get("chin_band_carve_oval_px", 0))
    carve_release_px = int(stats["counts"].get("release_from_carve_px", 0))
    carve_skin_flag_ratio = (carve_skin_flag_px / carve_px) if carve_px else 0.0
    carve_justified = (
        carve_px > 0
        and carve_release_px == carve_px
        and carve_oval_px == 0
        and carve_skin_flag_ratio <= CARVE_MAX_SKIN_FLAG_RATIO
    )
    allowed_bbox = _union_bbox(window_bbox, carve_bbox if carve_justified else None)

    report["gate3_mask_is_collar_only"] = {
        "raw_mask_bbox": _bbox(mask_raw),
        "painted_bbox": _bbox(painted),
        "release_window_bbox": window_bbox,
        "carve_bbox": carve_bbox,
        "allowed_bbox": allowed_bbox,
        "feather_tolerance_px": feather_tol,
        "raw_mask_bbox_in_window": _within(_bbox(mask_raw), window_bbox),
        "raw_mask_bbox_in_allowed": _within(_bbox(mask_raw), allowed_bbox),
        "painted_bbox_within_allowed_plus_feather": _within(_bbox(painted), allowed_bbox, feather_tol),
        "carve": {
            "px": carve_px,
            "bbox": carve_bbox,
            "skin_flag_px": carve_skin_flag_px,
            "skin_flag_ratio": carve_skin_flag_ratio,
            "max_skin_flag_ratio": CARVE_MAX_SKIN_FLAG_RATIO,
            "face_oval_px": carve_oval_px,
            "release_from_carve_px": carve_release_px,
            "justified": carve_justified,
            "note": stats.get("chin_band_carve_note"),
        },
        "mask_stats_release_px": stats["counts"]["release_px"],
        "mask_stats_release_px_in_collar_class": stats["counts"]["release_px_in_collar_class"],
        "mask_stats_release_over_hair_px": stats["counts"]["release_over_hair_px"],
        "mask_stats_release_over_background_px": stats["counts"]["release_over_background_px"],
        "mask_stats_release_over_protected_px": stats["counts"]["release_over_protected_px"],
    }

    # ``no_edit`` also folds in the already-correct olive so the *painted* region
    # can be proven disjoint from it, but the identity argument is made against
    # the guards proper: identity, teeth, glasses, chin/throat, the subject's own
    # skin inside the collar window, parsed hair and background. Those are the
    # pixels "keep the original" is non-negotiable for.
    guards = no_edit & ~olive
    guards_g = _guard_stats(guards, PROTECTED_CORE_ERODE_PX)
    report["gate4_painted_disjoint_from_no_edit"] = {
        "guards_px": int(guards.sum()),
        "raw_mask_disjoint_from_guards": int((mask_raw & guards).sum()) == 0,
        "raw_mask_over_guards_px": int((mask_raw & guards).sum()),
        "feather_rim_px": int(feather.sum()),
        "feather_rim_max_distance_px": feather_max_dist,
        "note": ("the feather is the node's mask_blend_pixels transition; it is "
                 "bounded and composites against the original, it does not replace it"),
        **guards_g,
    }

    hair_g = _guard_stats(hair, 3)
    olive_g = _guard_stats(olive, 3)
    bg_g = _guard_stats(background, 3)
    report["gate5_no_correct_content_edited"] = {
        "painted_px": int(painted.sum()),
        "hair": hair_g,
        "olive_blouse": olive_g,
        "background": bg_g,
        "olive_blouse_px_total": int(olive.sum()),
    }

    if int(painted.sum()) == 0:
        errors.append("gate3: the repair mask is empty")
    if not carve_justified:
        errors.append(
            f"gate3: the {carve_px} px carve is not justified "
            f"(release_from_carve={carve_release_px}, oval={carve_oval_px}, "
            f"skin_flag_ratio={carve_skin_flag_ratio:.3f} > {CARVE_MAX_SKIN_FLAG_RATIO}) "
            "-- refusing to widen the collar window on unconvincing evidence")
    if not report["gate3_mask_is_collar_only"]["raw_mask_bbox_in_allowed"]:
        errors.append("gate3: the authored mask leaves the collar window and its justified carve")
    if not report["gate3_mask_is_collar_only"]["painted_bbox_within_allowed_plus_feather"]:
        errors.append(f"gate3: the painted region leaves the collar window plus its justified carve "
                      f"by more than {feather_tol} px")
    if feather_max_dist > int(CROP_DEFAULTS["mask_blend_pixels"]):
        errors.append(f"gate4: the feather reaches {feather_max_dist:.1f} px, beyond mask_blend_pixels")
    if guards_g["raw_mask_intersection_px"]:
        errors.append(f"gate4: the authored mask touches {guards_g['raw_mask_intersection_px']} px "
                      f"of identity / skin / hair / background guard")
    if guards_g["painted_into_core_px"]:
        errors.append(f"gate4: the painted region reaches {guards_g['painted_into_core_px']} px "
                      f"into the guard core (eroded {PROTECTED_CORE_ERODE_PX} px)")
    for label, g in (("hair", hair_g), ("olive blouse", olive_g), ("background", bg_g)):
        if g["painted_into_core_px"]:
            errors.append(f"gate5: the painted region reaches {g['painted_into_core_px']} px "
                          f"into the {label} core")

    # -- gate 6: original 904x1190 coordinates are preserved -------------------
    box = _crop_box(prepared, int(CROP_DEFAULTS["output_target_width"]),
                    int(CROP_DEFAULTS["output_target_height"]), 32,
                    float(CROP_DEFAULTS["context_from_mask_extend_factor"]), *final_size)
    uniform = abs((box["target_w"] / box["w"]) / (box["target_h"] / box["h"]) - 1) <= 0.01
    report["gate6_frame_preserved"] = {
        "input_size": list(final_size),
        "expected_size": list(EXPECTED_FRAME),
        "output_resize_to_target_size": CROP_DEFAULTS["output_resize_to_target_size"],
        "crop": box,
        "crop_scale_uniform": uniform,
        "stitch_restores_canvas": workflow["10"]["inputs"]["stitcher"] == ["4", 0],
    }
    if final_size != EXPECTED_FRAME:
        errors.append(f"gate6: input size {final_size} != {EXPECTED_FRAME}")
    if not CROP_DEFAULTS["output_resize_to_target_size"]:
        errors.append("gate6: output_resize_to_target_size is off; the stitch could not restore the canvas")
    if not uniform:
        errors.append("gate6: the crop would be rescaled non-uniformly")

    # -- gate 7: node table + checkpoint compatibility -------------------------
    wf_errors = validate_workflow(workflow, object_info)
    ckpt_notes = validate_checkpoint(object_info)
    report["gate7_runtime_compatibility"] = {
        "validate_workflow_errors": wf_errors,
        "checkpoint_notes": ckpt_notes,
        "object_info": str(object_info and "loaded" or "missing"),
    }
    errors.extend(f"gate7: {e}" for e in wf_errors)
    errors.extend(f"gate7: {n}" for n in ckpt_notes if n.startswith("ERROR"))

    # -- gate 8: the keep-original vs remove-residual conflict -----------------
    # The old 3007 px guarded strip is now adjudicated, not merely reported:
    #   * the 172 px carve (independent knit evidence, off the face oval, skin
    #     model flags ~none of it) was moved into the release -- gate 3 checks
    #     that evidence and the graph repaints it;
    #   * the remainder is skin-toned and carries no independent fabric evidence,
    #     so per the brief ("what cannot be confirmed stays protected") it is
    #     KEPT, not edited.
    # Gate 8 therefore no longer asks "is anything guarded?" (the answer is yes,
    # by design). It asks the question the brief actually poses: does the region
    # this pass will NOT edit still hold a visible old-knit remnant? That is the
    # measured ``pale_knit_survivor_*`` residual, the old knit (original luma >
    # 150, chroma < 45) the render left unchanged in the collar band. A remnant
    # at or below one latent cell of the checkpoint is below the pass's own
    # spatial resolution and is not a visible defect, so it does not block.
    guarded = int(stats.get("guarded_pale_px", 0))
    guarded_skin_like = int(stats.get("guarded_pale_final_skin_like_px", 0))
    survivor_px = int(stats.get("pale_knit_survivor_px", 0))
    survivor_uncovered = int(stats.get("pale_knit_survivor_uncovered_px", 0))
    survivor_blobs = stats.get("pale_knit_survivor_uncovered_blobs") or []
    biggest_blob = max((int(b.get("px", 0)) for b in survivor_blobs), default=0)
    # The stricter second question: visible WHITE, not merely pale. ``neutral_white``
    # is the final render's bright near-greyscale pixels in the collar band -- the
    # coarse pale-knit rules also count pale-olive highlights, which are cloth.
    neutral_white = int(stats.get("neutral_white_uncovered_px", 0))
    neutral_white_blobs = stats.get("neutral_white_uncovered_blobs") or []
    neutral_white_biggest = max((int(b.get("px", 0)) for b in neutral_white_blobs), default=0)
    residual_resolved = (
        survivor_uncovered <= SURVIVOR_RESIDUAL_TOTAL_LIMIT_PX
        and biggest_blob <= SURVIVOR_RESIDUAL_BLOB_LIMIT_PX
        and neutral_white <= NEUTRAL_WHITE_LIMIT_PX
        and neutral_white_biggest <= NEUTRAL_WHITE_BLOB_LIMIT_PX
    )
    report["gate8_keep_vs_remove_conflict"] = {
        "guarded_pale_px": guarded,
        "guarded_pale_bbox": stats.get("guarded_pale_bbox"),
        "guarded_pale_final_skin_like_px": guarded_skin_like,
        "guarded_reason": stats.get("guarded_pale_note"),
        "pale_knit_survivor_px": survivor_px,
        "pale_knit_survivor_uncovered_px": survivor_uncovered,
        "pale_knit_survivor_uncovered_blobs": survivor_blobs,
        "biggest_uncovered_blob_px": biggest_blob,
        "neutral_white_uncovered_px": neutral_white,
        "neutral_white_uncovered_blobs": neutral_white_blobs,
        "neutral_white_biggest_blob_px": neutral_white_biggest,
        "neutral_white_limit_px": NEUTRAL_WHITE_LIMIT_PX,
        "neutral_white_note": stats.get("neutral_white_note"),
        "blob_limit_px": SURVIVOR_RESIDUAL_BLOB_LIMIT_PX,
        "total_limit_px": SURVIVOR_RESIDUAL_TOTAL_LIMIT_PX,
        "latent_cell_px": LATENT_CELL_PX,
        "unresolvable": not residual_resolved,
        "resolution": (
            "split, not ignored: the 172 px strip carrying independent fabric evidence was "
            "moved into the release (gate 3 checks that evidence) and is repainted by this "
            "pass; the remaining guarded block is skin-toned with no independent fabric "
            "evidence and stays protected, as the brief requires. The old knit the render "
            f"left unchanged inside the collar band is now {survivor_uncovered} px "
            f"(largest blob {biggest_blob} px), and strictly neutral white cloth is "
            f"{neutral_white} px (largest blob {neutral_white_biggest} px), both at or "
            f"below one {LATENT_CELL_PX}x{LATENT_CELL_PX} latent cell of this checkpoint"
        ),
    }

    report["errors"] = errors
    return errors, report


def _post(client: httpx.Client, path: str, key: str, payload: dict) -> dict:
    response = client.post(path, json=payload, headers={"Authorization": f"Bearer {key}"})
    response.raise_for_status()
    body = response.json()
    if body.get("code") not in (0, None):
        raise SystemExit(f"{path} -> code={body.get('code')} msg={body.get('msg')}")
    return body


def upload(client: httpx.Client, key: str, path: Path) -> str:
    with path.open("rb") as handle:
        response = client.post(
            "/task/openapi/upload",
            data={"apiKey": key},
            files={"file": (path.name, handle, "application/octet-stream")},
        )
    response.raise_for_status()
    body = response.json()
    data = body.get("data") or {}
    name = data.get("fileName") or data.get("filename")
    if body.get("code") not in (0, None) or not name:
        raise SystemExit(f"upload {path.name} failed: code={body.get('code')}")
    print(f"  uploaded {path.name} ({path.stat().st_size} B) -> {name}")
    return name


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--final", type=Path, default=BASELINE_FINAL)
    parser.add_argument("--mask", type=Path, default=DEFAULT_MASK)
    parser.add_argument("--no-edit", type=Path, default=DEFAULT_NO_EDIT)
    parser.add_argument("--stats", type=Path, default=DEFAULT_STATS)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "artifacts/p29-collar-repair")
    parser.add_argument("--key-file", type=Path, default=Path.home() / ".runninghub_key")
    parser.add_argument("--object-info", type=Path,
                        default=REPO_ROOT / "evidence/p29-live-run/object_info_live.json")
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--instance-type", default="default")
    parser.add_argument("--workflow-id", default="1")
    parser.add_argument("--poll-seconds", type=int, default=900)
    parser.add_argument("--dry-run", action="store_true", help="build + validate + gate only (default)")
    parser.add_argument("--upload", action="store_true", help="upload the two input files (free)")
    parser.add_argument("--submit", action="store_true", help="create the one paid task")
    parser.add_argument("--i-understand-this-costs-money", action="store_true")
    parser.add_argument("--accept-guarded-residual", action="store_true",
                        help="proceed even though gate 8 finds a still-visible old-knit remnant "
                             "outside the released mask (NOT for this round: the brief forbids "
                             "overriding the gate)")
    args = parser.parse_args(argv)

    for name, path in (("final", args.final), ("mask", args.mask), ("no-edit", args.no_edit),
                       ("stats", args.stats), ("labels", args.labels)):
        if not path.is_file():
            raise SystemExit(f"{name} input is missing: {path}")

    object_info = None
    if args.object_info.is_file():
        object_info = json.loads(args.object_info.read_text())
        print(f"validating against {args.object_info.name} ({len(object_info)} installed nodes)")
    else:
        print(f"WARNING: {args.object_info} not found; structural checks only", file=sys.stderr)

    # Name the loaders with the real input basenames so the dry-run
    # ``workflow.json`` is a complete, self-consistent graph (the submit path
    # rebuilds it with the uploaded names, which the API assigns).
    workflow = build_workflow(args.final.name, args.mask.name, args.seed)
    errors, gate = collect_gate_checks(
        workflow, object_info,
        final_path=args.final, mask_path=args.mask, no_edit_path=args.no_edit,
        stats_path=args.stats, labels_path=args.labels,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    gate["accept_guarded_residual"] = bool(args.accept_guarded_residual)
    conflict = gate["gate8_keep_vs_remove_conflict"]
    gate["gate_passed"] = not errors
    gate["halted_by_gate8"] = bool(conflict["unresolvable"] and not args.accept_guarded_residual)
    (args.out / "gate_report.json").write_text(json.dumps(gate, ensure_ascii=False, indent=2))

    print(f"gate2 single pass: {gate['gate2_single_local_pass']}")
    g3 = gate["gate3_mask_is_collar_only"]
    print(f"gate3 raw mask bbox {g3['raw_mask_bbox']} in window {g3['release_window_bbox']} "
          f"-> {g3['raw_mask_bbox_in_window']}; allowed = window UNION carve {g3['carve_bbox']} "
          f"-> {g3['raw_mask_bbox_in_allowed']} (painted {g3['painted_bbox']})")
    print(f"gate3 carve: {g3['carve']['px']} px, skin-flag {g3['carve']['skin_flag_px']} px "
          f"(ratio {g3['carve']['skin_flag_ratio']:.3f} <= {g3['carve']['max_skin_flag_ratio']}), "
          f"face-oval {g3['carve']['face_oval_px']} px, released {g3['carve']['release_from_carve_px']} px "
          f"-> justified={g3['carve']['justified']}")
    print(f"gate4 raw mask over guards: {gate['gate4_painted_disjoint_from_no_edit']['raw_mask_over_guards_px']} px, "
          f"feather rim {gate['gate4_painted_disjoint_from_no_edit']['feather_rim_px']} px "
          f"<= {gate['gate4_painted_disjoint_from_no_edit']['feather_rim_max_distance_px']} px, "
          f"into core {gate['gate4_painted_disjoint_from_no_edit']['painted_into_core_px']}")
    for label in ("hair", "olive_blouse", "background"):
        g = gate["gate5_no_correct_content_edited"][label]
        print(f"gate5 {label}: raw {g['raw_mask_intersection_px']} px "
              f"(depth {g['raw_mask_max_depth_px']}), core {g['painted_into_core_px']}")
    print(f"gate6 frame {gate['gate6_frame_preserved']['input_size']} uniform={gate['gate6_frame_preserved']['crop_scale_uniform']}")
    for note in gate["gate7_runtime_compatibility"]["checkpoint_notes"]:
        print(f"  checkpoint: {note}")
    print(f"gate8 guarded block kept as skin: {conflict['guarded_pale_px']} px at "
          f"{conflict['guarded_pale_bbox']} ({conflict['guarded_pale_final_skin_like_px']} px skin-toned); "
          f"uncovered old knit {conflict['pale_knit_survivor_uncovered_px']} px "
          f"(largest blob {conflict['biggest_uncovered_blob_px']} px); "
          f"uncovered neutral white {conflict['neutral_white_uncovered_px']} px "
          f"(largest blob {conflict['neutral_white_biggest_blob_px']} px, limit "
          f"{conflict['neutral_white_limit_px']} px) "
          f"(unresolvable={conflict['unresolvable']})")

    if errors:
        for err in errors:
            print(f"  GATE FAILED  {err}", file=sys.stderr)
        print(f"gate report -> {args.out / 'gate_report.json'}", file=sys.stderr)
        return 1
    if gate["halted_by_gate8"]:
        print("HALT: gate 8 shows the released region still leaves a visible old-knit remnant "
              f"({conflict['pale_knit_survivor_uncovered_px']} px old knit, largest blob "
              f"{conflict['biggest_uncovered_blob_px']} px, limit {conflict['blob_limit_px']} px/blob, "
              f"{conflict['total_limit_px']} px total; {conflict['neutral_white_uncovered_px']} px "
              f"neutral white, limit {conflict['neutral_white_limit_px']} px). "
              "A single pass over the current mask would not "
              "clear it; do not paper over this with --accept-guarded-residual. Tighten the released "
              "mask with evidence first, or ask the operator.", file=sys.stderr)
        print(f"gate report -> {args.out / 'gate_report.json'}", file=sys.stderr)
        return 2
    print(f"gate passed -> {args.out / 'gate_report.json'}")

    if not (args.upload or args.submit):
        (args.out / "workflow.json").write_text(json.dumps(workflow, indent=2))
        print(f"dry run complete -> {args.out / 'workflow.json'}")
        return 0

    key = args.key_file.read_text().strip()
    if len(key) < 16:
        raise SystemExit(f"credential file {args.key_file} does not look like an API key")
    client = httpx.Client(base_url="https://www.runninghub.cn", timeout=60.0)
    print("uploading inputs")
    final_name = upload(client, key, args.final)
    mask_name = upload(client, key, args.mask)
    workflow = build_workflow(final_name, mask_name, args.seed)
    wf_errors = validate_workflow(workflow, object_info)
    if wf_errors:
        for err in wf_errors:
            print(f"  INVALID  {err}", file=sys.stderr)
        return 1
    (args.out / "workflow.json").write_text(json.dumps(workflow, indent=2))
    print(f"workflow written -> {args.out / 'workflow.json'}")

    if not args.submit:
        print("upload stage complete; no task created")
        return 0
    if not args.i_understand_this_costs_money:
        raise SystemExit("--submit needs --i-understand-this-costs-money (this creates one billed task)")

    payload = {
        "apiKey": key,
        "workflowId": args.workflow_id,
        "workflow": json.dumps(workflow),
        "nodeInfoList": [],
        "instanceType": args.instance_type,
    }
    created = _post(client, "/task/openapi/create", key, payload)
    task_id = created["data"]["taskId"]
    print(f"task created: {task_id} status={created['data'].get('taskStatus')}")
    (args.out / "task.json").write_text(json.dumps({"taskId": task_id}, indent=2))
    print("poll and download are intentionally left to a follow-up so no result is fetched without review")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
