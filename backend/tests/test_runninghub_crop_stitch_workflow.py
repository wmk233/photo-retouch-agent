"""P2.9 fix round: the RunningHub CropAndStitch graph must edit, not re-draw.

The first billed run (task ``2108222523123662850``) came back ``SUCCESS`` with a
byte-exact ``904x1190`` frame and a clean background, and still had to be judged
a visual failure: the model had painted a *second face* -- a magnified copy of
the original, glasses and smile included -- over the photo. The cause was in the
graph, not in the model:

* both passes ran ``VAEEncodeForInpaint``, which encodes the masked region as
  flat 0.5 grey and therefore hands the sampler a blank slate inside the mask;
* both passes ran ``denoise=1.0``, which throws away even that much of the
  original crop;
* the positive prompts carried the plan's "here is who the person currently is"
  preamble plus "keep the facial features exactly" boilerplate, which is a
  *description of a person* attached to a crop that already contains a full head.

Every test below pins one of the fixes. They are deliberate contract tests: this
graph is what a billed task is built from, so the assumptions it rests on are
asserted mechanically rather than trusted. Nothing here touches the network.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tools import runninghub_crop_stitch_run as rh

#: The two plan clauses, copied verbatim from ``data/jobs/job_45f7920ad505.json``
#: (``plan.plan_id == "ai_elegant_refined"``). They are inlined rather than read
#: from ``artifacts/`` on purpose: the real package is gitignored, and the graph
#: must be verifiable from the repository alone.
CLOTHING_PROMPT = (
    "the person currently wears an off-white (米白) textured knit sweater. "
    "Replace it with a 深橄榄绿法式复古衬衫 (deep olive-green French vintage blouse): "
    "soft chiffon/silk with delicate detailing, worn with a slim metal necklace. "
    "Effortless Parisian chic. replace the current round thin metal-frame glasses with "
    "半框眉线眼镜 (browline / clubmaster eyeglasses): a thick black acetate top rim across "
    "the brow, slim silver wire across the lower half, clear lenses, sitting naturally on the face."
)
HAIR_PROMPT = (
    "the person currently has long straight hair falling past the shoulders, in deep black / "
    "dark brown (it reads as a very dark brown under warm light). Change it into "
    "侧分蓬松大波浪 (Side-Part Long Waves): a deep side parting on the left, voluminous softly "
    "layered wavy hair with natural movement and shine, cascading past the chest. Vintage "
    "Hollywood-glamour quality but kept natural, mature and elegant. Also change the hair colour "
    "from the current deep black / dark brown to 栗子棕 (Chestnut Brown): medium brown with rich "
    "warm reddish-brown undertones, like the inside of a roasted chestnut, healthy natural look."
)

MASK_RULE = ("Edit only inside the supplied mask; every pixel outside it must be the "
             "original photo's own pixels.")

#: The artefacts the failed run actually produced, with the glasses and the mouth
#: it invented a second time.
FAILED_RUN_ARTEFACTS = (
    "duplicate face", "extra eyes", "extra mouth", "extra glasses",
    "double head", "second person", "bald patches", "deformed hairline",
)


def _bundle(**overrides) -> dict:
    bundle = {
        "style_id": rh.EXPECTED_STYLE_ID,
        "style_name": "精致优雅风",
        "hair_prompt": HAIR_PROMPT,
        "clothing_prompt": CLOTHING_PROMPT,
        "identity_constraints": {"rules": [{"id": rh.MASK_RULE_ID, "en": MASK_RULE}]},
    }
    bundle.update(overrides)
    return bundle


def _filenames() -> dict:
    return {"original": "orig.jpg", "clothing_mask": "c.png", "hair_mask": "h.png"}


def _workflow() -> dict:
    return rh.build_workflow(Path("."), _filenames(), rh.build_prompts(_bundle()), seed=7)


def _synthetic_package(root: Path) -> Path:
    """A tiny stand-in for a real package: same layout, no photo of anybody."""
    width, height = 256, 320
    package = root / "package"
    package.mkdir()

    frame = Image.new("RGB", (width, height), (120, 120, 120))

    def mask(what: str) -> Image.Image:
        canvas = np.zeros((height, width), dtype=np.uint8)
        if what == "hair":
            canvas[20:90, 40:210] = 255
        elif what == "clothing":
            canvas[230:300, 30:220] = 255
        else:
            canvas[120:200, 100:156] = 255
        return Image.fromarray(canvas, mode="L")

    frame.save(package / "original.jpg", quality=90)
    mask("hair").save(package / "hair_edit_mask.png")
    mask("clothing").save(package / "clothing_edit_mask.png")
    mask("protected").save(package / "protected_identity_mask.png")
    (package / "manifest.json").write_text(json.dumps(
        {"source_image": {"width": width, "height": height}}, ensure_ascii=False))
    (package / "prompt_bundle.json").write_text(json.dumps(_bundle(), ensure_ascii=False))
    return package


# ---------------------------------------------------------------------------
# Prompts: the plan's own target wording, and nothing about the person.
# ---------------------------------------------------------------------------


def test_prompts_are_the_plan_target_half_plus_the_mask_rule():
    prompts = rh.build_prompts(_bundle())

    assert "深橄榄绿法式复古衬衫" in prompts["clothing"]
    assert "Effortless Parisian chic" in prompts["clothing"]
    assert "栗子棕" in prompts["hair"]
    assert prompts["mask_rule"] == MASK_RULE
    assert prompts["clothing"].endswith(MASK_RULE)
    assert prompts["hair"].endswith(MASK_RULE)


def test_positive_prompts_drop_the_current_state_preamble():
    """'The person currently ...' is a caption of the input, not an instruction."""
    prompts = rh.build_prompts(_bundle())

    assert "currently wears" not in prompts["clothing"]
    assert "currently has" not in prompts["hair"]
    assert prompts["current_hair_clause_dropped"] is True


def test_accessory_clauses_that_the_masks_do_not_cover_are_dropped():
    """The plan asks for new glasses and a necklace; neither is inside a mask.

    The glasses clause is the trap: its descriptors spell out ``brow`` and
    ``face``, so dropping the lead marker alone left facial-feature words in a
    prompt that must not describe facial features.
    """
    prompts = rh.build_prompts(_bundle())

    assert prompts["glasses_clause_dropped"] is True
    assert prompts["necklace_clause_dropped"] is True
    assert "glasses" not in prompts["clothing"].lower()
    assert "necklace" not in prompts["clothing"].lower()
    assert "browline" not in prompts["clothing"].lower()


def test_positive_prompts_never_name_a_facial_feature_or_a_person():
    prompts = rh.build_prompts(_bundle())

    for kind in ("clothing", "hair"):
        hit = rh.FORBIDDEN_POSITIVE_PATTERN.search(prompts[kind])
        assert hit is None, f"{kind} prompt contains {hit.group(0)!r}"


def test_negative_prompt_excludes_every_artefact_the_failed_run_produced():
    negative = rh.build_prompts(_bundle())["negative"]

    for term in FAILED_RUN_ARTEFACTS:
        assert term in negative
    for term in rh.REQUIRED_NEGATIVE_TERMS:
        assert term in negative


def test_clothing_prompt_pins_a_closed_collar_and_rules_out_a_deep_v():
    """Both runs opened the neckline; "shirt" was not enough of an instruction."""
    prompts = rh.build_prompts(_bundle())

    assert rh.COLLAR_CLAUSE in prompts["clothing"]
    assert "collar" in prompts["clothing"].lower()
    assert "collarbone" in prompts["clothing"].lower()
    for term in rh.COLLAR_NEGATIVE_TERMS:
        assert term in prompts["negative"]
    assert "deep v neck" in prompts["negative"]
    assert "exposed chest" in prompts["negative"]
    assert "sheer mesh shoulder" in prompts["negative"]
    assert "statement necklace" in prompts["negative"]


def test_the_collar_clause_pins_the_collar_to_the_blouse_colour():
    """The last run closed the neckline but kept ~10.5k px of old off-white knit.

    A collar that is only "higher and closed" still lets the sampler leave the
    fabric already at the neck, so the clause now names the colour and the
    negative terms rule out a second, lighter layer.
    """
    prompts = rh.build_prompts(_bundle())

    assert "olive-green" in prompts["clothing"]
    assert "lining" in prompts["clothing"]
    assert "scarf" in prompts["negative"]
    assert "undershirt" in prompts["negative"]
    assert "beige collar lining" in prompts["negative"]


def test_the_collar_negative_terms_name_no_person_and_no_facial_feature():
    """The negative list is clipped too, so it must stay free of anatomy words."""
    for term in rh.COLLAR_NEGATIVE_TERMS:
        assert rh.FORBIDDEN_POSITIVE_PATTERN.search(term) is None


def test_the_collar_clause_does_not_smuggle_a_facial_feature_back_in():
    prompts = rh.build_prompts(_bundle())

    assert rh.FORBIDDEN_POSITIVE_PATTERN.search(prompts["clothing"]) is None


def test_build_prompts_refuses_a_bundle_from_another_style():
    """``artistic_wave`` is the neighbouring axis and must not slip in as the plan."""
    with pytest.raises(SystemExit):
        rh.build_prompts(_bundle(style_id="artistic_wave"))


def test_build_prompts_refuses_a_bundle_without_the_mask_rule():
    with pytest.raises(SystemExit):
        rh.build_prompts(_bundle(identity_constraints={"rules": []}))


# ---------------------------------------------------------------------------
# Graph: the crop's own pixels have to reach the sampler.
# ---------------------------------------------------------------------------


def test_graph_contains_no_text_to_image_latent():
    workflow = _workflow()

    for node_id, node in workflow.items():
        assert node["class_type"] not in {"EmptyLatentImage", "EmptySD3LatentImage"}, node_id
        assert node["class_type"] != "VAEEncodeForInpaint", node_id


@pytest.mark.parametrize("kind, sampler_id, cond_id, crop_id, clip_ids", [
    ("clothing", "8", "7", "4", ("5", "6")),
    ("hair", "16", "15", "12", ("13", "14")),
])
def test_each_pass_is_conditioned_on_its_own_crop(kind, sampler_id, cond_id, crop_id, clip_ids):
    workflow = _workflow()
    cond, sampler = workflow[cond_id], workflow[sampler_id]

    assert cond["class_type"] == "InpaintModelConditioning"
    assert cond["inputs"]["pixels"] == [crop_id, 1], "must be the crop image, not the whole photo"
    assert cond["inputs"]["mask"] == [crop_id, 2]
    assert cond["inputs"]["positive"] == [clip_ids[0], 0]
    assert cond["inputs"]["negative"] == [clip_ids[1], 0]
    assert cond["inputs"]["vae"] == ["3", 2]
    assert cond["inputs"]["noise_mask"] is True

    assert sampler["inputs"]["latent_image"] == [cond_id, 2]
    assert sampler["inputs"]["positive"] == [cond_id, 0]
    assert sampler["inputs"]["negative"] == [cond_id, 1]
    assert sampler["inputs"]["model"] == ["3", 0]


def test_denoise_is_low_enough_to_keep_the_original_crop():
    workflow = _workflow()

    for kind, sampler_id in (("clothing", "8"), ("hair", "16")):
        denoise = workflow[sampler_id]["inputs"]["denoise"]
        assert denoise == rh.PASS_DENOISE[kind]
        assert denoise < 1.0, "denoise=1.0 discards the crop and draws a new person"
        assert rh.DENOISE_BAND[0] <= denoise <= rh.DENOISE_BAND[1]


def test_hair_pass_consumes_the_clothing_pass_stitch():
    """Order is clothing -> hair: the hair mask overlaps the clothing mask and
    the later pass owns the overlap, so the fringe lands on top of the collar."""
    workflow = _workflow()

    assert workflow["4"]["inputs"]["image"] == ["1", 0]
    assert workflow["10"]["inputs"]["stitcher"] == ["4", 0]
    assert workflow["12"]["inputs"]["image"] == ["10", 0]
    assert workflow["18"]["inputs"]["stitcher"] == ["12", 0]
    assert workflow["19"]["inputs"]["images"] == ["18", 0]


def test_each_pass_leaves_a_stage_record_without_extra_sampling():
    """A pale band under the chin has to be attributable to one pass afterwards.

    Node 20 saves the clothing pass's own stitch; node 19 saves the finished
    frame. Both read nodes that already exist, so the graph still samples exactly
    twice -- a diagnostic must never add a second generation.
    """
    workflow = _workflow()

    assert workflow["20"]["class_type"] == "SaveImage"
    assert workflow["20"]["inputs"]["images"] == ["10", 0]
    assert workflow["19"]["inputs"]["images"] == ["18", 0]

    saves = [nid for nid, node in workflow.items() if node["class_type"] == "SaveImage"]
    assert sorted(saves) == ["19", "20"]
    samplers = [nid for nid, node in workflow.items() if node["class_type"] == "KSampler"]
    assert sorted(samplers) == ["16", "8"]
    decodes = [nid for nid, node in workflow.items() if node["class_type"] == "VAEDecode"]
    assert sorted(decodes) == ["17", "9"]


def test_workflow_diff_reports_the_conditioning_swap():
    """The report has to show the fix, not just the finished graph."""
    before = {"7": {"class_type": "VAEEncodeForInpaint",
                    "inputs": {"pixels": ["4", 1], "vae": ["3", 2], "mask": ["4", 2]}},
              "8": {"class_type": "KSampler", "inputs": {"latent_image": ["7", 0], "denoise": 1.0}}}
    after = {"7": {"class_type": "InpaintModelConditioning",
                   "inputs": {"pixels": ["4", 1], "vae": ["3", 2], "mask": ["4", 2], "noise_mask": True}},
             "8": {"class_type": "KSampler", "inputs": {"latent_image": ["7", 2], "denoise": 0.75}}}

    diff = rh._workflow_diff(before, after)
    assert "~ node 7 VAEEncodeForInpaint -> InpaintModelConditioning" in diff
    assert any("denoise: 1.0 -> 0.75" in line for line in diff)


# ---------------------------------------------------------------------------
# The local gate a billed run has to clear.
# ---------------------------------------------------------------------------


def test_gate_passes_on_a_package_without_any_api_key(tmp_path):
    package = _synthetic_package(tmp_path)
    prompts = rh.build_prompts(_bundle())
    workflow = rh.build_workflow(package, _filenames(), prompts, seed=7)

    errors, report = rh.collect_gate_checks(workflow, None, prompts, _bundle(), package)

    assert errors == []
    assert report["style_id_matches_plan"] is True
    assert report["prompt_has_plan_phrases"] is True
    assert report["negative_prompt_terms_ok"] is True
    assert report["collar_clause_ok"] is True
    assert report["collar_negative_terms_ok"] is True
    assert report["stage_records"]["clothing_pass"] == ["20"]
    assert report["stage_records"]["final"] == ["19"]
    assert all(size == (256, 320) for size in report["image_sizes"].values())


def test_gate_rejects_a_mask_that_reaches_the_facial_core(tmp_path):
    """A painted mask over the face is the one failure a billed run must never
    discover after the fact."""
    package = _synthetic_package(tmp_path)
    canvas = np.zeros((320, 256), dtype=np.uint8)
    canvas[20:250, 40:210] = 255  # hair painted straight through the protected face
    Image.fromarray(canvas, mode="L").save(package / "hair_edit_mask.png")

    prompts = rh.build_prompts(_bundle())
    workflow = rh.build_workflow(package, _filenames(), prompts, seed=7)
    errors, report = rh.collect_gate_checks(workflow, None, prompts, _bundle(), package)

    assert report["mask_geometry"]["hair"]["painted_over_facial_core"] > 0
    assert any("facial core" in err for err in errors)


def test_gate_rejects_a_plain_text_to_image_graph(tmp_path):
    package = _synthetic_package(tmp_path)
    prompts = rh.build_prompts(_bundle())
    workflow = rh.build_workflow(package, _filenames(), prompts, seed=7)
    workflow["7"] = {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}}

    errors, _ = rh.collect_gate_checks(workflow, None, prompts, _bundle(), package)

    assert any("EmptyLatentImage" in err for err in errors)


def test_cli_dry_run_writes_the_graph_and_the_gate_report(tmp_path):
    package = _synthetic_package(tmp_path)
    out = tmp_path / "out"

    code = rh.main([
        "--package", str(package),
        "--out", str(out),
        "--object-info", str(tmp_path / "absent.json"),
        "--compare-to", str(tmp_path / "absent-workflow.json"),
        "--dry-run",
    ])

    assert code == 0
    workflow = json.loads((out / "workflow.json").read_text())
    assert workflow["7"]["class_type"] == "InpaintModelConditioning"
    assert workflow["15"]["class_type"] == "InpaintModelConditioning"
    report = json.loads((out / "gate_report.json").read_text())
    assert report["errors"] == []
