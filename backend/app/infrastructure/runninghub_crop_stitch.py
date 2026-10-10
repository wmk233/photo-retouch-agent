"""The validated RunningHub CropAndStitch edit, shared by the CLI and the app.

This module is the single implementation of the two-pass masked edit that
produced the best accepted result (RunningHub task ``2108330225116082177``,
plan ``ai_elegant_refined``). ``tools/runninghub_crop_stitch_run.py`` is a thin
CLI over it; ``app/providers/runninghub_crop_stitch_provider.py`` drives it from
the web app.

The graph runs on RunningHub with the ``lquesada/ComfyUI-Inpaint-CropAndStitch``
nodes::

    pass 1  clothing_edit_mask  on the original image
    pass 2  hair_edit_mask      on pass 1's stitched output

Pass order is not cosmetic. CropAndStitch composites as
``new = mask * inpainted + (1 - mask) * canvas`` and each pass blends over the
image the previous pass returned, so the mask that runs last owns the
hair/clothing overlap band. Hair runs last on purpose: strands that fall over
the new garment stay hair.

Each pass is conditioned on the *original crop pixels* rather than on an empty
latent: ``InpaintCropImproved`` feeds ``cropped_image`` / ``cropped_mask`` into
``InpaintModelConditioning`` and the sampler takes its positive, negative and
latent from that node, with ``denoise`` below 1.0. The first real run used
``VAEEncodeForInpaint`` (which encodes a flat grey blob inside the mask) at
``denoise=1.0``, so the model had no original signal to preserve and painted a
second face and a second pair of glasses into the hair mask -- see
``evidence/p29-fix-run/``.

Everything is validated locally before anything is sent, because RunningHub
creates a task (and then fails it) even when the workflow references a node the
runtime does not have -- see ``evidence/p29-runninghub-live-audit.md``.

Generalisation note: the style is never hard-coded here. ``build_prompts``
checks the bundle against ``expected_style_id``, which the caller sets to the
plan the user actually selected. The one plan this round was tuned against stays
available as :data:`TUNED_STYLE_ID` for the CLI's own default.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Callable

import httpx


class RunningHubError(RuntimeError):
    """A RunningHub call failed; the caller decides whether the job fails."""


BASE_URL = "https://www.runninghub.cn"
CREATE_PATH = "/task/openapi/create"
STATUS_PATH = "/task/openapi/status"
OUTPUTS_PATH = "/task/openapi/outputs"
# The v2 binary upload endpoint rejects this account's key with errorCode 1000;
# the classic task-scoped upload endpoint authenticates via the apiKey form
# field and returns the server-side ``fileName`` that LoadImage consumes.
UPLOAD_PATH = "/task/openapi/upload"

CHECKPOINT = "juggernautXL_juggXIByRundiffusion.safetensors"

# The style this round must realise. The masks, the identity constraints and the
# acceptance list were all built for this one plan; driving the run from any other
# style (the previously wired ``artistic_wave`` approximation in particular) would
# silently change what the picture is supposed to be.
EXPECTED_STYLE_ID = "ai_elegant_refined"

#: The plan every parameter in this module was tuned against (the one whose
#: RunningHub task ``2108330225116082177`` was accepted). It is only a default
#: for the CLI; the app passes the plan the user actually selected.
TUNED_STYLE_ID = EXPECTED_STYLE_ID

# The runtime has no dedicated inpainting checkpoint. Every ``CheckpointLoaderSimple``
# option is a plain SD1.5/SDXL base or a Flux UNet, and the only inpaint weights on
# the platform are ControlNet / LLLite patches that plug into a Qwen-Image or Flux
# graph -- splicing one of those into this SDXL CropAndStitch graph is exactly the
# incompatible-node assembly the brief forbids, so it is not attempted. What *is*
# supported for a base SDXL model is the classic masked-latent path: VAE-encode the
# crop, confine sampling with a ``noise_mask``, and stay below denoise 1.0 so the
# existing pixels still drive the sampler. ``validate_checkpoint`` states this
# verdict explicitly instead of pretending the concat-inpaint path is available.

# Mask geometry decisions (see evidence/p29-fix-run/mask_pipeline_probe.py):
#   * mask_fill_holes=False -- the node's iterative fill *added* 282 px of
#     protected chin/collar skin to the clothing mask and 2 px to the hair mask,
#     i.e. it repainted skin we promised to keep. The 283 px of speckle holes it
#     was closing are far below one latent pixel (a 1024 px crop is a 128x128
#     latent) and they fail *safe*: a hole keeps the original pixel.
#   * mask_blend_pixels=2 -- expand(2)+blur(1) reaches ~4 px past the mask edge
#     rather than ~8 px, and that ring is the only place a generated pixel can
#     touch the protected identity contour. Not 0, because a hard edge would cut
#     the hair/skin boundary the stitch has to feather.
#   * mask_expand_pixels=0 -- the SCHP masks already cover the target silhouette.
#   * context_from_mask_extend_factor=1.2 -- widens the crop the model sees, not
#     the mask it is allowed to paint.
CROP_DEFAULTS: dict[str, object] = {
    "downscale_algorithm": "bilinear",
    "upscale_algorithm": "bicubic",
    "preresize": False,
    "preresize_mode": "ensure minimum resolution",
    "preresize_min_width": 1024,
    "preresize_min_height": 1024,
    "preresize_max_width": 16384,
    "preresize_max_height": 16384,
    "mask_fill_holes": False,
    "mask_expand_pixels": 0,
    "mask_invert": False,
    "mask_blend_pixels": 2,
    "mask_hipass_filter": 0.1,
    "extend_for_outpainting": False,
    "extend_up_factor": 1.0,
    "extend_down_factor": 1.0,
    "extend_left_factor": 1.0,
    "extend_right_factor": 1.0,
    "context_from_mask_extend_factor": 1.2,
    "output_resize_to_target_size": True,
    "output_target_width": 1024,
    "output_target_height": 1024,
    "output_padding": "32",
    "device_mode": "gpu (much faster)",
}

SAMPLER = {
    "steps": 30,
    "cfg": 6.5,
    "sampler_name": "dpmpp_2m_sde",
    "scheduler": "karras",
}

# Denoise is per pass. With ``noise_mask`` set, the sampler starts the masked
# region from ``encode(original_crop) + noise * sigma(denoise)`` and blends the
# unmasked region back to the clean latent after every step, so denoise is the
# img2img strength *inside the mask only*. Because the runtime has no inpainting
# checkpoint, this strength is the only handle on "keep the person, change the
# garment": denoise=1.0 discards the original signal and was the direct cause of
# the duplicated head. The band is the range in which the model still has to
# follow the silhouette that is already there.
PASS_DENOISE = {"clothing": 0.75, "hair": 0.70}
DENOISE_BAND = (0.60, 0.80)

# The plan's Accessories clause asks for new browline glasses, but this round's
# acceptance list pins "glasses unchanged" and the glasses sit inside the
# protected identity mask, so the clause is dropped from the prompt instead of
# contradicting the identity constraints in the same conditioning.
GLASSES_CLAUSE = "replace the current round thin metal-frame glasses"

# The garment clause ends with a new necklace, which is an accessory the masks do
# not cover, so it is dropped the same way.
NECKLACE_CLAUSE = "worn with a slim metal necklace"

# The plan names a *shirt*, and a shirt alone left the sampler free to open the
# neckline: both billed runs came back with a deep V and an exposed chest, which
# the plan's own silhouette explicitly forbids. The neckline is therefore pinned
# on the positive side as well -- a higher natural collar closed near the
# collarbone, with the vintage folds the plan asks for -- so "follow the existing
# garment" does not read as "open the front".
# The second billed run closed the neckline but did not recolour it: both the
# Clothing Pass and the FINAL kept ~10.5k px of the *old* off-white knit at
# the neck -- a pale vertical strip and a pale patch under the jaw, present in
# the Clothing Pass itself and therefore not a hair-pass artefact. The clause is
# now pinned on the colour side as well, so "a higher collar" cannot be read as
# "keep whatever light fabric already sits there".
COLLAR_CLAUSE = (
    "a higher natural collar closed near the collarbone, cut from the same deep "
    "olive-green cloth as the blouse, its inner lining and its folds all in that "
    "same deep olive-green tone, with fine pleats and soft vintage folds"
)

# The neckline artefacts the plan rules out, carried as negative terms so the
# sampler is pushed away from an open front even where the mask leaves the
# neckline editable. These are hints only: the mask and the original-crop
# conditioning are what actually hold the silhouette.
COLLAR_NEGATIVE_TERMS = (
    "deep v neck",
    "plunging neckline",
    "exposed chest",
    "bare cleavage",
    "sheer mesh shoulder",
    "statement necklace",
    # The garment being replaced is an off-white textured knit, so every term
    # above still left the sampler free to keep a light layer at the neck. These
    # rule out a second, lighter layer -- a white/beige collar lining, a scarf or
    # an inner top -- which is what the last run printed as the pale residual.
    "white collar",
    "cream collar",
    "beige collar lining",
    "white inner layer",
    "pale knit at the neckline",
    "white scarf",
    "beige knitted scarf",
    "white undershirt",
    "ivory lining",
)

# ``hair_prompt`` names the hair being replaced. At denoise below 1.0 a
# description of the input biases the model toward keeping it, so only the
# target colour is kept.
CURRENT_HAIR_CLAUSE = "from the current deep black / dark brown"

# The plan clause is split at the verb that introduces the target, so the positive
# prompt is still the plan's own wording -- just without the "here is what the
# person currently looks like" preamble that made the model redraw the person.
CLOTHING_TARGET_MARKER = "Replace it with"
HAIR_TARGET_MARKER = "Change it into"
MARKER_BY_KIND = {"clothing": CLOTHING_TARGET_MARKER, "hair": HAIR_TARGET_MARKER}
CLOTHING_TARGET_SUFFIX = (
    "draped naturally over the body, following the original pose and body shape, "
    "with natural cloth folds"
)
HAIR_TARGET_SUFFIX = (
    "real individual hair strands, natural hairline, growing naturally from the "
    "existing scalp"
)

# Positive prompts may describe only what is inside the mask. Spelled-out facial
# features are the one thing a masked edit must never ask for: the crop already
# contains a whole head as context, and "keep the facial features exactly" reads
# as an instruction to draw them. Matched on word boundaries so "chestnut brown"
# is not mistaken for "brow".
FORBIDDEN_POSITIVE_PATTERN = re.compile(
    r"\b(facial feature|facial features|eye|eyes|eyebrow|eyebrows|brow|brows|"
    r"nose|lips|lip|mouth|teeth|smile|face|glasses|person|portrait|identity)\b",
    re.IGNORECASE,
)

# English rendering of prompt_bundle.negative_prompt (which is Chinese; CLIP for
# SDXL is English-trained). The first eight terms are the artefacts the previous
# run actually produced -- a second face with its own glasses and mouth, and
# patchy hair -- and the rest carry the identity / background / seam rules. The
# negative prompt is a hint only: the mask and the original-pixel conditioning
# are what actually hold the identity.
NEGATIVE_PROMPT = (
    "duplicate face, extra eyes, extra mouth, extra glasses, double head, "
    "second person, bald patches, deformed hairline, "
    "plastic airbrushed skin, blurred face, reshaped or replaced facial features, "
    "a different person's face, obvious age drift, changing the number of people, "
    "a broken or damaged background, extra fingers or limbs, text, watermark, "
    "visible seam, hard cut-out edge"
)
REQUIRED_NEGATIVE_TERMS = (
    "duplicate face",
    "extra eyes",
    "extra mouth",
    "extra glasses",
    "double head",
    "second person",
    "bald patches",
    "deformed hairline",
)

# The plan writes each clause as "here is the current state ... <verb> <target>".
# At denoise below 1.0 the "current state" half is a description of the input,
# which biases the sampler toward keeping it -- and, worse, it is a description
# of the person, which is exactly what made the model draw one. Only the half
# after the verb is used, re-introduced with a neutral sentence frame.
TARGET_INTRO = {"clothing": "Replace the garment with ", "hair": "Replace the hairstyle with "}

# The plan's own identity rule, kept verbatim as the single mask instruction. The
# rest of ``identity_constraints.prompt_text_en`` describes facial features and is
# deliberately NOT sent: with a whole head inside the crop, "keep the facial
# features exactly" reads as an instruction to draw them.
MASK_RULE_ID = "edit_only_inside_masks"

#: Fallback when a plan carries no identity rules of its own. Identical in
#: meaning to the rule the plan generator writes, so the two paths cannot drift.
DEFAULT_MASK_RULE = (
    "Edit only inside the supplied mask; every pixel outside it must be the "
    "original photo's own pixels."
)

# The mask does not cover glasses or a necklace, so any accessory clause has to
# leave the prompt -- otherwise the conditioning asks for a repaint the mask
# will never deliver. The tuned plan's clauses are matched verbatim above; these
# patterns catch the same clause in a plan written in different words.
GENERIC_GLASSES_CLAUSE = re.compile(
    r"\s*replace the current [^.]*?\bglasses\b[^.]*?\.", re.IGNORECASE
)
GENERIC_NECKLACE_CLAUSE = re.compile(
    r"\s*,?\s*(?:worn (?:with|over)|paired with|accessorised with|accessorized with)"
    r"[^.]*?\bnecklace\b[^.]*\.?",
    re.IGNORECASE,
)

# The current plan generator writes each clause on its own line, with the target
# introduced by a lowercase verb. These patterns lift the target half out of
# that wording; the tool's capitalised markers stay supported for the plan text
# the tuned run was built from.
PLAN_CLAUSE_PATTERNS = {
    "hair": (
        re.compile(r"\bchange the hair from\b.*?\bto\b\s*(.+)$", re.IGNORECASE | re.DOTALL),
        re.compile(r"\bchange it into\b\s*(.+)$", re.IGNORECASE | re.DOTALL),
    ),
    "clothing": (
        re.compile(r"\breplace the current\b.*?\bwith\b\s*(.+)$", re.IGNORECASE | re.DOTALL),
        re.compile(r"\breplace it with\b\s*(.+)$", re.IGNORECASE | re.DOTALL),
    ),
}

#: The plan's own line labels. The composition tool writes each clause on its own
#: line, and the AI stylist labels the garment clause ``Outfit:`` while the
#: template axes label it ``Clothing:``. Both spellings are accepted so a plan is
#: never rejected just for its label -- the *target text* is what is checked.
PLAN_CLAUSE_LABELS = {
    "hair": ("hair", "hairstyle"),
    "clothing": ("clothing", "outfit"),
}

#: A leading clause that only describes the input ("the person currently wears
#: an off-white knit sweater.") biases the sampler toward keeping it.
CURRENT_STATE_PATTERN = re.compile(
    r"\b(?:the person )?currently (?:wears|has|is wearing)\b[^.]*\.\s*", re.IGNORECASE
)

#: The hair line carries two targets (style and colour). The colour sentence is
#: part of the target, the "current colour" half is not.
CURRENT_HAIRCOLOR_PATTERN = re.compile(
    r"\bfrom the current\b[^,.]*?(?=,\s*to\b|\s+to\b)", re.IGNORECASE
)


#: A run of CJK characters -- how a plan names its target garments and colours.
CJK_TOKEN = re.compile(r"[\u3400-\u9fff][\u3400-\u9fff\w]*")


#: A labelled clause begins at ``Label:``; anything without a label is either a
#: clause the rung does not use (``Makeup:``) or a hard-wrapped continuation of
#: the clause above it. The AI stylist wraps long clauses mid-sentence, so lines
#: are joined back before the target is lifted -- otherwise the target is cut in
#: half at the first newline.
CLAUSE_LABEL_PATTERN = re.compile(r"^([A-Za-z][A-Za-z /_-]*?)\s*[:\uFF1A]\s*(.*)$")


def plan_clauses(edit_prompt: str) -> dict[str, str]:
    """Group a plan's hard-wrapped prompt lines back into their labelled clauses."""
    clauses: dict[str, list[str]] = {}
    current: str | None = None
    for raw in str(edit_prompt or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        match = CLAUSE_LABEL_PATTERN.match(line)
        if match:
            current = match.group(1).strip().lower()
            clauses.setdefault(current, []).append(match.group(2))
        elif current is not None:
            clauses[current].append(line)
    return {key: _tidy(" ".join(parts)) for key, parts in clauses.items()}


def extract_plan_targets(edit_prompt: str) -> dict:
    """Lift the clothing and hair target clauses out of a plan's own prompt.

    Returns both the target text and whether it was found, so a caller can
    fall back to the plan's structured fields instead of sending a prompt that
    only pretends to be about the selected look.
    """
    targets: dict = {}
    clauses = plan_clauses(edit_prompt)
    for kind, patterns in PLAN_CLAUSE_PATTERNS.items():
        found = ""
        for label in PLAN_CLAUSE_LABELS[kind]:
            clause = clauses.get(label)
            if not clause:
                continue
            for pattern in patterns:
                match = pattern.search(clause)
                if match:
                    found = _tidy(match.group(1))
                    break
            if found:
                break
        targets[f"{kind}_target"] = found
        targets[f"{kind}_found"] = bool(found)
    return targets


def _strip_current_state(text: str) -> str:
    """Remove the "here is what the person looks like now" preamble."""
    text = CURRENT_STATE_PATTERN.sub("", text)
    text = CURRENT_HAIRCOLOR_PATTERN.sub("from ", text)
    return _tidy(text)


def _drop_generic_accessories(text: str) -> tuple[str, bool]:
    dropped = False
    for pattern in (GENERIC_GLASSES_CLAUSE, GENERIC_NECKLACE_CLAUSE):
        replaced, changed = pattern.subn(" ", text)
        if changed:
            dropped = True
            text = replaced
    return _tidy(text), dropped


#: Sentence boundary used to lift descriptive prose back out of a plan clause.
#: The AI stylist writes the target as flowing English ("...air bangs falling
#: across the forehead just above the eyebrows ... to softly frame the face"),
#: and a masked edit must never ask the sampler to draw a facial feature. The
#: whole offending sentence is removed so the surviving prose stays grammatical;
#: ``build_prompts`` still re-checks the assembled prompt afterwards, so this is
#: a cleaner, not a relaxation, of the guard.
FORBIDDEN_SENTENCE_SPLIT = re.compile(r"(?<=[.;!?\u3002\uff01\uff1f])\s+")


def _drop_forbidden_clauses(text: str) -> tuple[str, list[str]]:
    """Drop whole sentences that name a facial feature or a person."""
    kept: list[str] = []
    dropped: list[str] = []
    for sentence in FORBIDDEN_SENTENCE_SPLIT.split(text):
        sentence = sentence.strip()
        if not sentence:
            continue
        if FORBIDDEN_POSITIVE_PATTERN.search(sentence):
            dropped.append(sentence)
        else:
            kept.append(sentence)
    return _tidy(" ".join(kept)), dropped


def _plan_mask_rule(bundle: dict) -> str:
    for rule in bundle.get("identity_constraints", {}).get("rules", []):
        if rule.get("id") == MASK_RULE_ID and rule.get("en"):
            return rule["en"]
    raise SystemExit(f"prompt bundle has no {MASK_RULE_ID!r} identity rule")


def _tidy(text: str) -> str:
    """Collapse whitespace and remove the space the clause removal left before punctuation."""
    return re.sub(r"\s+", " ", re.sub(r"\s+([.,])", r"\1", text)).strip()


def _join_head_tail(head: str, tail: str) -> str:
    if head.rstrip().endswith((",", ".", ";", ":")):
        head = head.rstrip()[:-1] + " "
    return _tidy(head + tail)


def _drop_clause(text: str, clause: str) -> tuple[str, bool]:
    """Remove ``clause`` and the punctuation/space that glued it to the sentence."""
    idx = text.find(clause)
    if idx < 0:
        return text, False
    head, tail = text[:idx], text[idx + len(clause):]
    return _join_head_tail(head, tail), True


def _drop_clause_through(text: str, clause: str) -> tuple[str, bool]:
    """Remove ``clause`` *and the rest of the sentence it opens*.

    Used for plan clauses whose tail is a comma-separated descriptor list. The
    plan's glasses clause is ``replace the current round thin metal-frame glasses
    with 半框眉线眼镜 (browline / clubmaster eyeglasses): ... sitting naturally on
    the face.`` -- dropping only the lead marker would leave ``brow`` and
    ``face`` in a positive prompt whose whole point is that facial features are
    not to be described.
    """
    idx = text.find(clause)
    if idx < 0:
        return text, False
    tail = text[idx + len(clause):]
    end = re.search(r"\.(?=\s|$)", tail)
    tail = tail[end.end():] if end else ""
    return _join_head_tail(text[:idx], tail), True


def _target_half(text: str, marker: str, kind: str) -> str:
    """Keep only the clause after ``marker`` and give it a neutral sentence frame."""
    idx = text.find(marker)
    if idx < 0:
        raise SystemExit(f"{kind}: plan prompt has no {marker!r} marker")
    rest = text[idx + len(marker):].strip()
    rest = re.sub(r"\s+", " ", rest)
    frame = TARGET_INTRO[kind]
    return frame + rest[0].lower() + rest[1:] if rest else frame.strip()


def _read_key(key_file: Path) -> str:
    key = key_file.read_text().strip()
    if len(key) < 16:
        raise SystemExit(f"credential file {key_file} does not look like an API key")
    return key


def _plan_target(bundle: dict, kind: str) -> str:
    """The target half of one plan clause, from the most trustworthy source.

    Preference order:

    1. an explicitly extracted ``{kind}_target`` (the app lifts it from the real
       plan's prompt, so nothing is guessed from wording);
    2. the capitalised marker the tuned plan uses;
    3. the plan generator's own lowercase phrasing.
    """
    explicit = str(bundle.get(f"{kind}_target") or "").strip()
    if explicit:
        return _strip_current_state(explicit)

    source = str(bundle.get(f"{kind}_prompt") or "").strip()
    if not source:
        raise SystemExit(f"prompt bundle has no {kind} clause to build a prompt from")
    marker = MARKER_BY_KIND[kind]
    if marker in source:
        return _target_half(source, marker, kind)
    extracted = extract_plan_targets(source)
    if extracted[f"{kind}_found"]:
        return _target_half(
            f"{MARKER_BY_KIND[kind]} {extracted[f'{kind}_target']}", MARKER_BY_KIND[kind], kind
        )
    raise SystemExit(f"{kind}: plan prompt has no target marker to extract")


def build_prompts(bundle: dict, *, expected_style_id: str | None = EXPECTED_STYLE_ID) -> dict:
    """Turn the stored plan into the two positive prompts and the shared negative.

    Everything is taken from the plan text; the only edits are the ones the mask
    scope demands (drop the accessory clauses whose target sits outside the mask)
    and the ones identity preservation demands (drop the "who the person is"
    preamble, which the first run's model answered by drawing a second person).

    ``expected_style_id`` is the caller's guarantee that the bundle belongs to
    the plan it claims. The app passes the *user's selected* plan id here, so a
    bundle can never be driven by a style the user did not choose. Pass ``None``
    only when the caller has already proven the pairing another way.
    """
    style_id = bundle.get("style_id")
    if expected_style_id is not None and style_id != expected_style_id:
        raise SystemExit(
            f"prompt bundle style_id is {style_id!r}, expected {expected_style_id!r}"
        )

    clothing = _plan_target(bundle, "clothing")
    clothing, necklace_dropped = _drop_clause(clothing, NECKLACE_CLAUSE)
    clothing, glasses_dropped = _drop_clause_through(clothing, GLASSES_CLAUSE)
    clothing, generic_dropped = _drop_generic_accessories(clothing)
    clothing, clothing_forbidden_dropped = _drop_forbidden_clauses(clothing)
    if not clothing:
        raise SystemExit(
            "clothing plan clause is all facial-feature prose; nothing left to prompt with"
        )
    clothing = (
        f"{clothing.rstrip(' .,;')}, {CLOTHING_TARGET_SUFFIX}, {COLLAR_CLAUSE}"
    )

    hair = _plan_target(bundle, "hair")
    hair, current_hair_dropped = _drop_clause(hair, CURRENT_HAIR_CLAUSE)
    if not current_hair_dropped:
        stripped = _strip_current_state(hair)
        current_hair_dropped = stripped != hair
        hair = stripped
    hair, hair_forbidden_dropped = _drop_forbidden_clauses(hair)
    if not hair:
        raise SystemExit(
            "hair plan clause is all facial-feature prose; nothing left to prompt with"
        )
    hair = f"{hair.rstrip(' .,;')}, {HAIR_TARGET_SUFFIX}"

    negative = f"{NEGATIVE_PROMPT}, {', '.join(COLLAR_NEGATIVE_TERMS)}"
    rule = _plan_mask_rule(bundle)
    prompts = {
        "style_id": style_id,
        "style_name": bundle.get("style_name"),
        "clothing": f"{clothing} {rule}",
        "hair": f"{hair} {rule}",
        "negative": negative,
        "glasses_clause_dropped": glasses_dropped,
        "necklace_clause_dropped": necklace_dropped,
        "accessory_clauses_dropped": bool(
            glasses_dropped or necklace_dropped or generic_dropped
        ),
        "current_hair_clause_dropped": current_hair_dropped,
        "clothing_forbidden_clauses_dropped": clothing_forbidden_dropped,
        "hair_forbidden_clauses_dropped": hair_forbidden_dropped,
        "collar_clause": COLLAR_CLAUSE,
        "mask_rule": rule,
    }

    for key in ("clothing", "hair"):
        hit = FORBIDDEN_POSITIVE_PATTERN.search(prompts[key])
        if hit:
            raise SystemExit(
                f"{key} positive prompt names a facial feature or a person: {hit.group(0)!r} "
                f"in {prompts[key]!r}"
            )
    missing = [term for term in REQUIRED_NEGATIVE_TERMS if term not in prompts["negative"]]
    if missing:
        raise SystemExit(f"negative prompt is missing {missing}")
    missing_collar = [term for term in COLLAR_NEGATIVE_TERMS if term not in prompts["negative"]]
    if missing_collar:
        raise SystemExit(f"negative prompt is missing the neckline terms {missing_collar}")
    if COLLAR_CLAUSE not in prompts["clothing"]:
        raise SystemExit("clothing prompt lost the closed-collar clause")
    for kind in ("clothing", "hair"):
        denoise = PASS_DENOISE[kind]
        if not DENOISE_BAND[0] <= denoise <= DENOISE_BAND[1]:
            raise SystemExit(f"{kind} denoise {denoise} outside {DENOISE_BAND}")
    return prompts


def build_workflow(
    package: Path,
    filenames: dict[str, str],
    prompts: dict,
    seed: int,
    *,
    filename_prefix: str = "p29_elegant",
    checkpoint: str = CHECKPOINT,
) -> dict:
    """The two-pass CropAndStitch graph, with the runtime's file names spliced in.

    ``package`` is kept for signature compatibility with the CLI; the graph only
    depends on the uploaded file names, the prompts and the seed.
    """
    hair_mask = filenames["hair_mask"]
    clothing_mask = filenames["clothing_mask"]
    original = filenames["original"]
    crop = CROP_DEFAULTS

    return {
        "1": {"class_type": "LoadImage", "inputs": {"image": original}},
        "2": {"class_type": "LoadImageMask", "inputs": {"image": clothing_mask, "channel": "red"}},
        "3": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": checkpoint},
        },
        "4": {
            "class_type": "InpaintCropImproved",
            "inputs": {"image": ["1", 0], "mask": ["2", 0], **crop},
        },
        "5": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": prompts["clothing"], "clip": ["3", 1]},
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": prompts["negative"], "clip": ["3", 1]},
        },
        # The crop's own pixels and the crop's own mask go into the conditioning.
        # ``InpaintModelConditioning`` returns a latent whose samples are the VAE
        # encoding of the *original* crop (not of a grey fill) plus a ``noise_mask``
        # built from the crop mask, and it is that latent -- reached through this
        # node's three outputs -- that the sampler must start from.
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
                "denoise": PASS_DENOISE["clothing"],
            },
        },
        "9": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["3", 2]}},
        "10": {
            "class_type": "InpaintStitchImproved",
            "inputs": {"stitcher": ["4", 0], "inpainted_image": ["9", 0]},
        },
        "11": {"class_type": "LoadImageMask", "inputs": {"image": hair_mask, "channel": "red"}},
        "12": {
            "class_type": "InpaintCropImproved",
            "inputs": {"image": ["10", 0], "mask": ["11", 0], **crop},
        },
        "13": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": prompts["hair"], "clip": ["3", 1]},
        },
        "14": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": prompts["negative"], "clip": ["3", 1]},
        },
        "15": {
            "class_type": "InpaintModelConditioning",
            "inputs": {
                "positive": ["13", 0],
                "negative": ["14", 0],
                "vae": ["3", 2],
                "pixels": ["12", 1],
                "mask": ["12", 2],
                "noise_mask": True,
            },
        },
        "16": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["3", 0],
                "seed": seed + 1,
                "positive": ["15", 0],
                "negative": ["15", 1],
                "latent_image": ["15", 2],
                **SAMPLER,
                "denoise": PASS_DENOISE["hair"],
            },
        },
        "17": {"class_type": "VAEDecode", "inputs": {"samples": ["16", 0], "vae": ["3", 2]}},
        "18": {
            "class_type": "InpaintStitchImproved",
            "inputs": {"stitcher": ["12", 0], "inpainted_image": ["17", 0]},
        },
        "19": {
            "class_type": "SaveImage",
            "inputs": {"images": ["18", 0], "filename_prefix": filename_prefix},
        },
        # Stage record for the two-pass replay. Node 19 is the finished frame;
        # this second save is the clothing pass's own stitch, captured before the
        # hair pass runs. Both hang off existing nodes, so the run still samples
        # exactly twice -- nothing here creates an extra task or a second
        # sampling pass, it only names which stage a pixel came from. Without it
        # a pale band under the chin cannot be attributed to one pass or the
        # other after the fact.
        "20": {
            "class_type": "SaveImage",
            "inputs": {
                "images": ["10", 0],
                "filename_prefix": f"{filename_prefix}_clothing_pass",
            },
        },
    }


def _split_field(field):
    """Normalise a ComfyUI INPUT_TYPES entry into (kind, opts).

    ``field`` is either a bare type name, ``[type_name, opts]``, a combo
    ``[[option, ...]]`` or a combo with opts ``[[option, ...], opts]``.
    """
    if isinstance(field, list) and field:
        kind = field[0]
        opts = field[1] if len(field) > 1 and isinstance(field[1], dict) else {}
        return kind, opts
    return field, {}


def validate_workflow(workflow: dict, object_info: dict | None) -> list[str]:
    """Mimic ComfyUI prompt validation so a bad graph never reaches the platform."""
    errors: list[str] = []
    ids = set(workflow)
    slot_count: dict[str, int] = {}

    for node_id, node in workflow.items():
        spec = (object_info or {}).get(node["class_type"])
        if object_info is not None and spec is None:
            errors.append(f"node {node_id}: class_type {node['class_type']!r} is not installed")
            continue
        slot_count[node_id] = len(spec["output"]) if spec else 3

    for node_id, node in workflow.items():
        spec = (object_info or {}).get(node["class_type"])
        declared = {}
        if spec:
            declared = {**spec["input"].get("required", {}), **spec["input"].get("optional", {})}
        for name, value in node["inputs"].items():
            is_link = isinstance(value, list) and len(value) == 2 and isinstance(value[0], str)
            if is_link:
                target, slot = value
                if target not in ids:
                    errors.append(f"node {node_id}.{name}: links to missing node {target!r}")
                elif slot >= slot_count.get(target, 0):
                    errors.append(
                        f"node {node_id}.{name}: node {target} has no output slot {slot} "
                        f"(has {slot_count.get(target)})"
                    )
                continue
            field = declared.get(name)
            if field is None:
                if spec is not None:
                    errors.append(f"node {node_id}.{name}: unknown input for {node['class_type']}")
                continue
            kind, opts = _split_field(field)
            if isinstance(kind, list):  # COMBO
                # image_upload inputs hold a server-side filename, not a bundled
                # example, so any non-empty name is accepted.
                if opts.get("image_upload") and isinstance(value, str) and value:
                    continue
                if value not in kind:
                    errors.append(f"node {node_id}.{name}: {value!r} not in allowed options")
            elif kind == "INT" and not isinstance(value, int):
                errors.append(f"node {node_id}.{name}: expected int, got {value!r}")
            elif kind == "FLOAT" and not isinstance(value, (int, float)):
                errors.append(f"node {node_id}.{name}: expected float, got {value!r}")
            elif kind == "STRING" and not isinstance(value, str):
                errors.append(f"node {node_id}.{name}: expected string, got {value!r}")
            elif kind in {"INT", "FLOAT"} and "min" in opts and value < opts["min"]:
                errors.append(f"node {node_id}.{name}: {value} below min {opts['min']}")
            elif kind in {"INT", "FLOAT"} and "max" in opts and value > opts["max"]:
                errors.append(f"node {node_id}.{name}: {value} above max {opts['max']}")

        if spec is None:
            continue
        provided = set(node["inputs"])
        for name in spec["input"].get("required", {}):
            if name not in provided:
                errors.append(f"node {node_id}.{name}: missing required input")
    return errors


# ---------------------------------------------------------------------------
# Pre-flight gate. Nothing below this line talks to the network; every check is
# local so a graph that would burn a paid run on a broken assumption is stopped
# here. The checks deliberately re-derive the *geometry* the runtime will use
# (mask processing, crop box, resize) instead of trusting the workflow JSON.
# ---------------------------------------------------------------------------

PROTECTED_CORE_ERODE_PX = 8


def _mask_pipeline_np(mask, blend_pixels: int, expand_pixels: int, fill_holes: bool):
    """Port of InpaintCropImproved's mask prep: hipass -> [fill] -> expand -> blur.

    Kept in numpy (the runtime is torch) so the geometry can be recomputed offline;
    ``evidence/p29-fix-run/mask_pipeline_probe.py`` checks this port against the
    real node source line by line.
    """
    import numpy as np
    from scipy import ndimage

    out = mask.copy()
    out[out < CROP_DEFAULTS["mask_hipass_filter"]] = 0
    if fill_holes:
        for threshold in (1, 0.99, 0.97, 0.95, 0.93, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1):
            closed = ndimage.binary_closing(out >= threshold, structure=np.ones((3, 3)), border_value=1)
            filled = ndimage.binary_fill_holes(closed)
            out = np.maximum(out, np.where(filled != 0, threshold, 0))
    for pixels in (expand_pixels, blend_pixels):
        if pixels > 0:
            sigma = pixels / 4
            size = int(np.ceil(sigma * 1.5 + 1))
            out = np.clip(ndimage.grey_dilation(out, size=(size, size)), 0, 1)
    if blend_pixels > 0:
        out = np.clip(ndimage.gaussian_filter(out, sigma=blend_pixels * 0.5), 0, 1)
    return out


def _crop_box(mask, target_w: int, target_h: int, padding: int, factor: float, width: int, height: int):
    """Port of findcontextarea -> growcontextarea -> crop_magic_im's aspect fit."""
    import numpy as np

    ys, xs = np.nonzero(mask)
    bx, by = int(xs.min()), int(ys.min())
    bw, bh = int(xs.max() - bx + 1), int(ys.max() - by + 1)
    gx, gy = round(bw * (factor - 1.0) / 2.0), round(bh * (factor - 1.0) / 2.0)
    bx, by = max(0, bx - gx), max(0, by - gy)
    bw = min(width, bx + bw + 2 * gx) - bx
    bh = min(height, by + bh + 2 * gy) - by
    target_w = -(-target_w // padding) * padding
    target_h = -(-target_h // padding) * padding
    aspect = target_w / target_h
    if bw / bh < aspect:
        bw = int(bh * aspect)
    else:
        bh = int(bw / aspect)
    return dict(x=bx, y=by, w=bw, h=bh, target_w=target_w, target_h=target_h)


def _workflow_diff(before: dict, after: dict) -> list[str]:
    """Node-level diff of two API-format graphs, for the report's evidence trail."""
    lines: list[str] = []
    for node_id in sorted(set(before) | set(after), key=int):
        old, new = before.get(node_id), after.get(node_id)
        if old is None:
            lines.append(f"+ node {node_id} {new['class_type']}")
            continue
        if new is None:
            lines.append(f"- node {node_id} {old['class_type']}")
            continue
        if old["class_type"] != new["class_type"]:
            lines.append(f"~ node {node_id} {old['class_type']} -> {new['class_type']}")
            continue
        changed = [
            f"{key}: {old['inputs'].get(key)!r} -> {value!r}"
            for key, value in new["inputs"].items()
            if old["inputs"].get(key) != value
        ]
        changed += [f"{key}: {old['inputs'][key]!r} -> (removed)"
                    for key in old["inputs"] if key not in new["inputs"]]
        if changed:
            lines.append(f"~ node {node_id} {new['class_type']}: " + "; ".join(changed))
    return lines


def validate_checkpoint(object_info: dict | None, *, checkpoint: str = CHECKPOINT) -> list[str]:
    """Report what the runtime can actually do with this checkpoint.

    The platform's whole checkpoint list is SD1.5 / SDXL base and Flux UNets -- no
    inpainting UNet, and the only inpaint weights are ControlNet / LLLite patches
    for Qwen-Image and Flux. A base SDXL model has no 9-channel inpainting input,
    so ``InpaintModelConditioning``'s ``concat_latent_image`` / ``concat_mask`` are
    ignored and the only thing that survives is its ``noise_mask`` plus the encoded
    original crop. That is a real, supported path -- google's mask-latent inpainting
    -- and it is what this workflow is built on; the verdict is printed so nobody
    later reads the graph as "concat inpainting that silently did nothing".
    """
    notes = [f"checkpoint {checkpoint} has no inpainting UNet input; the run uses the "
             f"masked-latent path (InpaintModelConditioning noise_mask + encoded original crop)"]
    if object_info is None:
        notes.append("WARNING: no object_info, checkpoint existence not confirmed")
        return notes
    spec = object_info.get("CheckpointLoaderSimple")
    options = ((spec or {}).get("input", {}).get("required", {}).get("ckpt_name") or [[]])[0]
    if checkpoint not in options:
        notes.append(f"ERROR: {checkpoint} is not offered by the runtime")
    if any("inpaint" in name.lower() for name in options):
        notes.append("NOTE: the runtime lists an inpainting checkpoint; a future run may "
                     "prefer it and can then use the concat path")
    return notes


def _plan_phrases(prompts: dict, bundle: dict) -> dict[str, str]:
    """The plan's own name for each target, derived from the plan text itself.

    Checking that these survive into the prompts proves the run really is driven
    by the selected plan rather than by a look-alike prompt. Derived rather than
    hard-coded, so a plan other than the tuned one is checked just as strictly.
    """
    phrases: dict[str, str] = {}
    for kind in ("clothing", "hair"):
        text = str(bundle.get(f"{kind}_target") or "").strip()
        if not text:
            text = extract_plan_targets(str(bundle.get(f"{kind}_prompt") or "")).get(
                f"{kind}_target", ""
            )
        if not text:
            text = str(prompts.get(kind) or "")
        token = CJK_TOKEN.search(text)
        if token:
            phrases[kind] = token.group(0)
        else:
            words = text.split()
            phrases[kind] = words[0] if words else ""
    return phrases


def collect_gate_checks(
    workflow: dict,
    object_info: dict | None,
    prompts: dict,
    bundle: dict,
    package: Path,
    *,
    expected_style_id: str | None = EXPECTED_STYLE_ID,
    checkpoint: str = CHECKPOINT,
) -> tuple[list[str], dict]:
    """Every check that must pass before a billed task is created."""
    import numpy as np
    from PIL import Image

    phrases = _plan_phrases(prompts, bundle)
    errors: list[str] = []
    report: dict = {
        "style_id": prompts["style_id"],
        "style_name": prompts.get("style_name"),
        "checkpoint": checkpoint,
        "denoise": dict(PASS_DENOISE),
        "crop": {k: CROP_DEFAULTS[k] for k in
                 ("mask_fill_holes", "mask_expand_pixels", "mask_blend_pixels",
                  "context_from_mask_extend_factor", "output_target_width", "output_target_height")},
        "prompts": {k: prompts[k] for k in ("clothing", "hair", "negative")},
        "style_id_matches_plan": (
            expected_style_id is None or prompts["style_id"] == expected_style_id
        ),
        "plan_phrases": phrases,
    }

    # 1. no plain text-to-image anywhere: an empty latent would ignore the photo.
    for node_id, node in workflow.items():
        if node["class_type"] in {"EmptyLatentImage", "EmptySD3LatentImage", "VAEEncodeForInpaint"}:
            errors.append(f"node {node_id}: {node['class_type']} must not appear in a masked edit")

    passes = {"clothing": ("8", "7", "4"), "hair": ("16", "15", "12")}
    for kind, (sampler_id, cond_id, crop_id) in passes.items():
        sampler, cond = workflow[sampler_id], workflow[cond_id]
        if cond["class_type"] != "InpaintModelConditioning":
            errors.append(f"{kind}: node {cond_id} is {cond['class_type']}, expected InpaintModelConditioning")
            continue
        if sampler["inputs"]["latent_image"] != [cond_id, 2]:
            errors.append(f"{kind}: sampler {sampler_id} latent is not {cond_id}'s latent output")
        if sampler["inputs"]["positive"] != [cond_id, 0] or sampler["inputs"]["negative"] != [cond_id, 1]:
            errors.append(f"{kind}: sampler {sampler_id} conditioning is not {cond_id}'s outputs")
        if cond["inputs"]["pixels"] != [crop_id, 1] or cond["inputs"]["mask"] != [crop_id, 2]:
            errors.append(f"{kind}: {cond_id} is not conditioned on crop node {crop_id}'s image/mask")
        if cond["inputs"]["noise_mask"] is not True:
            errors.append(f"{kind}: noise_mask must be enabled or the mask is not enforced")
        denoise = sampler["inputs"].get("denoise")
        if denoise is None or denoise >= 1.0:
            errors.append(f"{kind}: denoise {denoise} would discard the original crop pixels")
        elif not DENOISE_BAND[0] <= denoise <= DENOISE_BAND[1]:
            errors.append(f"{kind}: denoise {denoise} outside {DENOISE_BAND}")
        if workflow[sampler_id]["inputs"]["model"] != ["3", 0]:
            errors.append(f"{kind}: sampler is not driven by the loaded checkpoint")

    if workflow["10"]["inputs"]["stitcher"] != ["4", 0] or workflow["4"]["inputs"]["image"] != ["1", 0]:
        errors.append("clothing pass does not start from the original photo")
    if workflow["12"]["inputs"]["image"] != ["10", 0]:
        errors.append("hair pass does not start from the clothing pass's stitched output")

    # The two stage records. A SaveImage per pass is how a failed run can be
    # attributed to the clothing pass or the hair pass afterwards; both must hang
    # off existing nodes so the diagnostic adds no sampling.
    saves = {nid: node for nid, node in workflow.items() if node["class_type"] == "SaveImage"}
    report["stage_records"] = {
        "save_nodes": sorted(saves),
        "clothing_pass": [nid for nid, n in saves.items() if n["inputs"]["images"] == ["10", 0]],
        "final": [nid for nid, n in saves.items() if n["inputs"]["images"] == ["18", 0]],
    }
    if not report["stage_records"]["clothing_pass"]:
        errors.append("no SaveImage captures the clothing pass before the hair pass runs")
    if not report["stage_records"]["final"]:
        errors.append("no SaveImage captures the final hair pass")
    extra_saves = set(saves) - set(report["stage_records"]["clothing_pass"]) - set(
        report["stage_records"]["final"]
    )
    if extra_saves:
        errors.append(f"unexpected SaveImage nodes {sorted(extra_saves)}")

    ckpt = validate_checkpoint(object_info, checkpoint=checkpoint)
    report["checkpoint_notes"] = ckpt
    errors.extend(n for n in ckpt if n.startswith("ERROR"))

    # 2. the plan's own wording has to survive into the prompts.
    for kind, phrase in phrases.items():
        if phrase and phrase not in prompts[kind]:
            errors.append(f"{kind} prompt lost the plan phrase {phrase!r}")
    report["prompt_has_plan_phrases"] = all(
        phrase in prompts[kind] for kind, phrase in phrases.items()
    )
    report["negative_prompt_terms_ok"] = all(t in prompts["negative"] for t in REQUIRED_NEGATIVE_TERMS)
    report["collar_clause_ok"] = COLLAR_CLAUSE in prompts["clothing"]
    report["collar_negative_terms_ok"] = all(
        term in prompts["negative"] for term in COLLAR_NEGATIVE_TERMS
    )
    if not report["collar_clause_ok"]:
        errors.append("clothing prompt does not pin a closed, higher collar")
    if not report["collar_negative_terms_ok"]:
        errors.append("negative prompt does not rule out a deep V neckline")

    # 3. files, sizes and coordinates.
    source = json.loads((package / "manifest.json").read_text())["source_image"]
    frame = (source["width"], source["height"])
    sizes = {}
    for name in ("original.jpg", "hair_edit_mask.png", "clothing_edit_mask.png", "protected_identity_mask.png"):
        path = package / name
        if not path.is_file():
            errors.append(f"package is missing {name}")
            continue
        with Image.open(path) as image:
            sizes[name] = image.size
        if sizes[name] != frame:
            errors.append(f"{name} is {sizes[name]}, source photo is {frame}")
    report["image_sizes"] = sizes

    # 4. mask geometry as the node will actually build it.
    if len(sizes) == 4:
        load = lambda n: (np.array(Image.open(package / n).convert("L")) > 127).astype(np.float32)
        hair_raw, clothing_raw = load("hair_edit_mask.png"), load("clothing_edit_mask.png")
        protected = load("protected_identity_mask.png") > 0.5
        from scipy import ndimage
        core = ndimage.binary_erosion(protected, iterations=PROTECTED_CORE_ERODE_PX)
        geometry = {}
        for kind, raw in (("hair", hair_raw), ("clothing", clothing_raw)):
            prepared = _mask_pipeline_np(
                raw,
                blend_pixels=int(CROP_DEFAULTS["mask_blend_pixels"]),
                expand_pixels=int(CROP_DEFAULTS["mask_expand_pixels"]),
                fill_holes=bool(CROP_DEFAULTS["mask_fill_holes"]),
            )
            hard, rim = prepared > 0.5, prepared > 0.02
            box = _crop_box(prepared, int(CROP_DEFAULTS["output_target_width"]),
                            int(CROP_DEFAULTS["output_target_height"]), 32,
                            float(CROP_DEFAULTS["context_from_mask_extend_factor"]), *frame)
            geometry[kind] = {
                "raw_pixels": int((raw > 0.5).sum()),
                "painted_pixels": int(hard.sum()),
                "soft_ring_pixels": int((rim & ~hard).sum()),
                "painted_over_protected": int((hard & protected).sum()),
                "painted_over_facial_core": int((hard & core).sum()),
                "crop": box,
                "scale_x": round(box["target_w"] / box["w"], 4),
                "scale_y": round(box["target_h"] / box["h"], 4),
            }
            if int((hard & core).sum()):
                errors.append(f"{kind}: the painted mask reaches the facial core "
                              f"({int((hard & core).sum())} px inside protected-by-{PROTECTED_CORE_ERODE_PX}px)")
            if abs(geometry[kind]["scale_x"] / geometry[kind]["scale_y"] - 1) > 0.01:
                errors.append(f"{kind}: crop is rescaled non-uniformly "
                              f"({geometry[kind]['scale_x']} x {geometry[kind]['scale_y']})")
        report["mask_geometry"] = geometry

    report["errors"] = errors
    return errors, report


def _post(client: httpx.Client, path: str, key: str, payload: dict) -> dict:
    """POST one RunningHub openapi call and unwrap its envelope.

    Raises :class:`RunningHubError` rather than exiting: the app has to turn a
    failed call into a job status, not into a process exit.
    """
    response = client.post(path, json=payload, headers={"Authorization": f"Bearer {key}"})
    response.raise_for_status()
    body = response.json()
    if body.get("code") not in (0, None):
        raise RunningHubError(
            f"{path} -> code={body.get('code')} msg={body.get('msg')}"
        )
    return body


def upload(client: httpx.Client, key: str, path: Path) -> str:
    with path.open("rb") as handle:
        response = client.post(
            UPLOAD_PATH,
            data={"apiKey": key},
            files={"file": (path.name, handle, "application/octet-stream")},
        )
    response.raise_for_status()
    body = response.json()
    data = body.get("data") or {}
    name = data.get("fileName") or data.get("filename")
    if body.get("code") not in (0, None) or not name:
        raise RunningHubError(
            f"upload {path.name} failed: code={body.get('code')} keys={sorted(data)}"
        )
    return name




# ---------------------------------------------------------------------------
# Network stage. Everything below talks to RunningHub; the paid call is
# ``create_task`` and it is deliberately the only function that spends money.
# ---------------------------------------------------------------------------


def create_task(
    client: httpx.Client,
    key: str,
    workflow: dict,
    *,
    workflow_id: str,
    instance_type: str,
) -> dict:
    """Create the ONE billed task for this workflow. Never call this twice."""
    payload = {
        "apiKey": key,
        "workflowId": workflow_id,
        "workflow": json.dumps(workflow),
        "nodeInfoList": [],
        "instanceType": instance_type,
    }
    created = _post(client, CREATE_PATH, key, payload)
    data = created.get("data") or {}
    if not data.get("taskId"):
        raise RunningHubError(f"create returned no taskId: {sorted(data)}")
    return created


def task_status(client: httpx.Client, key: str, task_id: str) -> str:
    return str(_post(client, STATUS_PATH, key, {"apiKey": key, "taskId": task_id})["data"])


def poll_task(
    client: httpx.Client,
    key: str,
    task_id: str,
    *,
    timeout_seconds: float,
    interval_seconds: float = 6.0,
    on_status: Callable[[str], None] | None = None,
) -> str:
    """Poll a created task until it succeeds, fails or the deadline passes."""
    deadline = time.time() + timeout_seconds
    status = "QUEUED"
    while True:
        if on_status is not None:
            on_status(status)
        if status in {"SUCCESS", "FAILED"}:
            return status
        if time.time() >= deadline:
            return status
        time.sleep(min(interval_seconds, max(0.0, deadline - time.time())))
        status = task_status(client, key, task_id)


def fetch_outputs(client: httpx.Client, key: str, task_id: str) -> list[dict]:
    data = _post(client, OUTPUTS_PATH, key, {"apiKey": key, "taskId": task_id})["data"]
    return list(data or [])


def download(client: httpx.Client, url: str) -> bytes:
    response = client.get(url)
    response.raise_for_status()
    return response.content


def pick_image_output(outputs: list[dict]) -> dict:
    """The finished frame, not the clothing-pass stage record.

    Node 19 saves the final stitch and node 20 saves the clothing pass; the
    platform returns them in workflow order, so the last image is the final one.
    """
    images = [item for item in outputs if str(item.get("fileUrl") or "").strip()]
    if not images:
        raise RunningHubError("the task returned no image output")
    return images[-1]


def run_task(
    client: httpx.Client,
    key: str,
    workflow: dict,
    *,
    workflow_id: str,
    instance_type: str,
    timeout_seconds: float,
    on_status: Callable[[str], None] | None = None,
) -> tuple[str, str]:
    """Create, poll and download in one call, returning ``(task_id, image_url)``.

    One task, one billed request, no retry: if the create call's outcome is not
    known the caller must query the task list rather than resubmit.
    """
    created = create_task(
        client, key, workflow, workflow_id=workflow_id, instance_type=instance_type
    )
    task_id = str(created["data"]["taskId"])
    status = poll_task(
        client, key, task_id, timeout_seconds=timeout_seconds, on_status=on_status
    )
    if status != "SUCCESS":
        raise RunningHubError(f"task {task_id} did not succeed (status={status})")
    output = pick_image_output(fetch_outputs(client, key, task_id))
    return task_id, str(output["fileUrl"])


__all__ = [
    "BASE_URL",
    "CHECKPOINT",
    "COLLAR_CLAUSE",
    "COLLAR_NEGATIVE_TERMS",
    "CROP_DEFAULTS",
    "DEFAULT_MASK_RULE",
    "DENOISE_BAND",
    "EXPECTED_STYLE_ID",
    "FORBIDDEN_POSITIVE_PATTERN",
    "MASK_RULE_ID",
    "NEGATIVE_PROMPT",
    "PASS_DENOISE",
    "REQUIRED_NEGATIVE_TERMS",
    "RunningHubError",
    "TUNED_STYLE_ID",
    "build_prompts",
    "build_workflow",
    "collect_gate_checks",
    "create_task",
    "download",
    "extract_plan_targets",
    "fetch_outputs",
    "pick_image_output",
    "poll_task",
    "run_task",
    "task_status",
    "upload",
    "validate_checkpoint",
    "validate_workflow",
]
