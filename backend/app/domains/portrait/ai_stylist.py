"""Turn a validated AI styling payload into domain proposals.

The model does the *designing* (see :mod:`app.brains.styling_planner`); this
module only translates its structured answer into the same ``StylingProposal``
shape the deterministic ``_Axis`` template produces. That means the whole
downstream workflow -- the cards, the single-look generation, the refine loop --
is unchanged and does not need to know which mode authored the design.

Two entry points:

* :func:`preset_catalogue` / :func:`valid_preset_ids` -- the exact stylekit
  vocabulary handed to the model, so it can only pick presets that really exist.
* :func:`build_ai_proposals` -- pure function (no network, no model) that maps
  the model's JSON onto ``StylingProposal`` objects, resolves every chosen
  preset id back to the real stylekit fragment, and composes the concrete image
  edit prompt.

Keeping this layer free of any ``app.brains`` import is deliberate: the domain
package must not depend on the model layer (see
``tests/test_architecture_boundaries.py``), so the planner hands over plain data.
"""

from __future__ import annotations

import re

from app.domain.styling import StylingProposal
from app.domains.portrait.style_vocabulary import (
    HAIRSTYLE_BUCKETS,
    LENGTH_BAND_LABELS,
    StyleFragment,
    StyleVocabulary,
)
from app.domains.portrait.stylist import (
    BASE_NEGATIVES,
    _current_hair_en,
    _current_haircolor_en,
    _current_top_en,
    _identity_anchors_zh,
    _identity_anchors_en,
    _is_known,
    _reference,
)

# The buckets the model must fill, in the order they appear in ``selectedPresets``.
AI_BUCKETS: tuple[str, ...] = (
    "hairstyle",
    "haircolor",
    "makeup",
    "outfit",
    "accessory",
)

_INTENSITIES = {"mild", "medium", "bold"}


def preset_catalogue(
    vocabulary: StyleVocabulary,
) -> dict[str, list[dict[str, object]]]:
    """The stylekit presets the AI may choose from, grouped by bucket.

    Female and male hairstyles are merged into a single ``hairstyle`` bucket so
    the model can weigh every silhouette; the chosen id is resolved back to the
    real bucket (and file) afterwards, so nothing is invented and the design
    still traces to an actual preset.
    """
    catalogue: dict[str, list[dict[str, object]]] = {}
    for bucket in AI_BUCKETS:
        entries: list[dict[str, object]] = []
        seen: set[str] = set()
        if bucket == "hairstyle":
            fragments = [
                fragment
                for hair_bucket in HAIRSTYLE_BUCKETS
                for fragment in vocabulary.list(hair_bucket)
            ]
        else:
            fragments = list(vocabulary.list(bucket))
        for fragment in fragments:
            if fragment.preset_id in seen:
                continue
            seen.add(fragment.preset_id)
            entries.append(
                {
                    "presetId": fragment.preset_id,
                    "nameZh": fragment.name_zh,
                    "descriptionZh": fragment.description_zh,
                    "bestFor": dict(fragment.best_for or {}),
                }
            )
        catalogue[bucket] = entries
    return catalogue


def valid_preset_ids(
    vocabulary: StyleVocabulary,
    catalogue: dict[str, list[dict[str, object]]] | None = None,
) -> dict[str, set[str]]:
    """The ids the catalogue above exposes, used to reject invented presets.

    ``catalogue`` may be passed in when the caller already built it, so one
    design pass never composes the vocabulary twice.
    """
    catalogue = catalogue if catalogue is not None else preset_catalogue(vocabulary)
    return {
        bucket: {entry["presetId"] for entry in entries}
        for bucket, entries in catalogue.items()
    }


def person_identity_anchors(facts: dict) -> tuple[str, ...]:
    """Observable identity anchors to hand the planner, in Chinese.

    Reuses the exact anchors the template stylist pins onto its prompts, so both
    design modes protect the same visible features of the same person.
    """
    return _identity_anchors_zh(facts)


def build_ai_proposals(
    planner_payload: dict,
    facts: dict,
    domain_type: str,
    vocabulary: StyleVocabulary,
    provider_name: str,
) -> tuple[StylingProposal, ...]:
    """Map a validated styling payload onto three ``StylingProposal`` objects.

    The payload is assumed to have already passed the planner's validation (three
    distinct looks, every preset id present in the catalogue). A missing preset
    here means the vocabulary shifted underneath us, so it fails loudly rather
    than quietly fabricating a fragment.
    """
    proposals = planner_payload.get("proposals")
    if not isinstance(proposals, list) or len(proposals) < 3:
        raise RuntimeError("AI styling payload did not contain three complete looks.")

    used_plan_ids: set[str] = set()
    built: list[StylingProposal] = []
    for index, payload in enumerate(proposals[:3], start=1):
        if not isinstance(payload, dict):
            raise RuntimeError("AI styling payload contained a malformed look.")
        built.append(
            _build_ai_proposal(
                payload,
                facts,
                domain_type,
                vocabulary,
                provider_name,
                index,
                used_plan_ids,
            )
        )
    return tuple(built)


# --------------------------------------------------------------------- helpers


def _build_ai_proposal(
    payload: dict,
    facts: dict,
    domain_type: str,
    vocabulary: StyleVocabulary,
    provider_name: str,
    index: int,
    used_plan_ids: set[str],
) -> StylingProposal:
    selection = payload.get("selectedPresets")
    if not isinstance(selection, dict):
        raise RuntimeError("AI styling look is missing its preset selection.")

    hair = _resolve_hair(vocabulary, str(selection.get("hair") or "").strip())
    haircolor = _resolve(vocabulary, "haircolor", selection.get("haircolor"))
    makeup = _resolve(vocabulary, "makeup", selection.get("makeup"))
    outfit = _resolve(vocabulary, "outfit", selection.get("outfit"))
    accessory = _resolve(vocabulary, "accessory", selection.get("accessory"))

    name = str(payload.get("name") or "").strip()
    if not name:
        raise RuntimeError("AI styling look is missing a name.")
    direction = str(payload.get("designDirection") or "").strip()
    reason = str(payload.get("designReason") or "").strip()
    description = str(payload.get("description") or "").strip() or direction or name
    why_it_fits = str(payload.get("whyItFits") or "").strip() or reason

    intensity = str(payload.get("intensity") or "").strip().lower()
    if intensity not in _INTENSITIES:
        intensity = "medium"

    hair_fields = payload.get("hair") if isinstance(payload.get("hair"), dict) else {}
    makeup_fields = payload.get("makeup") if isinstance(payload.get("makeup"), dict) else {}
    outfit_fields = payload.get("outfit") if isinstance(payload.get("outfit"), dict) else {}

    length_band = LENGTH_BAND_LABELS.get(
        hair.length_band, LENGTH_BAND_LABELS["unknown"]
    )
    hair_plan = (
        f"长度：{_text(hair_fields.get('length'))}（参考长度档位：{length_band}）；"
        f"结构：{_text(hair_fields.get('structure'))}；"
        f"刘海：{_text(hair_fields.get('bangs'))}；"
        f"层次：{_text(hair_fields.get('layers'))}。"
        f"发色改为「{haircolor.name_zh}」：{haircolor.description_zh}。"
        f"参考风格「{hair.name_zh}」：{hair.description_zh}"
    )
    makeup_plan = (
        f"眉毛：{_text(makeup_fields.get('brows'))}；"
        f"眼妆：{_text(makeup_fields.get('eyes'))}；"
        f"唇妆：{_text(makeup_fields.get('lips'))}；"
        f"底妆：{_text(makeup_fields.get('base'))}。"
        f"参考妆容「{makeup.name_zh}」：{makeup.description_zh}"
    )
    outfit_plan = (
        f"款式：{_text(outfit_fields.get('style'))}；"
        f"材质：{_text(outfit_fields.get('material'))}；"
        f"颜色：{_text(outfit_fields.get('color'))}；"
        f"搭配：{_text(outfit_fields.get('pairing'))}。"
        f"参考穿搭「{outfit.name_zh}」：{outfit.description_zh}"
    )
    accessory_text = _text(payload.get("accessory"))
    accessory_plan = (
        f"{accessory_text}。参考配饰「{accessory.name_zh}」：{accessory.description_zh}"
    )

    anchors_zh = _merge_anchors(payload.get("identityAnchors"), _identity_anchors_zh(facts))
    palette = _text(payload.get("palette")) or direction

    expected_changes = (
        f"发型：{_current_label(facts.get('current_hair_length_label'), '现有发型')} → {hair.name_zh}",
        f"发色：{_current_label(facts.get('hair_color_name'), '现有发色')} → {haircolor.name_zh}",
        f"妆容：现有妆容 → {makeup.name_zh}（{_text(makeup_fields.get('eyes'))}）",
        f"服装：{_current_label(facts.get('upper_body_color_name'), '现有上衣')} → {outfit.name_zh}",
        f"配饰：新增或替换为{accessory.name_zh}",
    )

    edit_prompt = _build_edit_prompt(
        name=name,
        direction=direction,
        payload=payload,
        facts=facts,
        hair=hair,
        haircolor=haircolor,
        makeup=makeup,
        outfit=outfit,
        accessory=accessory,
        hair_fields=hair_fields,
        makeup_fields=makeup_fields,
        outfit_fields=outfit_fields,
        accessory_text=accessory_text,
        anchors_en=_identity_anchors_en(facts),
    )

    negatives = [
        _text(item) for item in (payload.get("negativePrompt") or []) if _text(item)
    ]

    plan_id = _unique_plan_id(payload.get("slug"), index, used_plan_ids)
    selected = (hair.preset_id, haircolor.preset_id, makeup.preset_id, outfit.preset_id, accessory.preset_id)

    return StylingProposal(
        plan_id=plan_id,
        domain_type=domain_type,
        title=name,
        name=name,
        design_axis=plan_id,
        design_direction=direction or name,
        description=description,
        design_reason=reason or description,
        intensity=intensity,
        hair_plan=hair_plan,
        makeup_plan=makeup_plan,
        outfit_plan=outfit_plan,
        accessory_plan=accessory_plan,
        palette=_palette_with_skin(palette, facts),
        identity_anchors=anchors_zh,
        why_it_fits=why_it_fits,
        expected_changes=expected_changes,
        edit_prompt=edit_prompt,
        negative_prompt="避免" + "、".join((*negatives, *BASE_NEGATIVES)),
        reference_presets=(
            _reference(hair),
            _reference(haircolor),
            _reference(makeup),
            _reference(outfit),
            _reference(accessory),
        ),
        estimated_paid_requests=1,
        design_mode="ai",
        source_provider=provider_name,
        selected_presets=selected,
    )


def _build_edit_prompt(
    *,
    name: str,
    direction: str,
    payload: dict,
    facts: dict,
    hair: StyleFragment,
    haircolor: StyleFragment,
    makeup: StyleFragment,
    outfit: StyleFragment,
    accessory: StyleFragment,
    hair_fields: dict,
    makeup_fields: dict,
    outfit_fields: dict,
    accessory_text: str,
    anchors_en: tuple[str, ...],
) -> str:
    hair_desc = " ".join(
        part
        for part in (
            _text(hair_fields.get("length")),
            _text(hair_fields.get("structure")),
            _text(hair_fields.get("bangs")),
            _text(hair_fields.get("layers")),
        )
        if part
    )
    makeup_desc = " ".join(
        part
        for part in (
            _text(makeup_fields.get("base")),
            _text(makeup_fields.get("brows")),
            _text(makeup_fields.get("eyes")),
            _text(makeup_fields.get("lips")),
        )
        if part
    )
    outfit_desc = " ".join(
        part
        for part in (
            _text(outfit_fields.get("style")),
            _text(outfit_fields.get("material")),
            _text(outfit_fields.get("color")),
            _text(outfit_fields.get("pairing")),
        )
        if part
    )
    palette = _text(payload.get("palette"))
    ai_instruction = _text(payload.get("editPrompt"))

    lines = (
        f"Restyle the person in the uploaded photo into the '{name}' look "
        f"({direction}). Edit this photo directly; keep the same person, "
        f"do not generate a different person.",
        "Identity anchors (best effort, no absolute guarantee): "
        + "; ".join(anchors_en)
        + ".",
        f"Hair: change the hair from {_current_hair_en(facts)} to {hair.name_zh} "
        f"({hair.name_en}). {hair_desc} "
        f"Also change the hair colour from {_current_haircolor_en(facts)} to "
        f"{haircolor.name_zh} ({haircolor.name_en}). {hair.prompt}".strip(),
        f"Makeup: change the makeup from the current makeup to {makeup.name_zh} "
        f"({makeup.name_en}). {makeup_desc or makeup.prompt}".strip(),
        f"Outfit: replace the current {_current_top_en(facts)} with {outfit.name_zh} "
        f"({outfit.name_en}). {outfit_desc or outfit.prompt}".strip(),
        f"Accessories: {accessory_text or accessory.name_zh}. {accessory.prompt}".strip(),
        f"Colour palette: {palette}." if palette else "",
        # Provenance: pin the exact stylekit presets this instruction was
        # composed from, so a prompt can always be traced back to real
        # vocabulary (and can never drift from ``selectedPresets``).
        "Stylekit presets applied: "
        f"hairstyle={hair.preset_id}, haircolor={haircolor.preset_id}, "
        f"makeup={makeup.preset_id}, outfit={outfit.preset_id}, "
        f"accessory={accessory.preset_id}.",
        f"Styling director's specific instructions: {ai_instruction}"
        if ai_instruction
        else "",
    )
    return "\n".join(line for line in lines if line)


def _palette_with_skin(palette: str, facts: dict) -> str:
    tone = facts.get("skin_temperature_label")
    if _is_known(tone):
        return f"{palette}；与「{tone}」肤色基调协调"
    return palette


def _merge_anchors(raw: object, local: tuple[str, ...]) -> tuple[str, ...]:
    merged: list[str] = []
    if isinstance(raw, (list, tuple)):
        for item in raw:
            text = _text(item)
            if text and text not in merged:
                merged.append(text)
    for anchor in local:
        if anchor not in merged:
            merged.append(anchor)
    return tuple(merged)


def _text(value: object) -> str:
    return str(value).strip() if value is not None else ""


def _current_label(value: object, fallback: str) -> str:
    return str(value) if _is_known(value) else fallback


def _unique_plan_id(raw_slug: object, index: int, used: set[str]) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", str(raw_slug or "").strip().lower()).strip("_")
    base = f"ai_{slug}" if slug else f"ai_look_{index}"
    candidate = base
    suffix = 2
    while candidate in used:
        candidate = f"{base}_{suffix}"
        suffix += 1
    used.add(candidate)
    return candidate


def _resolve(
    vocabulary: StyleVocabulary,
    bucket: str,
    raw_id: object,
) -> StyleFragment:
    preset_id = str(raw_id or "").strip()
    fragment = vocabulary.get(bucket, preset_id)
    if fragment is None:
        raise RuntimeError(
            f"AI styling selected an unknown {bucket} preset: {preset_id!r}"
        )
    return fragment


def _resolve_hair(vocabulary: StyleVocabulary, preset_id: str) -> StyleFragment:
    fragment = vocabulary.find_hair(preset_id)
    if fragment is None:
        raise RuntimeError(f"AI styling selected an unknown hairstyle preset: {preset_id!r}")
    return fragment
