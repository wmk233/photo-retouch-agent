"""AI stylist: designs three complete, distinct looks for one person.

The stylist does not invent styling vocabulary. Each look is composed out of the
curated presets that ship with the sibling ``stylekit`` package (see
``style_vocabulary``), then turned into person-specific copy plus a concrete
image-edit prompt.

Design rules:

* Three mutually exclusive art directions ("axes"): a crisp urban crop, an
  artistic long wave and a bold street look. Every axis selects *different*
  presets for hair, hair colour, makeup, outfit and accessories, so the three
  proposals never collapse into three adjectives on top of one edit.
* The copy is driven by the observable ``FaceFacts`` measured from the photo
  (face shape, skin tone, current hair length/colour, glasses, current top
  colour). No private attribute (gender, age, ethnicity, occupation,
  personality) is ever inferred.
* Every prompt carries explicit identity anchors taken from those same facts,
  and hedges the promise: face preservation is best effort, never guaranteed.

``StylingProposal`` is shape-compatible with ``RetouchPlan`` for the retouch
fields (``planId``/``editPrompt``/``negativePrompt``/...), so a chosen proposal
can be posted straight to ``POST /api/retouch/jobs`` without translation.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.domain.analysis import PhotoAnalysis
from app.domain.styling import StylingProposal, StylingReference
from app.domains.portrait.style_vocabulary import (
    LENGTH_BAND_LABELS,
    StyleFragment,
    StyleVocabulary,
    fallback_fragment,
)

UNKNOWN = "未识别"

FACE_SHAPE_LABELS = {
    "round": "圆润脸型",
    "oval": "均衡脸型",
    "square": "方正脸型",
    "heart": "心形脸型",
    "oblong": "偏长脸型",
}

FACE_SHAPE_EN = {
    "round": "round",
    "oval": "oval",
    "square": "square",
    "heart": "heart-shaped",
    "oblong": "elongated",
}

HAIR_LENGTH_EN = {
    "short": "short",
    "medium": "shoulder-length",
    "long": "long",
    "unknown": "current",
}

BASE_NEGATIVES: tuple[str, ...] = (
    "换成陌生人的脸",
    "五官整体重塑",
    "年龄感明显漂移",
    "人物数量变化",
    "塑料感磨皮",
    "面部模糊",
    "背景崩坏",
    "多余的手指或肢体",
)


@dataclass(frozen=True)
class _Axis:
    """One authored art direction, plus where it draws its stylekit presets."""

    axis_id: str
    name: str
    direction: str
    description: str
    intensity: str
    reason: str
    why_template: str
    hair_ids: tuple[str, ...]
    hair_length: str
    hair_structure: str
    hair_bangs: str
    hair_layers: str
    hair_en: str
    haircolor_ids: tuple[str, ...]
    makeup_ids: tuple[str, ...]
    makeup_brows: str
    makeup_eyes: str
    makeup_lips: str
    makeup_base: str
    makeup_en: str
    outfit_ids: tuple[str, ...]
    outfit_style: str
    outfit_material: str
    outfit_color: str
    outfit_pairing: str
    outfit_en: str
    accessory_ids: tuple[str, ...]
    accessory_use: str
    accessory_en: str
    palette: str
    palette_en: str
    negatives: tuple[str, ...]


AXES: tuple[_Axis, ...] = (
    _Axis(
        axis_id="urban_short",
        name="都市利落短发",
        direction="利落短发 · 干净裸妆 · 智性通勤",
        description="剪短轮廓、露出颈部线条，用干净底妆和利落通勤装把注意力拉回眉眼。",
        intensity="medium",
        reason="短发把视觉重心上移到眉眼，低饱和的通勤配色让整体更利落。",
        why_template="{face}配上顶部有体积、两侧收紧的短发会强化下颌与颈部的清爽线条，日常也最好打理。",
        hair_ids=("korean_perm_short", "pixie_cut", "french_bangs_bob"),
        hair_length="耳下到下巴之间的短发",
        hair_structure="顶部保留体积、两侧收紧的利落剪裁",
        hair_bangs="轻薄空气感碎刘海，或直接露额展示眉眼",
        hair_layers="后颈做高层次收薄，轮廓干净不厚重",
        hair_en="a crisp chin-length short cut with soft volume on top and a clean tapered nape",
        haircolor_ids=("chocolate_brown", "cool_tea_brown", "milk_tea_brown"),
        makeup_ids=("clean_nude", "cool_neutral_male"),
        makeup_brows="自然平直眉，弱化眉峰",
        makeup_eyes="奶茶色眼影配细致眼线，强调眼神",
        makeup_lips="雾面豆沙唇",
        makeup_base="干净哑光底妆，保留皮肤纹理",
        makeup_en="a clean matte everyday makeup with soft natural brows and a muted rose lip",
        outfit_ids=("business_casual", "prep_blazer", "knit_sweater"),
        outfit_style="合身西装外套或细针织上衣",
        outfit_material="羊毛与棉混纺，版型挺括",
        outfit_color="藏蓝、燕麦、炭灰等低饱和色",
        outfit_pairing="配简约直筒裤与干净皮鞋",
        outfit_en="a tailored low-saturation smart-casual outfit (structured blazer or fine knit top)",
        accessory_ids=("round_metal_glasses", "pearl_stud_earrings"),
        accessory_use="用小幅金属质感或珍珠点缀上半身，不抢五官",
        accessory_en="subtle round metal glasses, or small pearl stud earrings",
        palette="燕麦白 + 藏蓝 + 炭灰的低饱和通勤配色",
        palette_en="oatmeal white, navy and charcoal — a low-saturation commuter palette",
        negatives=("夸张发色", "烟熏眼妆", "街头大廓形"),
    ),
    _Axis(
        axis_id="artistic_wave",
        name="艺术感长发微卷",
        direction="蓬松长卷 · 个性发色 · 复古法式",
        description="用大波浪和个性发色放大氛围感，妆容走甜系，服装走复古法式。",
        intensity="medium",
        reason="波浪的体积感能平衡面部比例，个性发色让造型一眼就有记忆点。",
        why_template="{face}搭配中段起卷的长发能柔化面部线条，个性发色把整体的艺术氛围拉满。",
        hair_ids=("air_bangs_long", "side_part_waves", "beach_waves"),
        hair_length="过肩长发",
        hair_structure="中段起做蓬松大卷，发尾自然外翻",
        hair_bangs="法式八字长刘海，修饰颧骨",
        hair_layers="中高层次让发丝有空气感与重量感",
        hair_en="shoulder-past long hair with soft large waves and curtain bangs",
        haircolor_ids=("burgundy_brown", "muted_green", "misty_pink_purple"),
        makeup_ids=("peach_sweet", "korean_dewy"),
        makeup_brows="自然野生眉，眉尾柔和",
        makeup_eyes="蜜桃色眼影配卧蚕提亮",
        makeup_lips="水润蜜桃唇",
        makeup_base="韩系水光肌底妆，颧骨轻扫腮红",
        makeup_en="a sweet dewy makeup with peach eyeshadow and a glossy peach lip",
        outfit_ids=("vintage_french", "vintage_chinese"),
        outfit_style="法式复古衬衫叠穿或连衣裙",
        outfit_material="真丝、雪纺与棉麻，垂顺有光泽",
        outfit_color="奶白、酒红、复古绿",
        outfit_pairing="用细腰带或丝巾收束腰线",
        outfit_en="a vintage French outfit in cream, burgundy and muted green with soft draping",
        accessory_ids=("gold_hoop_earrings", "wool_beanie"),
        accessory_use="用金色圈形耳环放大复古氛围",
        accessory_en="gold hoop earrings",
        palette="奶白 + 酒红 + 复古金的法式复古配色",
        palette_en="cream, burgundy and antique gold — a vintage French palette",
        negatives=("贴头皮的直发", "哑光工装", "冷调烟熏妆"),
    ),
    _Axis(
        axis_id="bold_street",
        name="先锋街头高马尾",
        direction="高马尾抓髻 · 冷调发色 · 街头机能",
        description="把头发梳起做大轮廓，冷调发色配机能街头装，直接把气场拉出来。",
        intensity="bold",
        reason="上提的发型结构强调眼神和下颌，冷色调与机能面料带来强烈的视觉反差。",
        why_template="{face}在头发上提、露颈的结构下会更显气场，冷调发色和街头廓形能形成完全不同的一面。",
        hair_ids=("high_ponytail_bangs", "man_bun", "side_part_waves"),
        hair_length="长发全部上梳，露出颈部",
        hair_structure="高马尾或抓髻，顶部制造高体积",
        hair_bangs="两侧留少量胎毛碎发修饰发际线",
        hair_layers="额前与鬓角做碎层次，保持锋利凌乱感",
        hair_en="long hair swept up into a high ponytail or top knot with a tall silhouette",
        haircolor_ids=("misty_blue_gray", "navy_blue", "dreamy_lake_blue", "misty_ash_brown"),
        makeup_ids=("smoky_evening", "peach_sweet"),
        makeup_brows="微挑剑眉，增强锐度",
        makeup_eyes="冷调烟熏眼妆配下眼线",
        makeup_lips="哑光深色唇",
        makeup_base="半哑光底妆，颧骨轻打阴影",
        makeup_en="a cool smoky eye with a deep matte lip",
        outfit_ids=("oversized_streetwear", "techwear", "party_leather_male"),
        outfit_style="宽松机能外套或皮质外套",
        outfit_material="尼龙与皮革等有光泽的材质",
        outfit_color="黑色、冷灰、亮蓝",
        outfit_pairing="配工装裤与厚底鞋",
        outfit_en="oversized streetwear or techwear outerwear in black, cool grey and electric blue",
        accessory_ids=("black_baseball_cap", "aviator_sunglasses"),
        accessory_use="用棒球帽或飞行员墨镜强化街头感",
        accessory_en="a black baseball cap, or aviator sunglasses",
        palette="黑 + 冷灰 + 电光蓝的街头冷调配色",
        palette_en="black, cool grey and electric blue — a cold street palette",
        negatives=("甜系腮红", "法式碎花", "柔软毛呢"),
    ),
)


def _fragment(
    vocabulary: StyleVocabulary,
    bucket: str,
    preset_ids: tuple[str, ...],
    label: str,
) -> StyleFragment:
    """Resolve the first available preset, falling back to the built-in set."""
    fragment = vocabulary.first(bucket, preset_ids)
    if fragment is not None:
        return fragment
    for preset_id in preset_ids:
        fragment = fallback_fragment(bucket, preset_id)
        if fragment is not None:
            return fragment
    available = vocabulary.list(bucket)
    if available:
        return available[0]
    raise RuntimeError(f"style vocabulary has no fragment for {bucket}:{label}")


def _reference(fragment: StyleFragment) -> StylingReference:
    return StylingReference(
        preset_id=fragment.preset_id,
        bucket=fragment.bucket,
        name=fragment.name_zh,
        source_file=fragment.source_file,
        vocabulary_ref=fragment.ref,
        prompt=fragment.prompt,
    )


def _is_known(value: object) -> bool:
    if not value:
        return False
    text = str(value)
    if text == "unknown" or text.startswith("未") or "未能" in text:
        return False
    return True


def _face_shape_label(facts: dict) -> str | None:
    shape = facts.get("face_shape")
    if shape not in FACE_SHAPE_LABELS:
        return None
    reason = str(facts.get("face_shape_reason") or "")
    if "不足" in reason:
        return None
    return FACE_SHAPE_LABELS[shape]


def _identity_anchors_zh(facts: dict) -> tuple[str, ...]:
    anchors: list[str] = []
    shape_label = _face_shape_label(facts)
    if shape_label:
        anchors.append(f"保留{shape_label}的轮廓与颧骨位置")
    tone = facts.get("skin_tone_label")
    if _is_known(tone):
        anchors.append(f"保留原本「{tone}」的肤色基调")
    if facts.get("wears_glasses"):
        eyewear = facts.get("eyewear_label")
        anchors.append(
            f"保留原本佩戴的{eyewear}" if _is_known(eyewear) else "保留原本佩戴的眼镜"
        )
    forehead = facts.get("forehead_height_label")
    if _is_known(forehead):
        anchors.append(f"保留原有发际线特征（{forehead}）")
    cheek = facts.get("cheek_fullness")
    if _is_known(cheek):
        anchors.append(f"保留原图「{cheek}」的面部轮廓比例")
    anchors.append("保留原图五官的相对位置与比例")
    return tuple(anchors)


def _identity_anchors_en(facts: dict) -> tuple[str, ...]:
    anchors: list[str] = []
    shape = facts.get("face_shape")
    if _face_shape_label(facts) and shape in FACE_SHAPE_EN:
        anchors.append(f"keep the {FACE_SHAPE_EN[shape]} facial contour and cheekbone placement")
    tone = facts.get("skin_tone_label")
    if _is_known(tone):
        anchors.append("keep the natural skin tone")
    if facts.get("wears_glasses"):
        anchors.append("keep the person's glasses")
    if _is_known(facts.get("forehead_height_label")):
        anchors.append("keep the hairline height")
    anchors.append("keep the relative position and proportion of the facial features")
    return tuple(anchors)


def _current_hair_en(facts: dict) -> str:
    band = HAIR_LENGTH_EN.get(str(facts.get("current_hair_length") or "unknown"), "current")
    if band == "current":
        return "the person's current hair"
    return f"the person's current {band} hair"


def _current_haircolor_en(facts: dict) -> str:
    colour = facts.get("hair_color_name")
    if _is_known(colour):
        return f"its current colour ({colour})"
    return "its current colour"


def _current_top_en(facts: dict) -> str:
    colour = facts.get("upper_body_color_name")
    if _is_known(colour):
        return f"top (currently {colour})"
    return "top"


def _palette_with_skin(axis: _Axis, facts: dict) -> str:
    tone = facts.get("skin_temperature_label")
    if _is_known(tone):
        return f"{axis.palette}；与「{tone}」肤色基调协调"
    return axis.palette


def _build_proposal(
    axis: _Axis,
    facts: dict,
    domain_type: str,
    vocabulary: StyleVocabulary,
) -> StylingProposal:
    hair = _fragment(vocabulary, "hairstyle_female", axis.hair_ids, f"{axis.axis_id}:hair")
    haircolor = _fragment(vocabulary, "haircolor", axis.haircolor_ids, f"{axis.axis_id}:haircolor")
    makeup = _fragment(vocabulary, "makeup", axis.makeup_ids, f"{axis.axis_id}:makeup")
    outfit = _fragment(vocabulary, "outfit", axis.outfit_ids, f"{axis.axis_id}:outfit")
    accessory = _fragment(vocabulary, "accessory", axis.accessory_ids, f"{axis.axis_id}:accessory")

    length_label = LENGTH_BAND_LABELS.get(hair.length_band, LENGTH_BAND_LABELS["unknown"])
    hair_plan = (
        f"长度：{axis.hair_length}（参考长度档位：{length_label}）；"
        f"结构：{axis.hair_structure}；刘海：{axis.hair_bangs}；层次：{axis.hair_layers}。"
        f"参考风格「{hair.name_zh}」：{hair.description_zh}。"
        f"发色改为「{haircolor.name_zh}」：{haircolor.description_zh}"
    )
    makeup_plan = (
        f"眉毛：{axis.makeup_brows}；眼妆：{axis.makeup_eyes}；"
        f"唇妆：{axis.makeup_lips}；底妆：{axis.makeup_base}。"
        f"参考妆容「{makeup.name_zh}」：{makeup.description_zh}"
    )
    outfit_plan = (
        f"款式：{axis.outfit_style}；材质：{axis.outfit_material}；"
        f"颜色：{axis.outfit_color}；搭配：{axis.outfit_pairing}。"
        f"参考穿搭「{outfit.name_zh}」：{outfit.description_zh}"
    )
    accessory_plan = (
        f"{axis.accessory_use}。参考配饰「{accessory.name_zh}」：{accessory.description_zh}"
    )

    anchors_zh = _identity_anchors_zh(facts)
    anchors_en = _identity_anchors_en(facts)
    shape_label = _face_shape_label(facts) or "当前脸型"
    why_it_fits = axis.why_template.format(face=shape_label)
    directions = [d for d in facts.get("styling_directions") or [] if _is_known(d)]
    if directions:
        why_it_fits = f"{why_it_fits}照片观察到：{'；'.join(directions[:2])}。"

    design_reason = axis.reason
    shape_reason = facts.get("face_shape_reason")
    if _face_shape_label(facts) and _is_known(shape_reason):
        design_reason = f"针对{shape_label}：{shape_reason}。{axis.reason}"

    expected_changes = (
        f"发型：{facts.get('current_hair_length_label') if _is_known(facts.get('current_hair_length_label')) else '现有发型'} → {hair.name_zh}",
        f"发色：{facts.get('hair_color_name') if _is_known(facts.get('hair_color_name')) else '现有发色'} → {haircolor.name_zh}",
        f"妆容：现有妆容 → {makeup.name_zh}（{axis.makeup_eyes}）",
        f"服装：{facts.get('upper_body_color_name') if _is_known(facts.get('upper_body_color_name')) else '现有上衣'} → {outfit.name_zh}",
        f"配饰：新增或替换为{accessory.name_zh}",
    )

    edit_prompt = "\n".join(
        line
        for line in (
            f"Restyle the person in the uploaded photo into the '{axis.name}' look "
            f"({axis.direction}). Edit this photo directly; keep the same person, "
            f"do not generate a different person.",
            "Identity anchors (best effort, no absolute guarantee): "
            + "; ".join(anchors_en)
            + ".",
            f"Hair: change the hair from {_current_hair_en(facts)} to {axis.hair_en}, "
            f"and change the hair colour from {_current_haircolor_en(facts)} to "
            f"{haircolor.name_zh} ({haircolor.name_en}). {hair.prompt}".strip(),
            f"Makeup: change the makeup from the current light everyday makeup to "
            f"{axis.makeup_en}. {makeup.prompt}".strip(),
            f"Outfit: replace the current {_current_top_en(facts)} with {axis.outfit_en}. "
            f"{outfit.prompt}".strip(),
            f"Accessories: {axis.accessory_en}. {accessory.prompt}".strip(),
            f"Colour palette: {axis.palette_en}.",
        )
        if line
    )

    negative_prompt = "避免" + "、".join((*axis.negatives, *BASE_NEGATIVES))

    return StylingProposal(
        plan_id=axis.axis_id,
        domain_type=domain_type,
        title=axis.name,
        name=axis.name,
        design_axis=axis.axis_id,
        design_direction=axis.direction,
        description=axis.description,
        design_reason=design_reason,
        intensity=axis.intensity,
        hair_plan=hair_plan,
        makeup_plan=makeup_plan,
        outfit_plan=outfit_plan,
        accessory_plan=accessory_plan,
        palette=_palette_with_skin(axis, facts),
        identity_anchors=anchors_zh,
        why_it_fits=why_it_fits,
        expected_changes=expected_changes,
        edit_prompt=edit_prompt,
        negative_prompt=negative_prompt,
        reference_presets=(
            _reference(hair),
            _reference(haircolor),
            _reference(makeup),
            _reference(outfit),
            _reference(accessory),
        ),
        estimated_paid_requests=1,
    )


class PortraitStylist:
    """Designs three complete looks for the person in an analysed portrait."""

    domain_type = "portrait"

    def __init__(self, vocabulary: StyleVocabulary | None = None) -> None:
        self.vocabulary = vocabulary or StyleVocabulary()

    def create_proposals(self, analysis: PhotoAnalysis) -> tuple[StylingProposal, ...]:
        facts = analysis.face_facts or {}
        return tuple(
            _build_proposal(axis, facts, analysis.domain_type, self.vocabulary)
            for axis in AXES
        )
