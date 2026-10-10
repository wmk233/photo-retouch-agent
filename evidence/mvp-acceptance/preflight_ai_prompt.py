"""Zero-cost preflight: does the AI-designed plan survive RunningHub's gate?

Nothing here calls a paid model. It:
  1. uploads IMG_3651.jpg and analyses it with the *local* (free) brain,
  2. composes three AI proposals through the exact same
     ``build_ai_proposals`` composer the real flow uses, from a payload shaped
     like the response contract DeepSeek must fill,
  3. runs ``runninghub_crop_stitch.build_prompts`` on each plan, the same call
     the RunningHub provider makes before it ever contacts the service.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "backend"))

import httpx  # noqa: E402

from app.domains.portrait.ai_stylist import (  # noqa: E402
    build_ai_proposals,
    preset_catalogue,
    valid_preset_ids,
)
from app.domains.portrait.style_vocabulary import StyleVocabulary  # noqa: E402
from app.infrastructure import runninghub_crop_stitch as rh  # noqa: E402
from app.core.config import settings  # noqa: E402

API = "http://127.0.0.1:8000/api"
PHOTO = REPO / "data" / "inputs" / "IMG_3651.jpg"

vocabulary = StyleVocabulary()
catalogue = preset_catalogue(vocabulary)
valid = valid_preset_ids(vocabulary, catalogue)

print("== catalogue sizes ==")
for bucket, entries in catalogue.items():
    print(f"  {bucket}: {len(entries)}")

with httpx.Client(timeout=180.0) as client:
    with PHOTO.open("rb") as handle:
        up = client.post(
            f"{API}/photos/upload",
            files={"file": (PHOTO.name, handle, "image/jpeg")},
        )
    up.raise_for_status()
    image_id = up.json()["imageId"]
    print("== uploaded ==", image_id)

    an = client.post(
        f"{API}/photos/analyze",
        json={"imageId": image_id},
        headers={"X-Agent-Provider": "local", "X-Action-Provider": "mock"},
    )
    an.raise_for_status()
    analysis = an.json()

facts = analysis.get("faceFacts") or analysis.get("face_facts") or {}
print("== faceFacts keys ==", sorted(facts.keys()))


def pick(bucket: str, index: int) -> str:
    return str(catalogue[bucket][index % len(catalogue[bucket])]["presetId"])


# A payload shaped like the contract DeepSeek must fill. Deliberately hostile:
# Chinese field prose, a multi-line free-form editPrompt, and an English
# "face-framing" phrase inside the hair description so the forbid-guard is
# actually exercised.
STRESS_EDIT_PROMPT = (
    "换一个更蓬松的中长发型，增加头发的体积感。\n"
    "Hair: change the hair from the current to a voluminous wavy bob with soft face-framing layers\n"
    "Clothing: replace the current top with a cream silk blouse\n"
    "避免贴头皮、避免塑料感。"
)

PAYLOAD = {
    "proposals": [
        {
            "slug": "preflight_look_a",
            "name": f"预检造型A",
            "description": "预检用方案A",
            "designDirection": "蓬松长卷 · 暖调发色 · 法式复古",
            "designReason": "预检用，验证组合后的出图提示词能通过 RunningHub 前置校验。",
            "whyItFits": "预检用。",
            "intensity": "medium",
            "hair": {
                "length": "过肩长发",
                "structure": "中段起做蓬松大卷，发尾自然外翻",
                "bangs": "法式八字长刘海",
                "layers": "中高层次，增加体积感",
                "color": "暖栗棕",
            },
            "makeup": {"brows": "自然眉", "eyes": "蜜桃眼影", "lips": "水润唇", "base": "水光底妆"},
            "outfit": {"style": "法式衬衫", "material": "真丝", "color": "奶白", "pairing": "细腰带收腰"},
            "accessory": "金色圈形耳环",
            "palette": "奶白 + 酒红 + 复古金",
            "identityAnchors": ["圆润脸型轮廓", "健康小麦肤色基调", "原图眼镜"],
            "editPrompt": STRESS_EDIT_PROMPT,
            "negativePrompt": ["贴头皮直发", "塑料感磨皮"],
            "selectedPresets": {
                "hair": pick("hairstyle", 3),
                "haircolor": pick("haircolor", 1),
                "makeup": pick("makeup", 2),
                "outfit": pick("outfit", 4),
                "accessory": pick("accessory", 0),
            },
        },
        {
            "slug": "preflight_look_b",
            "name": f"预检造型B",
            "description": "预检用方案B",
            "designDirection": "利落短发 · 冷调发色 · 极简",
            "designReason": "预检用。",
            "whyItFits": "预检用。",
            "intensity": "bold",
            "hair": {"length": "齐耳短发", "structure": "利落直线", "bangs": "侧分", "layers": "低层次", "color": "冷棕"},
            "makeup": {"brows": "利落眉", "eyes": "大地色", "lips": "裸色", "base": "哑光"},
            "outfit": {"style": "西装外套", "material": "羊毛", "color": "深灰", "pairing": "直筒裤"},
            "accessory": "银色耳钉",
            "palette": "深灰 + 冷棕",
            "identityAnchors": ["圆润脸型轮廓", "原图眼镜"],
            "editPrompt": "改成利落短发。Hair: change the hair from the current to a sleek chin-length bob. Outfit: replace the current top with a charcoal wool blazer.",
            "negativePrompt": ["油腻"],
            "selectedPresets": {
                "hair": pick("hairstyle", 7),
                "haircolor": pick("haircolor", 3),
                "makeup": pick("makeup", 5),
                "outfit": pick("outfit", 8),
                "accessory": pick("accessory", 2),
            },
        },
        {
            "slug": "preflight_look_c",
            "name": f"预检造型C",
            "description": "预检用方案C",
            "designDirection": "编发 · 亮色挑染 · 波西米亚",
            "designReason": "预检用。",
            "whyItFits": "预检用。",
            "intensity": "bold",
            "hair": {"length": "长发", "structure": "松散编发", "bangs": "中分", "layers": "高层次", "color": "亚麻挑染"},
            "makeup": {"brows": "柔和眉", "eyes": "暖棕眼影", "lips": "砖红唇", "base": "自然"},
            "outfit": {"style": "针织长裙", "material": "棉麻", "color": "燕麦色", "pairing": "宽檐帽"},
            "accessory": "木质耳环",
            "palette": "燕麦 + 亚麻",
            "identityAnchors": ["健康小麦肤色基调", "原图眼镜"],
            "editPrompt": "改成松散编发，加亚麻挑染。",
            "negativePrompt": ["呆板"],
            "selectedPresets": {
                "hair": pick("hairstyle", 11),
                "haircolor": pick("haircolor", 5),
                "makeup": pick("makeup", 7),
                "outfit": pick("outfit", 12),
                "accessory": pick("accessory", 4),
            },
        },
    ]
}

proposals = build_ai_proposals(PAYLOAD, facts, analysis.get("domainType") or "portrait", vocabulary, "deepseek")
print("== composed", len(proposals), "AI proposals ==")

report = {"imageId": image_id, "proposals": [], "passed": True}
for proposal in proposals:
    entry = {"planId": proposal.plan_id, "title": proposal.title}
    entry["editPrompt"] = proposal.edit_prompt
    bundle = {
        "style_id": proposal.plan_id,
        "style_name": proposal.title,
        "hair_prompt": proposal.edit_prompt,
        "clothing_prompt": proposal.edit_prompt,
        "identity_constraints": {"rules": [{"id": rh.MASK_RULE_ID, "en": rh.DEFAULT_MASK_RULE}]},
    }
    try:
        prompts = rh.build_prompts(bundle, expected_style_id=proposal.plan_id)
        entry["ok"] = True
        entry["hairPrompt"] = prompts["hair"]
        entry["clothingPrompt"] = prompts["clothing"]
        entry["dropped"] = {
            "clothing_forbidden": prompts["clothing_forbidden_clauses_dropped"],
            "hair_forbidden": prompts["hair_forbidden_clauses_dropped"],
        }
    except SystemExit as exc:
        entry["ok"] = False
        entry["refusal"] = str(exc)
        report["passed"] = False
    report["proposals"].append(entry)

out = Path(__file__).with_name("preflight_ai_prompt.json")
out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
print("== RESULT ==", "PASS" if report["passed"] else "FAIL")
for entry in report["proposals"]:
    print(f"  {entry['planId']}: {'ok' if entry['ok'] else 'REFUSED: ' + entry.get('refusal','')}")
print("wrote", out)
