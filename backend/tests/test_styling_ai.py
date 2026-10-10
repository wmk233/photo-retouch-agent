"""AI styling planner tests.

Everything here runs offline: the styling brain is a real
:class:`AIStylingPlanner` whose only network call is served by an
``httpx.MockTransport``. That means the full production path -- provider
``provider_config`` hand-off, prompt build, JSON parsing, validation, preset
resolution -- is exercised with **zero paid requests**.

The tests pin the guarantees the P2.1 work is about:

* three looks are authored by the model, from the person's observable facts,
  and the free ``_Axis`` templates are *not* silently substituted on failure;
* every chosen preset id traces back to a real stylekit preset;
* the three edit prompts genuinely differ;
* a supplementary request reaches the model exactly once;
* designing never creates an image job.
"""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, Callable

import httpx
import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import get_agent_brain_factory
from app.brains.factory import AgentBrainFactory
from app.brains.styling_planner import AIStylingPlanner
from app.core.config import Settings
from app.domains.portrait.ai_stylist import preset_catalogue
from app.domains.portrait.style_vocabulary import StyleVocabulary
from tests.conftest import make_portrait_bytes

HAIR_LINES = ("change the hair from", "change the hair colour from")
MAKEUP_LINE = "change the makeup from"
OUTFIT_LINE = "replace the current"

# The names the free template composer produces. If any of these ever shows up
# in an AI-mode response, the planner silently fell back to a template.
TEMPLATE_NAMES = ("都市利落短发", "艺术感长发微卷", "先锋街头高马尾")


# ------------------------------------------------------------------ test doubles


class _RecordingTransport:
    """Serves the styling model and records exactly what was sent."""

    def __init__(self, responder: Callable[[dict[str, Any]], Any]) -> None:
        self._responder = responder
        self.raw_bodies: list[str] = []
        self.requests: list[dict[str, Any]] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        raw = request.content.decode("utf-8")
        self.raw_bodies.append(raw)
        body = json.loads(raw)
        self.requests.append(body)
        payload = self._responder(self.user_payload(body))
        if isinstance(payload, str):
            content = payload
        else:
            content = json.dumps(payload, ensure_ascii=False)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": content}}]},
        )

    @staticmethod
    def user_payload(body: dict[str, Any]) -> dict[str, Any]:
        content = body["messages"][1]["content"]
        text = (
            content
            if isinstance(content, str)
            else next(
                part["text"] for part in content if part.get("type") == "text"
            )
        )
        return json.loads(text)


def _install_stub_brain(
    client: TestClient,
    settings: Settings,
    responder: Callable[[dict[str, Any]], Any],
) -> _RecordingTransport:
    """Route the styling brain to an offline transport for this test only."""
    transport = _RecordingTransport(responder)
    real_factory = AgentBrainFactory(settings)

    class _StubFactory:
        def create(self, provider=None, api_key=None, workspace_id=None):
            if str(provider or "").strip().lower() == "siliconflow":
                return AIStylingPlanner(
                    provider_name="siliconflow",
                    api_key="sk-test-not-real",
                    endpoint="http://styling.test/v1/chat/completions",
                    model_name="Qwen/Qwen2.5-VL-72B-Instruct",
                    vision_mode="direct",
                    transport=httpx.MockTransport(transport.handler),
                )
            return real_factory.create(provider, api_key, workspace_id)

        def capabilities(self) -> dict[str, Any]:
            return real_factory.capabilities()

    client.app.dependency_overrides[get_agent_brain_factory] = lambda: _StubFactory()
    return transport


# ----------------------------------------------------------------- look payloads

_ROUND_LOOKS = (
    (
        "soft_wave_glow",
        "柔雾微卷通勤",
        "耳下微卷搭配奶茶棕，弱化脸部横向宽度",
        {
            "hair": "side_part_waves",
            "haircolor": "chocolate_brown",
            "makeup": "korean_dewy",
            "outfit": "vintage_french",
            "accessory": "round_metal_glasses",
        },
    ),
    (
        "clean_bob",
        "清透齐颌短发",
        "齐颌长度配合空气刘海，把视觉重心上移",
        {
            "hair": "eight_bangs_medium",
            "haircolor": "misty_ash_brown",
            "makeup": "clean_nude",
            "outfit": "business_casual",
            "accessory": "pearl_stud_earrings",
        },
    ),
    (
        "evening_pony",
        "宴会低马尾",
        "低马尾配酒红发色，拉长颈部线条",
        {
            "hair": "braided_low_pony",
            "haircolor": "burgundy_brown",
            "makeup": "smoky_evening",
            "outfit": "party_chic_female",
            "accessory": "aviator_sunglasses",
        },
    ),
)

_ANGULAR_LOOKS = (
    (
        "sharp_pixie",
        "利落精灵短发",
        "露出下颌线，突出清晰骨相",
        {
            "hair": "pixie_cut",
            "haircolor": "cool_tea_brown",
            "makeup": "cool_neutral_male",
            "outfit": "techwear",
            "accessory": "black_baseball_cap",
        },
    ),
    (
        "campus_crop",
        "校园感纹理短发",
        "短碎发配冷灰棕，干净利落",
        {
            "hair": "textured_crop",
            "haircolor": "misty_blue_gray",
            "makeup": "clean_nude",
            "outfit": "korean_campus",
            "accessory": "black_square_frames",
        },
    ),
    (
        "sport_high_pony",
        "活力高马尾",
        "高马尾配湖蓝挑染，拉长脸型比例",
        {
            "hair": "high_ponytail_bangs",
            "haircolor": "dreamy_lake_blue",
            "makeup": "peach_sweet",
            "outfit": "athleisure",
            "accessory": "wool_beanie",
        },
    ),
)


def _look(index: int, spec: tuple) -> dict[str, Any]:
    slug, name, direction, presets = spec
    return {
        "slug": slug,
        "name": name,
        "designDirection": direction,
        "designReason": f"依据照片中可观察到的外形特征：{direction}",
        "whyItFits": f"这套方案贴合照片里这个人的可见特征：{direction}",
        "intensity": "medium",
        "hair": {
            "length": "耳下到下颌之间",
            "structure": "自然微卷、后脑饱满",
            "bangs": "空气感法式刘海",
            "layers": "高层次打薄",
            "color": "奶茶棕",
        },
        "makeup": {
            "brows": "柔和自然眉",
            "eyes": "大地色晕染",
            "lips": "豆沙色唇釉",
            "base": "清透水光底妆",
        },
        "outfit": {
            "style": "结构感外套",
            "material": "羊毛混纺",
            "color": "燕麦色",
            "pairing": "内搭高领针织",
        },
        "accessory": "细金属圆框眼镜",
        "palette": "燕麦色 × 奶茶棕",
        "identityAnchors": ["保留脸型轮廓", "保留原本肤色基调"],
        "editPrompt": f"针对这套「{name}」方案：发型换成{direction}，妆容改为眼妆与唇妆的完整方案，"
        f"服装替换为结构感外套，保留本人五官比例。（第 {index} 套独立指令）",
        "negativePrompt": ["不要更换成另一个人", "不要改变人物数量"],
        "selectedPresets": {
            "hair": presets["hair"],
            "haircolor": presets["haircolor"],
            "makeup": presets["makeup"],
            "outfit": presets["outfit"],
            "accessory": presets["accessory"],
        },
    }


def _payload_for(facts: dict[str, Any]) -> dict[str, Any]:
    """Mimic a model that designs differently for different visible features."""
    # ``face_shape`` is the canonical domain key (``FaceFacts.model_dump()``);
    # the planner hands the measurements over verbatim.
    is_round = "round" in str(facts.get("face_shape") or "").lower()
    specs = _ROUND_LOOKS if is_round else _ANGULAR_LOOKS
    return {"proposals": [_look(index, spec) for index, spec in enumerate(specs, 1)]}


def _designer(facts_probe: dict[str, Any] | None = None):
    def responder(user: dict[str, Any]) -> Any:
        facts = user.get("faceFacts") or {}
        if facts_probe is not None:
            facts_probe.update(user)
            facts_probe["__facts__"] = facts
        return _payload_for(facts)

    return responder


# ---------------------------------------------------------------------- helpers


def _portrait_analysis(client: TestClient) -> tuple[str, dict[str, Any]]:
    upload = client.post(
        "/api/photos/upload",
        files={"file": ("portrait.png", make_portrait_bytes(), "image/png")},
    )
    assert upload.status_code == 200
    image_id = upload.json()["imageId"]
    analysis = client.post("/api/photos/analyze", json={"imageId": image_id})
    assert analysis.status_code == 200
    body = analysis.json()
    assert body["domainType"] == "portrait"
    assert body["faceFacts"] is not None
    return image_id, body


def _design(
    client: TestClient,
    analysis: dict[str, Any],
    *,
    mode: str = "ai",
    user_instruction: str = "",
    provider: str = "siliconflow",
):
    return client.post(
        "/api/styling/proposals",
        json={
            "analysis": analysis,
            "mode": mode,
            "userInstruction": user_instruction,
        },
        headers={
            "X-Agent-Provider": provider,
            "X-Agent-API-Key": "sk-test-not-real",
        },
    )


def _with_facts(analysis: dict[str, Any], **facts: Any) -> dict[str, Any]:
    clone = deepcopy(analysis)
    clone["faceFacts"].update(facts)
    return clone


# ------------------------------------------------------------------------ tests


def test_ai_mode_authors_three_distinct_looks(client: TestClient, temp_settings: Settings) -> None:
    _install_stub_brain(client, temp_settings, _designer())
    _, analysis = _portrait_analysis(client)

    response = _design(client, analysis)
    assert response.status_code == 200, response.text
    proposals = response.json()

    assert len(proposals) == 3
    assert len({p["planId"] for p in proposals}) == 3
    assert len({p["editPrompt"] for p in proposals}) == 3, "prompts must differ"

    for proposal in proposals:
        # Provenance: this is an AI design, not a template preview.
        assert proposal["designMode"] == "ai"
        assert proposal["sourceProvider"] == "siliconflow"
        assert proposal["estimatedPaidRequests"] == 1
        assert len(proposal["selectedPresets"]) == 5

        # Every look is fully specified.
        assert proposal["hairPlan"] and "长度" in proposal["hairPlan"]
        assert "发色" in proposal["hairPlan"]
        for key in ("眉毛", "眼妆", "唇妆", "底妆"):
            assert key in proposal["makeupPlan"]
        for key in ("款式", "材质", "颜色", "搭配"):
            assert key in proposal["outfitPlan"]
        assert proposal["accessoryPlan"]
        assert proposal["palette"]
        assert proposal["identityAnchors"]
        assert proposal["whyItFits"]

        # The edit prompt describes concrete change, not generic beautification.
        for marker in (*HAIR_LINES, MAKEUP_LINE, OUTFIT_LINE):
            assert marker in proposal["editPrompt"]
        assert "no absolute guarantee" in proposal["editPrompt"]
        assert "不要更换成另一个人" not in proposal["editPrompt"]

        assert proposal["name"] not in TEMPLATE_NAMES

    # Designing must not generate images.
    assert client.get("/api/retouch/jobs").json() == []


def test_ai_mode_passes_person_facts_and_the_stylekit_catalogue_to_the_model(
    client: TestClient, temp_settings: Settings
) -> None:
    probe: dict[str, Any] = {}
    transport = _install_stub_brain(client, temp_settings, _designer(probe))
    _, analysis = _portrait_analysis(client)

    response = _design(client, analysis)
    assert response.status_code == 200, response.text

    assert len(transport.requests) == 1, "the design must use exactly one model request"
    facts = probe["__facts__"]
    assert facts["face_shape"] == analysis["faceFacts"]["faceShape"]
    assert facts["hair_color_name"] == analysis["faceFacts"]["hairColorName"]

    # The identity anchors the local measurement pinned must ride along.
    assert probe["identityAnchorsToKeep"]

    catalogue = probe["stylePresetCatalogue"]
    assert set(catalogue) == {"hairstyle", "haircolor", "makeup", "outfit", "accessory"}
    offered = {entry["presetId"] for entries in catalogue.values() for entry in entries}
    assert "pixie_cut" in offered and "chocolate_brown" in offered
    assert "vintage_french" in offered and "pearl_stud_earrings" in offered

    # The offered ids are exactly the real stylekit vocabulary.
    real = preset_catalogue(StyleVocabulary())
    real_ids = {entry["presetId"] for entries in real.values() for entry in entries}
    assert offered == real_ids


def test_every_chosen_preset_traces_to_a_real_stylekit_preset(
    client: TestClient, temp_settings: Settings
) -> None:
    _install_stub_brain(client, temp_settings, _designer())
    _, analysis = _portrait_analysis(client)
    proposals = _design(client, analysis).json()

    vocabulary = StyleVocabulary()
    for proposal in proposals:
        references = {ref["bucket"]: ref for ref in proposal["referencePresets"]}
        assert set(references) >= {
            "hairstyle_female",
            "haircolor",
            "makeup",
            "outfit",
            "accessory",
        }
        for reference in proposal["referencePresets"]:
            assert reference["sourceFile"].endswith(".yaml")
            assert ":" in reference["vocabularyRef"]

        hair = vocabulary.find_hair(proposal["selectedPresets"][0])
        assert hair is not None, "AI picked a hairstyle that is not in stylekit"
        assert proposal["selectedPresets"][0] in proposal["editPrompt"]


def test_different_visible_features_yield_different_designs(
    client: TestClient, temp_settings: Settings
) -> None:
    _install_stub_brain(client, temp_settings, _designer())
    _, analysis = _portrait_analysis(client)

    # The API payload is camelCase; the server converts it to the canonical
    # snake_case ``FaceFacts`` dump before handing it to the planner.
    round_face = _with_facts(analysis, faceShape="round")
    angular_face = _with_facts(analysis, faceShape="square")

    round_body = _design(client, round_face)
    angular_body = _design(client, angular_face)
    assert round_body.status_code == 200, round_body.text
    assert angular_body.status_code == 200, angular_body.text

    first = round_body.json()
    second = angular_body.json()

    assert [p["name"] for p in first] != [p["name"] for p in second]
    assert [p["selectedPresets"] for p in first] != [
        p["selectedPresets"] for p in second
    ]
    assert [p["editPrompt"] for p in first] != [p["editPrompt"] for p in second]
    for proposal in first:
        assert proposal["planId"].startswith("ai_")


def test_user_instruction_reaches_the_model_exactly_once(
    client: TestClient, temp_settings: Settings
) -> None:
    transport = _install_stub_brain(client, temp_settings, _designer())
    _, analysis = _portrait_analysis(client)

    response = _design(client, analysis, user_instruction="想要更干练")
    assert response.status_code == 200, response.text

    assert len(transport.raw_bodies) == 1
    raw = transport.raw_bodies[0]
    assert raw.count("想要更干练") == 1, "the request text must not be duplicated"
    assert transport.requests[0]
    assert (
        _RecordingTransport.user_payload(transport.requests[0])["userRequest"]
        == "想要更干练"
    )


def test_invalid_model_output_fails_loudly_without_template_fallback(
    client: TestClient, temp_settings: Settings
) -> None:
    _install_stub_brain(client, temp_settings, lambda _user: "抱歉，我无法完成这个请求。")
    _, analysis = _portrait_analysis(client)

    response = _design(client, analysis)
    assert response.status_code == 502
    payload = response.json()
    assert "AI styling design failed" in payload["detail"]
    # No designs, and certainly not the free templates masquerading as AI output.
    assert "planId" not in json.dumps(payload)
    for name in TEMPLATE_NAMES:
        assert name not in json.dumps(payload)
    assert client.get("/api/retouch/jobs").json() == []


def test_invented_or_duplicate_presets_fail_loudly(
    client: TestClient, temp_settings: Settings
) -> None:
    invented = _payload_for({"face_shape": "round"})
    invented["proposals"][1]["selectedPresets"]["hair"] = "totally_made_up_preset"
    _install_stub_brain(client, temp_settings, lambda _user: invented)
    _, analysis = _portrait_analysis(client)
    response = _design(client, analysis)
    assert response.status_code == 502
    assert "totally_made_up_preset" in response.json()["detail"]

    duplicate = _payload_for({"face_shape": "round"})
    for proposal in duplicate["proposals"]:
        proposal["selectedPresets"] = dict(duplicate["proposals"][0]["selectedPresets"])
        proposal["name"] = "同一种造型"
    client.app.dependency_overrides.pop(get_agent_brain_factory)
    _install_stub_brain(client, temp_settings, lambda _user: duplicate)
    response = _design(client, analysis)
    assert response.status_code == 502
    assert client.get("/api/retouch/jobs").json() == []


def test_ai_mode_requires_a_vision_capable_brain(
    client: TestClient, temp_settings: Settings
) -> None:
    _install_stub_brain(client, temp_settings, _designer())
    _, analysis = _portrait_analysis(client)

    # The local brain cannot author designs; it must be refused, not faked.
    response = _design(client, analysis, provider="local")
    assert response.status_code == 400
    assert "vision model" in response.json()["detail"]


def test_unknown_design_mode_is_rejected(client: TestClient) -> None:
    _, analysis = _portrait_analysis(client)
    response = _design(client, analysis, mode="magic")
    assert response.status_code == 400
    assert "magic" in response.json()["detail"]


def test_template_mode_stays_free_and_never_calls_the_model(
    client: TestClient, temp_settings: Settings
) -> None:
    transport = _install_stub_brain(client, temp_settings, _designer())
    _, analysis = _portrait_analysis(client)

    response = _design(client, analysis, mode="template")
    assert response.status_code == 200, response.text
    proposals = response.json()

    assert transport.requests == [], "the free preview must not call the model"
    assert len(proposals) == 3
    for proposal in proposals:
        assert proposal["designMode"] == "template"
        assert proposal["sourceProvider"] == ""
        assert proposal["selectedPresets"] == []
    assert client.get("/api/retouch/jobs").json() == []


def test_generating_one_ai_look_creates_exactly_one_job_from_the_original(
    client: TestClient, temp_settings: Settings
) -> None:
    _install_stub_brain(client, temp_settings, _designer())
    image_id, analysis = _portrait_analysis(client)
    proposal = _design(client, analysis).json()[1]

    created = client.post(
        "/api/retouch/jobs",
        json={"sourceImageId": image_id, "plan": proposal, "userInstruction": ""},
    )
    assert created.status_code == 202

    job = created.json()
    deadline_body = client.get(f"/api/retouch/jobs/{job['jobId']}").json()
    assert deadline_body["sourceImageId"] == image_id
    assert deadline_body["planId"] == proposal["planId"]
    # The other two looks were never generated.
    assert len(client.get("/api/retouch/jobs").json()) == 1


def test_ai_prompt_keeps_identity_anchors(client: TestClient, temp_settings: Settings) -> None:
    _install_stub_brain(client, temp_settings, _designer())
    _, analysis = _portrait_analysis(client)
    proposals = _design(client, analysis).json()

    for proposal in proposals:
        prompt = proposal["editPrompt"]
        assert proposal["identityAnchors"]
        assert "保留" in "".join(proposal["identityAnchors"])
        assert "keep the same person" in prompt
        assert "no absolute guarantee" in prompt
