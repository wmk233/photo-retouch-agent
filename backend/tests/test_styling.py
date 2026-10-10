from __future__ import annotations

import time

from fastapi.testclient import TestClient

from app.core.config import Settings
from app.domain.analysis import PhotoAnalysis, SubjectSummary
from app.domains.portrait.stylist import PortraitStylist
from app.domains.portrait.style_vocabulary import StyleVocabulary
from tests.conftest import make_landscape_bytes, make_portrait_bytes

HAIR_LINES = ("change the hair from", "change the hair colour from")
MAKEUP_LINE = "change the makeup from"
OUTFIT_LINE = "replace the current"


def _portrait_analysis(client: TestClient) -> tuple[str, dict]:
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


def _proposals(client: TestClient, analysis: dict) -> list[dict]:
    response = client.post("/api/styling/proposals", json={"analysis": analysis})
    assert response.status_code == 200
    return response.json()


def _wait_for_job(client: TestClient, job_id: str, timeout: float = 5.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = client.get(f"/api/retouch/jobs/{job_id}").json()
        if job["status"] in ("succeeded", "failed"):
            return job
        time.sleep(0.1)
    raise TimeoutError(f"Job {job_id} did not complete within {timeout}s")


# ------------------------------------------------------------------ proposals


def test_proposals_returns_three_distinct_looks(client: TestClient) -> None:
    _, analysis = _portrait_analysis(client)
    proposals = _proposals(client, analysis)
    assert len(proposals) == 3
    assert len({p["planId"] for p in proposals}) == 3
    assert len({p["name"] for p in proposals}) == 3
    assert len({p["designAxis"] for p in proposals}) == 3
    for proposal in proposals:
        assert proposal["domainType"] == "portrait"
        assert proposal["estimatedPaidRequests"] == 1


def test_each_proposal_covers_hair_makeup_outfit_and_identity(client: TestClient) -> None:
    _, analysis = _portrait_analysis(client)
    for proposal in _proposals(client, analysis):
        assert proposal["hairPlan"]
        assert "长度" in proposal["hairPlan"]
        assert "结构" in proposal["hairPlan"]
        assert "刘海" in proposal["hairPlan"]
        assert "层次" in proposal["hairPlan"]
        assert "发色" in proposal["hairPlan"]

        assert proposal["makeupPlan"]
        for key in ("眉毛", "眼妆", "唇妆", "底妆"):
            assert key in proposal["makeupPlan"]

        assert proposal["outfitPlan"]
        for key in ("款式", "材质", "颜色", "搭配"):
            assert key in proposal["outfitPlan"]

        assert proposal["accessoryPlan"]
        assert proposal["palette"]
        assert proposal["identityAnchors"]
        assert proposal["whyItFits"]
        assert len(proposal["expectedChanges"]) >= 4


def test_edit_prompts_are_distinct_and_concrete(client: TestClient) -> None:
    _, analysis = _portrait_analysis(client)
    proposals = _proposals(client, analysis)

    prompts = [p["editPrompt"] for p in proposals]
    assert len(set(prompts)) == 3, "the three looks must not share one edit prompt"

    for proposal in proposals:
        prompt = proposal["editPrompt"]
        # The prompt must describe a concrete transformation, not a generic
        # "beautify the person" instruction.
        for marker in (*HAIR_LINES, MAKEUP_LINE, OUTFIT_LINE):
            assert marker in prompt
        assert "no absolute guarantee" in prompt

        references = {ref["bucket"]: ref for ref in proposal["referencePresets"]}
        assert set(references) >= {
            "hairstyle_female",
            "haircolor",
            "makeup",
            "outfit",
            "accessory",
        }
        # The exact stylekit hair colour / makeup / outfit that this axis picked
        # must be named in the copy, proving the vocabulary drives the design.
        assert references["haircolor"]["name"] in prompt
        assert references["makeup"]["name"] in proposal["makeupPlan"]
        assert references["outfit"]["name"] in proposal["outfitPlan"]

    # Different axes must not collapse onto the same hair colour / outfit.
    hair_colours = {
        ref["presetId"]
        for proposal in proposals
        for ref in proposal["referencePresets"]
        if ref["bucket"] == "haircolor"
    }
    assert len(hair_colours) == 3


def test_reference_presets_trace_to_real_stylekit(client: TestClient) -> None:
    _, analysis = _portrait_analysis(client)
    for proposal in _proposals(client, analysis):
        assert proposal["referencePresets"]
        for reference in proposal["referencePresets"]:
            assert reference["sourceFile"].endswith(".yaml")
            assert ":" in reference["vocabularyRef"]


def test_proposals_are_free_and_leave_no_jobs(
    client: TestClient,
    temp_settings: Settings,
) -> None:
    _, analysis = _portrait_analysis(client)
    _proposals(client, analysis)
    assert client.get("/api/retouch/jobs").json() == []
    outputs_dir = temp_settings.outputs_dir
    assert not outputs_dir.exists() or not list(outputs_dir.glob("*"))


def test_proposals_reject_non_portrait_analysis(client: TestClient) -> None:
    upload = client.post(
        "/api/photos/upload",
        files={"file": ("scene.png", make_landscape_bytes(), "image/png")},
    )
    image_id = upload.json()["imageId"]
    analysis = client.post(
        "/api/photos/analyze",
        json={"imageId": image_id, "domainType": "general"},
    ).json()
    response = client.post("/api/styling/proposals", json={"analysis": analysis})
    assert response.status_code == 400
    assert "portrait" in response.json()["detail"]


# ------------------------------------------------------------------- providers


def test_styling_providers_report_vocabulary_and_guardrails(client: TestClient) -> None:
    response = client.get("/api/styling/providers")
    assert response.status_code == 200
    body = response.json()

    vocabulary = body["stylingVocabulary"]
    assert vocabulary["available"] is True
    assert "stylekit" in vocabulary["sourceLabel"]
    assert "hairstyles_female.yaml" in vocabulary["loadedFiles"]
    assert "haircolors.yaml" in vocabulary["loadedFiles"]

    assert body["estimatedRequestsPerGeneration"] == 1
    assert "free" in body["billingNote"]


# ------------------------------------------------------------ generate one look


def test_generating_one_look_uses_original_image_and_single_job(
    client: TestClient,
    temp_settings: Settings,
) -> None:
    image_id, analysis = _portrait_analysis(client)
    proposal = _proposals(client, analysis)[1]

    created = client.post(
        "/api/retouch/jobs",
        json={"sourceImageId": image_id, "plan": proposal, "userInstruction": ""},
    )
    assert created.status_code == 202

    job = _wait_for_job(client, created.json()["jobId"])
    assert job["status"] == "succeeded"
    assert job["sourceImageId"] == image_id
    assert job["planId"] == proposal["planId"]
    assert len(job["outputUrls"]) == 1

    # Only the single chosen look was generated; the other two were not.
    jobs = client.get("/api/retouch/jobs").json()
    assert len(jobs) == 1

    output_path = temp_settings.outputs_dir / f"{job['outputImageIds'][0]}.jpg"
    assert output_path.exists()


def test_refine_reuses_generated_output_as_base(client: TestClient) -> None:
    image_id, analysis = _portrait_analysis(client)
    proposal = _proposals(client, analysis)[0]
    created = client.post(
        "/api/retouch/jobs",
        json={"sourceImageId": image_id, "plan": proposal, "userInstruction": ""},
    ).json()
    parent = _wait_for_job(client, created["jobId"])

    refine = client.post(
        f"/api/retouch/jobs/{parent['jobId']}/refine",
        json={"userInstruction": "头发再短一点，衣服换成皮衣"},
    )
    assert refine.status_code == 202
    child = _wait_for_job(client, refine.json()["jobId"])

    assert child["status"] == "succeeded"
    assert child["jobId"] != parent["jobId"]
    assert child["sourceImageId"] == image_id
    assert child["baseImageId"] == parent["outputImageIds"][-1]
    assert child["userInstruction"] == "头发再短一点，衣服换成皮衣"
    assert len(child["outputUrls"]) == 1


# --------------------------------------------------------------- stylist unit


def _sparse_analysis(face_facts: dict | None) -> PhotoAnalysis:
    return PhotoAnalysis(
        image_id="img_unit",
        domain_type="portrait",
        scene_type="portrait",
        subjects=SubjectSummary(count=1, position="center", face_visibility="clear"),
        face_facts=face_facts,
    )


def test_stylist_falls_back_without_stylekit_or_facts() -> None:
    stylist = PortraitStylist(StyleVocabulary(presets_dir="/definitely/not/here"))
    proposals = stylist.create_proposals(_sparse_analysis(None))
    assert len(proposals) == 3
    assert len({p.plan_id for p in proposals}) == 3
    assert len({p.edit_prompt for p in proposals}) == 3
    for proposal in proposals:
        assert proposal.identity_anchors
        assert proposal.reference_presets
        assert proposal.hair_plan and proposal.makeup_plan and proposal.outfit_plan
        assert "change the hair from" in proposal.edit_prompt
        assert "replace the current" in proposal.edit_prompt
        assert "no absolute guarantee" in proposal.edit_prompt
