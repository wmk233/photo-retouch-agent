"""AI personal styling: design complete looks before any image is generated.

There are two clearly separated design modes:

* ``template`` -- the free, deterministic ``_Axis`` composition. Zero model
  calls, zero cost.
* ``ai`` -- a real vision model *authors* the three looks from the person's
  observable features plus the stylekit catalogue. This is one paid styling
  request, made only when the caller explicitly asks for it, and it is never
  triggered as a fallback: if the model fails, the request fails with 502
  instead of quietly returning templates as if they were AI designs.

The full "AI design three looks, then generate one" flow the UI runs is *three*
billable model calls: ``POST /photos/analyze`` first, then
``POST /styling/proposals`` for the design, and finally the ``brain.optimize``
prompt-planning call inside ``POST /api/retouch/jobs`` when the chosen look is
generated. The providers endpoint reports that honestly so the copy cannot
under-count the cost; when the analysis cache still holds the same photo
analysed by the same model, the first call is served from cache and the flow
costs two requests.

DeepSeek-class brains run with ``visionMode == "derived"``: they never receive
the pixels, they design from the locally measured ``faceFacts`` only. The copy
says so rather than implying the model looked at the photo.

Generating a look still goes through ``POST /api/retouch/jobs`` with the chosen
proposal posted as the plan and the *original* image id as ``sourceImageId``.
Designing is free of image generation; nothing here ever calls Image Edit.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, status

from app.api.dependencies import (
    get_agent_brain_factory,
    get_create_styling_proposals_use_case,
    get_domain_registry,
    get_provider_factory,
    get_storage_service,
)
from app.application.use_cases import CreateStylingProposals
from app.brains.factory import AgentBrainFactory
from app.brains.openai_compatible import PromptBrainError
from app.brains.styling_planner import AIStylingPlanner
from app.core.errors import bad_request
from app.domains.portrait.ai_stylist import (
    build_ai_proposals,
    person_identity_anchors,
    preset_catalogue,
    valid_preset_ids,
)
from app.domains.registry import DomainRegistry
from app.infrastructure.storage import StorageService
from app.providers.factory import ImageProviderFactory
from app.schemas.mappers import analysis_to_domain, styling_proposal_to_schema
from app.schemas.styling import StylingProposal, StylingProposalsRequest

router = APIRouter(prefix="/styling", tags=["styling"])

_TEMPLATE_MODE = "template"
_AI_MODE = "ai"
_DESIGN_MODES = (_TEMPLATE_MODE, _AI_MODE)

# Generating one look is exactly one image-edit request with the current
# SiliconFlow Qwen-Image-Edit flow. Kept as one constant so the UI copy and the
# guardrail can never drift apart.
_REQUESTS_PER_GENERATION = 1

# Authoring three looks with the styling planner is deliberately batched into a
# single model call, so the AI design pass costs one request, not three.
_AI_DESIGN_REQUESTS = 1

# The end-to-end "AI design three looks" flow the UI triggers also runs one
# billable analysis first (``POST /photos/analyze``). Uploading the chosen look
# to ``POST /api/retouch/jobs`` then runs the brain's ``optimize`` planning pass
# before the image edit, which is a third real provider request.
_AI_ANALYSIS_REQUESTS = 1
_AI_GENERATION_REQUESTS = 1
_AI_FLOW_REQUESTS = (
    _AI_ANALYSIS_REQUESTS + _AI_DESIGN_REQUESTS + _AI_GENERATION_REQUESTS
)
# With a cached analysis the analysis pass is free, but the design and the
# generation-planning passes still bill.
_AI_REUSED_ANALYSIS_REQUESTS = _AI_DESIGN_REQUESTS + _AI_GENERATION_REQUESTS


@router.get("/providers")
def get_styling_providers(
    image_factory: Annotated[ImageProviderFactory, Depends(get_provider_factory)],
    brain_factory: Annotated[AgentBrainFactory, Depends(get_agent_brain_factory)],
    domains: Annotated[DomainRegistry, Depends(get_domain_registry)],
) -> dict:
    vocabulary = domains.style_vocabulary
    return {
        **image_factory.capabilities(),
        **brain_factory.capabilities(),
        "stylingVocabulary": {
            "available": vocabulary.available,
            "sourceLabel": vocabulary.source_label,
            "loadedFiles": list(vocabulary.loaded_files),
        },
        "stylingDesignModes": {
            _TEMPLATE_MODE: {
                "id": _TEMPLATE_MODE,
                "label": "免费模板预览",
                "isAi": False,
                "paidRequests": 0,
                "costNote": "免费生成，0 次模型调用，不产生任何费用。",
            },
            _AI_MODE: {
                "id": _AI_MODE,
                "label": "AI 个性化设计",
                "isAi": True,
                "analysisRequests": _AI_ANALYSIS_REQUESTS,
                "designRequests": _AI_DESIGN_REQUESTS,
                "generationRequests": _AI_GENERATION_REQUESTS,
                "paidRequests": _AI_FLOW_REQUESTS,
                "reusedAnalysisRequests": _AI_REUSED_ANALYSIS_REQUESTS,
                "costNote": (
                    "由所选模型设计三套方案并出图：完整流程含 1 次人物事实分析 + "
                    f"1 次造型设计 + 1 次出图提示词规划，共 {_AI_FLOW_REQUESTS} 次付费"
                    "模型调用，只在你点击按钮时发起。无图像输入能力的模型（如 "
                    "DeepSeek）不会看到原图，只依据本地程序测量的 FaceFacts 设计。"
                    "若该照片已有同一模型的有效分析缓存，分析会免费复用，"
                    f"本次只计 {_AI_REUSED_ANALYSIS_REQUESTS} 次。"
                ),
            },
        },
        "stylingAiDesignRequests": _AI_DESIGN_REQUESTS,
        "stylingAiAnalysisRequests": _AI_ANALYSIS_REQUESTS,
        "stylingAiGenerationRequests": _AI_GENERATION_REQUESTS,
        "stylingAiFlowRequests": _AI_FLOW_REQUESTS,
        "estimatedRequestsPerGeneration": _REQUESTS_PER_GENERATION,
        "billingNote": (
            "Template proposals are free (no model calls). The full AI flow is "
            f"three paid model calls: 1 analysis (POST /photos/analyze) + 1 styling "
            f"design (POST /styling/proposals) + 1 prompt planning inside "
            f"POST /retouch/jobs = {_AI_FLOW_REQUESTS} requests, and only runs when "
            "you press the design and generate buttons. Brains without image input "
            '(visionMode "derived", e.g. DeepSeek) design from locally measured '
            "FaceFacts and never see the photo. A valid cached analysis for the same "
            "photo and model is reused for free, dropping the flow to "
            f"{_AI_REUSED_ANALYSIS_REQUESTS} requests. Generating one look costs "
            f"{_REQUESTS_PER_GENERATION} image-edit request."
        ),
    }


@router.post(
    "/proposals",
    response_model=list[StylingProposal],
    response_model_by_alias=True,
)
async def create_styling_proposals(
    request: StylingProposalsRequest,
    use_case: Annotated[
        CreateStylingProposals,
        Depends(get_create_styling_proposals_use_case),
    ],
    brain_factory: Annotated[AgentBrainFactory, Depends(get_agent_brain_factory)],
    storage: Annotated[StorageService, Depends(get_storage_service)],
    domains: Annotated[DomainRegistry, Depends(get_domain_registry)],
    agent_provider: Annotated[str | None, Header(alias="X-Agent-Provider")] = None,
    agent_api_key: Annotated[str | None, Header(alias="X-Agent-API-Key")] = None,
    agent_workspace_id: Annotated[
        str | None, Header(alias="X-Agent-Workspace-Id")
    ] = None,
) -> list[StylingProposal]:
    analysis = analysis_to_domain(request.analysis)
    mode = (request.mode or _TEMPLATE_MODE).strip().lower()
    if mode not in _DESIGN_MODES:
        raise bad_request(f"Unsupported styling design mode: {mode}")

    # Styling is portrait-only; this also raises a 400 for other domains in both
    # modes, before any model call is made.
    domains.get_stylist(analysis.domain_type)

    if mode == _TEMPLATE_MODE:
        proposals = use_case.execute(analysis)
    else:
        proposals = await _design_with_ai(
            analysis=analysis,
            request=request,
            brain_factory=brain_factory,
            storage=storage,
            domains=domains,
            agent_provider=agent_provider,
            agent_api_key=agent_api_key,
            agent_workspace_id=agent_workspace_id,
        )
    return [styling_proposal_to_schema(proposal) for proposal in proposals]


async def _design_with_ai(
    *,
    analysis,
    request: StylingProposalsRequest,
    brain_factory: AgentBrainFactory,
    storage: StorageService,
    domains: DomainRegistry,
    agent_provider: str | None,
    agent_api_key: str | None,
    agent_workspace_id: str | None,
):
    brain = brain_factory.create(agent_provider, agent_api_key, agent_workspace_id)
    provider_config = getattr(brain, "provider_config", None)
    if not callable(provider_config):
        raise bad_request(
            "The selected agent brain cannot author AI styling designs. "
            "Choose a vision model provider (e.g. SiliconFlow)."
        )

    vocabulary = domains.style_vocabulary
    facts = analysis.face_facts or {}
    image_path = storage.resolve_image_path(analysis.image_id)
    catalogue = preset_catalogue(vocabulary)

    planner = AIStylingPlanner(**provider_config())
    try:
        payload = await planner.plan(
            image_path=image_path,
            face_facts=facts,
            preset_catalogue=catalogue,
            valid_preset_ids=valid_preset_ids(vocabulary, catalogue),
            identity_anchors=person_identity_anchors(facts),
            user_instruction=request.user_instruction,
            analysis_summary=_analysis_summary(analysis),
        )
        return build_ai_proposals(
            payload,
            facts,
            analysis.domain_type,
            vocabulary,
            getattr(brain, "provider_name", "") or "",
        )
    except (PromptBrainError, RuntimeError) as exc:
        # Never fall back to the free template while claiming an AI design: the
        # caller must be able to tell the design failed.
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"AI styling design failed: {exc}",
        ) from exc


def _analysis_summary(analysis) -> dict:
    return {
        "sceneType": analysis.scene_type,
        "domainType": analysis.domain_type,
        "recommendedStyles": list(analysis.recommended_styles),
        "portraitSuggestions": list(analysis.portrait_suggestions),
        "riskFlags": list(analysis.risk_flags),
        "brainProvider": analysis.brain_provider,
        "visionMode": analysis.vision_mode,
    }
