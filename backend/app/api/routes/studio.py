"""The personal studio endpoints: quote first, spend second.

Every render is a real, billed SiliconFlow request, so the flow is split in two
on purpose. ``POST /studio/quote`` says exactly how many paid calls a request
would make and never touches the network; ``POST /studio/render`` refuses to run
unless the caller echoes back ``confirmPaid``. Typing a prompt can therefore
never bill anything, and the number the confirmation dialog shows is produced by
the same code path that issues the calls.

``POST /studio/interrogate`` (reverse-prompt a photo into a prompt) is a paid
call of its own and follows the same split, under its own ``/interrogate``
prefix: ``/studio/interrogate/quote`` prices it, ``/studio/interrogate`` spends
it, and neither path ever bundles the two paid calls together.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query

from app.api.dependencies import (
    get_studio_interrogate_service,
    get_studio_service,
)
from app.schemas.studio import (
    StudioInterrogateQuote,
    StudioInterrogateRequestBody,
    StudioInterrogateResult,
    StudioQuote,
    StudioRenderRequestBody,
    StudioRenderResult,
    StudioStatus,
)
from app.services.studio_interrogate import (
    StudioInterrogateRequest,
    StudioInterrogateService,
)
from app.services.studio_service import StudioRenderRequest, StudioService

router = APIRouter(prefix="/studio", tags=["studio"])


def _to_service_request(body: StudioRenderRequestBody) -> StudioRenderRequest:
    return StudioRenderRequest(
        prompt=body.prompt,
        mode=body.mode,
        model_id=body.model_id,
        source_image_id=body.source_image_id,
        image_size=body.image_size,
        negative_prompt=body.negative_prompt,
        count=body.count,
        confirm_paid=body.confirm_paid,
        seeds=tuple(body.seeds),
    )


@router.get("/status", response_model=StudioStatus, response_model_by_alias=True)
async def studio_status(
    service: Annotated[StudioService, Depends(get_studio_service)],
) -> StudioStatus:
    return StudioStatus.model_validate(service.status())


@router.post("/quote", response_model=StudioQuote, response_model_by_alias=True)
async def studio_quote(
    body: StudioRenderRequestBody,
    service: Annotated[StudioService, Depends(get_studio_service)],
) -> StudioQuote:
    return StudioQuote.model_validate(service.quote(_to_service_request(body)))


@router.post("/render", response_model=StudioRenderResult, response_model_by_alias=True)
async def studio_render(
    body: StudioRenderRequestBody,
    service: Annotated[StudioService, Depends(get_studio_service)],
) -> StudioRenderResult:
    result = await service.render(_to_service_request(body))
    return StudioRenderResult.model_validate(result)


def _to_interrogate_request(
    body: StudioInterrogateRequestBody,
) -> StudioInterrogateRequest:
    return StudioInterrogateRequest(
        source_image_id=body.source_image_id,
        model_id=body.model_id,
        focus=body.focus,
        hint=body.hint,
        confirm_paid=body.confirm_paid,
    )


@router.post(
    "/interrogate/quote",
    response_model=StudioInterrogateQuote,
    response_model_by_alias=True,
)
async def studio_interrogate_quote(
    body: StudioInterrogateRequestBody,
    service: Annotated[
        StudioInterrogateService, Depends(get_studio_interrogate_service)
    ],
) -> StudioInterrogateQuote:
    return StudioInterrogateQuote.model_validate(
        service.quote(_to_interrogate_request(body))
    )


@router.post(
    "/interrogate",
    response_model=StudioInterrogateResult,
    response_model_by_alias=True,
)
async def studio_interrogate(
    body: StudioInterrogateRequestBody,
    service: Annotated[
        StudioInterrogateService, Depends(get_studio_interrogate_service)
    ],
) -> StudioInterrogateResult:
    result = await service.interrogate(_to_interrogate_request(body))
    return StudioInterrogateResult.model_validate(result)


@router.get("/history")
async def studio_history(
    service: Annotated[StudioService, Depends(get_studio_service)],
    limit: Annotated[int, Query(ge=1, le=96)] = 24,
) -> list[dict]:
    return service.history(limit=limit)
