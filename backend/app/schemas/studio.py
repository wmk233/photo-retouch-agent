from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class StudioRenderRequestBody(BaseModel):
    """One studio render request.

    ``confirmPaid`` exists so the server can refuse to spend money on a caller
    that never saw a price. The UI always quotes first, shows the number, and
    only then sends this with the flag set.
    """

    model_config = ConfigDict(populate_by_name=True)

    prompt: str
    mode: Literal["edit", "text2image"] = "edit"
    model_id: str | None = Field(default=None, alias="modelId")
    source_image_id: str | None = Field(default=None, alias="sourceImageId")
    image_size: str | None = Field(default=None, alias="imageSize")
    negative_prompt: str = Field(default="", alias="negativePrompt")
    count: int = 1
    confirm_paid: bool = Field(default=False, alias="confirmPaid")
    seeds: list[int] = Field(default_factory=list)


class StudioQuote(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    model: str
    model_label: str = Field(alias="modelLabel")
    mode: str
    count: int
    paid_requests: int = Field(alias="paidRequests")
    image_size: str | None = Field(default=None, alias="imageSize")
    source_image_id: str | None = Field(default=None, alias="sourceImageId")
    requires_confirmation: bool = Field(default=True, alias="requiresConfirmation")
    note: str = ""


class StudioRenderItem(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    image_id: str = Field(alias="imageId")
    url: str
    width: int
    height: int
    bytes: int
    model: str
    model_label: str = Field(alias="modelLabel")
    mode: str
    seed: int | None = None
    prompt: str = ""
    negative_prompt: str = Field(default="", alias="negativePrompt")
    source_image_id: str | None = Field(default=None, alias="sourceImageId")
    request_id: str | None = Field(default=None, alias="requestId")
    generated_at: str | None = Field(default=None, alias="generatedAt")
    index: int = 0


class StudioStatus(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    configured: bool
    provider: str
    default_model: str = Field(alias="defaultModel")
    default_edit_model: str = Field(alias="defaultEditModel")
    default_text_model: str = Field(alias="defaultTextModel")
    models: list[dict[str, Any]]
    interrogate_models: list[dict[str, Any]] = Field(
        default_factory=list, alias="interrogateModels"
    )
    interrogate_model: str = Field(default="", alias="interrogateModel")
    max_renders: int = Field(alias="maxRenders")
    max_upload_mb: int = Field(alias="maxUploadMb")
    note: str = ""


class StudioInterrogateRequestBody(BaseModel):
    """One reverse-prompt request: a photo in, a usable prompt out.

    ``confirmPaid`` mirrors the render path: the interrogation is its own paid
    request, and the server refuses to issue it for a caller that never saw the
    price.
    """

    model_config = ConfigDict(populate_by_name=True)

    source_image_id: str = Field(alias="sourceImageId")
    model_id: str | None = Field(default=None, alias="modelId")
    focus: Literal["replicate", "style"] = "replicate"
    hint: str = ""
    confirm_paid: bool = Field(default=False, alias="confirmPaid")


class StudioInterrogateQuote(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    model: str
    model_label: str = Field(alias="modelLabel")
    focus: str
    source_image_id: str = Field(alias="sourceImageId")
    paid_requests: int = Field(alias="paidRequests")
    requires_confirmation: bool = Field(default=True, alias="requiresConfirmation")
    note: str = ""


class StudioInterrogateResult(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    quote: StudioInterrogateQuote
    prompt: str
    negative_prompt: str = Field(default="", alias="negativePrompt")
    alternatives: list[str] = Field(default_factory=list)
    model: str
    model_label: str = Field(alias="modelLabel")
    focus: str
    source_image_id: str = Field(alias="sourceImageId")
    generated_at: str | None = Field(default=None, alias="generatedAt")
    paid_requests: int = Field(alias="paidRequests")


class StudioRenderResult(BaseModel):
    """The outcome of a confirmed render.

    ``paid_requests`` counts the renders that actually produced a file *and*
    were billed upstream. A render that failed is listed in ``errors`` and is
    never retried: a failed paid call must not silently become a second one.
    """

    model_config = ConfigDict(populate_by_name=True)

    quote: StudioQuote
    items: list[StudioRenderItem]
    errors: list[str] = Field(default_factory=list)
    paid_requests: int = Field(alias="paidRequests")
