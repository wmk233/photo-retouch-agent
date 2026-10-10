"""Freeform SiliconFlow image provider for the personal studio workstation.

The styling provider (:mod:`app.providers.siliconflow_image_provider`) is wired
into the masked retouch pipeline. It refuses to run without a source photo and
it appends a fixed "keep the same person" persona block to every request, which
is exactly right for a portrait restyle and exactly wrong for everything else.

The studio is a different tool: prompt in, picture out. The instruction the user
types *is* the prompt -- nothing is bolted on in front of it -- and the same
workstation has to serve interiors, objects and scenes as well as people. Only
the transport layer is therefore shared with the styling provider: the request
body shape, response parsing, the SSRF-guarded download, the decode-or-refuse
check and the provenance metadata file. Prompt and payload construction are
replaced wholesale.

Two modes exist and they differ in the one field the upstream API will not
forgive:

* ``edit`` -- ``{model, prompt, image}`` starts from the user's own photo. The
  edit family rejects ``image_size`` outright, so it is never sent.
* ``text2image`` -- ``{model, prompt, image_size}`` starts from nothing, and
  Qwen-Image *requires* an explicit canvas.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.providers.siliconflow_image_provider import SiliconFlowImageProvider
from app.schemas.retouch import RetouchPlan

MODE_EDIT = "edit"
MODE_TEXT = "text2image"
MODES = (MODE_EDIT, MODE_TEXT)

#: The model's own recommended 1:1 canvas, used when the caller does not pick a
#: size. Only ever sent in ``text2image`` mode.
DEFAULT_IMAGE_SIZE = "1328x1328"

#: Sizes the Qwen-Image family documents as recommended. Kept as a tuple so the
#: API and the UI offer the same choices instead of drifting apart.
SUPPORTED_IMAGE_SIZES = (
    "1328x1328",
    "1664x928",
    "928x1664",
    "1472x1140",
    "1140x1472",
    "1584x1056",
    "1056x1584",
)

#: Seed ceiling published by SiliconFlow for the image generation endpoint.
MAX_SEED = 9_999_999_999


class StudioImageProvider(SiliconFlowImageProvider):
    """SiliconFlow image generation with the studio's prompt-pass-through rules."""

    provider_name = "siliconflow-studio"

    def __init__(
        self,
        *,
        mode: str = MODE_EDIT,
        image_size: str | None = None,
        seed: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if mode not in MODES:
            raise ValueError(f"Unsupported studio mode: {mode}")
        self.mode = mode
        self.image_size = (image_size or DEFAULT_IMAGE_SIZE).strip()
        self.seed = seed

    async def run(
        self,
        *,
        prompt: str,
        output_path: Path,
        source_path: Path | None = None,
        negative_prompt: str = "",
    ) -> None:
        """Studio-facing entry point.

        ``source_path`` is required in ``edit`` mode and ignored in
        ``text2image`` mode, so the caller never has to invent a placeholder
        photo for a prompt-only render.
        """
        plan = _studio_plan(
            prompt=prompt,
            negative_prompt=negative_prompt,
            source_provider=self.provider_name,
        )
        await self.edit_image(
            source_path or Path("."),
            output_path,
            plan,
            "",
        )

    def _build_payload(
        self,
        source_path: Path,
        plan: RetouchPlan,
        user_instruction: str,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model_name,
            "prompt": self._build_prompt(plan, user_instruction),
        }
        if plan.negative_prompt.strip():
            payload["negative_prompt"] = plan.negative_prompt.strip()
        if self.seed is not None:
            payload["seed"] = int(self.seed)

        if self.mode == MODE_TEXT:
            payload["image_size"] = self.image_size
            return payload

        payload["image"] = self._encode_image(source_path)
        # Optional extra references, reserved for the consistency feature that
        # keeps a subject stable across a batch. Omitted entirely until used,
        # because the single-image flow is the only one the UI drives today.
        for index, reference in enumerate(self.reference_image_paths[:2], start=2):
            payload[f"image{index}"] = self._encode_image(reference)
        return payload

    @staticmethod
    def _build_prompt(plan: RetouchPlan, user_instruction: str) -> str:
        """Return the user's own words, unedited and unadorned.

        No persona or identity boilerplate is added here on purpose: the studio
        serves interiors and objects too, where "keep the same person" would be
        nonsense, and silently rewriting the instruction would make the result
        impossible to explain.
        """
        parts = [plan.edit_prompt.strip()]
        extra = user_instruction.strip()
        if extra and extra != parts[0]:
            parts.append(extra)
        return "\n".join(part for part in parts if part)


def _studio_plan(
    *,
    prompt: str,
    negative_prompt: str = "",
    source_provider: str | None = None,
) -> RetouchPlan:
    """Adapt a free-form instruction to the plan record the renderer expects.

    The styling pipeline has a plan because a look is designed before it is
    rendered. The studio has no design pass, so the instruction *is* the plan;
    this keeps the provider protocol untouched instead of forking it.
    """
    text = prompt.strip()
    return RetouchPlan(
        planId="studio",
        domainType="studio",
        title=text[:60] or "Studio render",
        description=text,
        intensity="custom",
        editPrompt=text,
        negativePrompt=negative_prompt.strip(),
        expectedChanges=[],
        sourceProvider=source_provider,
    )
