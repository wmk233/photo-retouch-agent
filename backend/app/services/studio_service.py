"""The personal studio: a prompt in, a real picture out.

This is deliberately *not* the styling pipeline. There is no local face
analysis, no SCHP/MediaPipe mask, no crop-and-stitch and no pixel-identity
verification, because an instruction-following edit model does not need any of
them: it repaints the whole frame from the prompt. The old pipeline's job was to
keep a masked region byte-identical to the source, and that is precisely what
produced the collar seams it was built to prevent.

What the studio keeps from the old architecture is the part that was never the
problem: the provider seam, the "decode it or refuse it" download check, the
provenance metadata file and the explicit paid-request accounting.

One render is one paid request. The count is computed before anything is sent
and can only be spent by a caller that says ``confirmPaid``, so the price shown
in the UI and the requests actually issued cannot drift apart.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence
from uuid import uuid4

from PIL import Image, UnidentifiedImageError

from app.core.config import Settings, settings as default_settings
from app.core.errors import bad_request
from app.providers.siliconflow_image_provider import SiliconFlowProviderError
from app.providers.studio_image_provider import (
    DEFAULT_IMAGE_SIZE,
    MODE_EDIT,
    MODE_TEXT,
    SUPPORTED_IMAGE_SIZES,
    StudioImageProvider,
)
from app.services.storage import StorageService
from app.services.studio_interrogate import (
    DEFAULT_INTERROGATE_MODEL,
    INTERROGATE_MODELS,
)

logger = logging.getLogger("photo_retouch.studio")

#: The subset of the SiliconFlow catalogue this workstation drives. Only models
#: that answer ``/images/generations`` are listed, and every entry names the
#: mode it belongs to so the UI can never offer an edit model without a photo.
STUDIO_MODEL_IDS = (
    "Qwen/Qwen-Image-Edit-2509",
    "Qwen/Qwen-Image-Edit",
    "Qwen/Qwen-Image",
    "Tongyi-MAI/Z-Image-Turbo",
    "Tongyi-MAI/Z-Image",
)

#: Canvas choices per model. Qwen-Image and Z-Image document *different*
#: recommended sizes, so the two lists are kept apart instead of being merged
#: into one that would be rejected by half the catalogue.
_QWEN_SIZES = SUPPORTED_IMAGE_SIZES
_ZIMAGE_SIZES = ("512x512", "768x1024", "1024x576", "576x1024")

#: One render is one request, and a batch of N renders is N requests. Four is
#: the ceiling: enough for a side-by-side comparison, small enough that a typo
#: in the prompt cannot run up a bill.
MAX_RENDERS_PER_REQUEST = 4

STUDIO_DIRNAME = "studio"


@dataclass(frozen=True)
class StudioModel:
    model_id: str
    label: str
    mode: str
    note: str
    image_sizes: tuple[str, ...] = ()
    requires_source: bool = False

    def to_payload(self) -> dict[str, Any]:
        return {
            "id": self.model_id,
            "label": self.label,
            "mode": self.mode,
            "note": self.note,
            "imageSizes": list(self.image_sizes),
            "requiresSource": self.requires_source,
        }


STUDIO_MODELS: tuple[StudioModel, ...] = (
    StudioModel(
        model_id="Qwen/Qwen-Image-Edit-2509",
        label="Qwen-Image-Edit 2509",
        mode=MODE_EDIT,
        note="改图首选：按指令改，未点到的地方基本保持原样",
        requires_source=True,
    ),
    StudioModel(
        model_id="Qwen/Qwen-Image-Edit",
        label="Qwen-Image-Edit",
        mode=MODE_EDIT,
        note="改图的上一版，风格可能与 2509 略有差异",
        requires_source=True,
    ),
    StudioModel(
        model_id="Qwen/Qwen-Image",
        label="Qwen-Image",
        mode=MODE_TEXT,
        note="文生图：画质最好，适合出写真级成片",
        image_sizes=_QWEN_SIZES,
    ),
    StudioModel(
        model_id="Tongyi-MAI/Z-Image-Turbo",
        label="Z-Image Turbo",
        mode=MODE_TEXT,
        note="文生图：快，适合快速试构图",
        image_sizes=_ZIMAGE_SIZES,
    ),
    StudioModel(
        model_id="Tongyi-MAI/Z-Image",
        label="Z-Image",
        mode=MODE_TEXT,
        note="文生图：Z-Image 标准版",
        image_sizes=_ZIMAGE_SIZES,
    ),
)

_MODEL_INDEX = {model.model_id: model for model in STUDIO_MODELS}


@dataclass
class StudioRenderRequest:
    prompt: str
    mode: str
    model_id: str | None = None
    source_image_id: str | None = None
    image_size: str | None = None
    negative_prompt: str = ""
    count: int = 1
    confirm_paid: bool = False
    seeds: Sequence[int] = field(default_factory=tuple)


class StudioService:
    """Owns the studio's model catalogue, its paid-request accounting and its outputs."""

    def __init__(
        self,
        config: Settings = default_settings,
        storage: StorageService | None = None,
    ) -> None:
        self.config = config
        self.storage = storage or StorageService(config)

    # --- catalogue --------------------------------------------------------

    def status(self) -> dict[str, Any]:
        configured = bool(self.config.siliconflow_api_key)
        return {
            "configured": configured,
            "provider": "siliconflow",
            "defaultModel": self._default_model_id(configured),
            "defaultEditModel": "Qwen/Qwen-Image-Edit-2509",
            "defaultTextModel": "Qwen/Qwen-Image",
            "models": [model.to_payload() for model in STUDIO_MODELS],
            # The reverse-prompt catalogue is listed here so the page can build
            # its picker from one status call instead of a second round trip.
            "interrogateModels": [model.to_payload() for model in INTERROGATE_MODELS],
            "interrogateModel": DEFAULT_INTERROGATE_MODEL if configured else "",
            "maxRenders": MAX_RENDERS_PER_REQUEST,
            "maxUploadMb": self.config.max_upload_bytes // (1024 * 1024),
            "note": (
                "费用按硅基流动账单结算：每张成图 = 1 次付费请求。"
                "页面在发出请求前会先弹出费用确认。"
            ),
        }

    def _default_model_id(self, configured: bool) -> str:
        return "Qwen/Qwen-Image-Edit-2509" if configured else ""

    # --- quoting ----------------------------------------------------------

    def quote(self, request: StudioRenderRequest) -> dict[str, Any]:
        """Resolve and validate a request into the exact number of paid calls.

        Called by the UI before the confirmation dialog and again by the render
        path, so the price a person approves is the price that is charged.
        """
        prompt = (request.prompt or "").strip()
        if not prompt:
            raise bad_request("请先写下你要做什么（提示词不能为空）。")

        model = self._resolve_model(request)
        if model.mode == MODE_EDIT and not (request.source_image_id or "").strip():
            raise bad_request(f"{model.label} 是在你的照片上改，请先上传一张输入图。")

        if request.source_image_id and model.mode == MODE_EDIT:
            if self.storage.resolve_image_path(request.source_image_id) is None:
                raise bad_request("找不到这张输入图，请重新上传。")

        count = int(request.count or 1)
        if count < 1:
            count = 1
        if count > MAX_RENDERS_PER_REQUEST:
            raise bad_request(f"一次最多生成 {MAX_RENDERS_PER_REQUEST} 张。")

        image_size = (request.image_size or "").strip() or DEFAULT_IMAGE_SIZE
        if model.mode == MODE_TEXT:
            if model.image_sizes and image_size not in model.image_sizes:
                raise bad_request(
                    f"{model.label} 不支持这个尺寸，请从页面给出的选项里选。"
                )

        return {
            "model": model.model_id,
            "modelLabel": model.label,
            "mode": model.mode,
            "count": count,
            "paidRequests": count,
            "imageSize": image_size if model.mode == MODE_TEXT else None,
            "sourceImageId": request.source_image_id,
            "requiresConfirmation": True,
            "note": f"本次将发起 {count} 次真实付费请求（{model.label}）。",
        }

    def _resolve_model(self, request: StudioRenderRequest) -> StudioModel:
        requested = (request.model_id or "").strip()
        if not requested:
            requested = (
                "Qwen/Qwen-Image" if request.mode == MODE_TEXT else "Qwen/Qwen-Image-Edit-2509"
            )
        model = _MODEL_INDEX.get(requested)
        if model is None:
            raise bad_request(f"这个工作台还没有接入模型：{requested}")
        if request.mode and request.mode in (MODE_EDIT, MODE_TEXT) and model.mode != request.mode:
            raise bad_request(f"{model.label} 不属于「{request.mode}」模式。")
        return model

    # --- rendering --------------------------------------------------------

    async def render(self, request: StudioRenderRequest) -> dict[str, Any]:
        """Run the quoted renders. Never retries: a failure is reported, not re-billed."""
        if not request.confirm_paid:
            raise bad_request("未确认费用，已拒绝发起付费请求。")
        if not self.config.siliconflow_api_key:
            raise bad_request(
                "还没有配置 SILICONFLOW_API_KEY，无法出图。"
                "在仓库根目录的 .env 里写入后重启服务即可。"
            )

        quote = self.quote(request)
        model = _MODEL_INDEX[quote["model"]]
        source_path = (
            self.storage.resolve_image_path(request.source_image_id)
            if request.source_image_id
            else None
        )

        output_dir = self.config.outputs_dir / STUDIO_DIRNAME
        output_dir.mkdir(parents=True, exist_ok=True)

        seeds = [int(seed) for seed in request.seeds][: quote["count"]]
        while len(seeds) < quote["count"]:
            seeds.append(random.randint(0, 9_999_999_999))

        items: list[dict[str, Any]] = []
        errors: list[str] = []
        for index in range(quote["count"]):
            # One billed request per iteration, issued here and nowhere else.
            try:
                items.append(
                    await self._render_one(
                        model=model,
                        quote=quote,
                        prompt=request.prompt.strip(),
                        negative_prompt=request.negative_prompt.strip(),
                        source_path=source_path,
                        output_dir=output_dir,
                        seed=seeds[index],
                        index=index,
                    )
                )
            except (SiliconFlowProviderError, OSError) as exc:
                # No retry: a paid call that failed is reported, never repeated.
                logger.warning("studio render %s failed: %s", index, exc)
                errors.append(str(exc))

        if not items:
            raise bad_request(errors[0] if errors else "生成失败，请稍后再试。")

        return {
            "quote": quote,
            "items": items,
            "errors": errors,
            "paidRequests": len(items),
        }

    async def _render_one(
        self,
        *,
        model: StudioModel,
        quote: dict[str, Any],
        prompt: str,
        negative_prompt: str,
        source_path: Path | None,
        output_dir: Path,
        seed: int,
        index: int,
    ) -> dict[str, Any]:
        render_id = f"studio_{uuid4().hex[:12]}"
        raw_path = output_dir / f".{render_id}.raw"
        final_path = output_dir / f"{render_id}.png"

        provider = StudioImageProvider(
            api_key=self.config.siliconflow_api_key or "",
            base_url=self.config.siliconflow_base_url,
            model_name=model.model_id,
            mode=model.mode,
            image_size=quote.get("imageSize"),
            seed=seed,
            timeout_seconds=self.config.provider_timeout_seconds,
            download_limit_bytes=self.config.provider_download_limit_bytes,
        )

        try:
            await provider.run(
                prompt=prompt,
                output_path=raw_path,
                source_path=source_path,
                negative_prompt=negative_prompt,
            )
            width, height, size_bytes = self._normalise(raw_path, final_path)
        finally:
            raw_path.unlink(missing_ok=True)

        metadata = self._read_metadata(output_dir, final_path.name)
        return {
            "imageId": render_id,
            "url": f"/data/outputs/{STUDIO_DIRNAME}/{final_path.name}",
            "width": width,
            "height": height,
            "bytes": size_bytes,
            "model": model.model_id,
            "modelLabel": model.label,
            "mode": model.mode,
            "seed": seed,
            "prompt": prompt,
            "negativePrompt": negative_prompt,
            "sourceImageId": quote.get("sourceImageId"),
            "requestId": metadata.get("requestId"),
            "generatedAt": metadata.get("generatedAt")
            or datetime.now(timezone.utc).isoformat(),
            "index": index,
        }

    @staticmethod
    def _normalise(raw_path: Path, final_path: Path) -> tuple[int, int, int]:
        """Re-encode the downloaded bytes as a real PNG.

        The result CDN serves genuine JPEG bytes under a generic content type, so
        the bytes on disk and the ``.png`` extension the static server will
        advertise cannot be assumed to agree. Decoding and re-encoding settles
        it: what is served is always what the browser is promised.
        """
        try:
            with Image.open(raw_path) as image:
                image.load()
                rgb = image.convert("RGB") if image.mode not in ("RGB", "RGBA") else image
                rgb.save(final_path, format="PNG", optimize=True)
                width, height = rgb.size
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise SiliconFlowProviderError(
                "模型返回的图片无法解码，本次没有产出可用成片。"
            ) from exc
        return width, height, final_path.stat().st_size

    @staticmethod
    def _read_metadata(output_dir: Path, filename: str) -> dict[str, Any]:
        path = output_dir / "_meta" / f"{filename}.json"
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    # --- history ----------------------------------------------------------

    def history(self, limit: int = 24) -> list[dict[str, Any]]:
        """Most recent renders, newest first, read back from the on-disk trail.

        The metadata file is the record of what was actually billed, so the
        gallery survives a server restart instead of being a render-only view.
        """
        meta_dir = self.config.outputs_dir / STUDIO_DIRNAME / "_meta"
        if not meta_dir.exists():
            return []
        entries: list[dict[str, Any]] = []
        for meta_path in meta_dir.glob("studio_*.png.json"):
            image_path = meta_dir.parent / meta_path.name[: -len(".json")]
            if not image_path.exists():
                continue
            record = self._read_metadata(meta_dir.parent, image_path.name)
            try:
                with Image.open(image_path) as image:
                    width, height = image.size
            except (UnidentifiedImageError, OSError):
                continue
            entries.append(
                {
                    "imageId": image_path.stem,
                    "url": f"/data/outputs/{STUDIO_DIRNAME}/{image_path.name}",
                    "width": width,
                    "height": height,
                    "bytes": image_path.stat().st_size,
                    "model": record.get("model"),
                    "requestId": record.get("requestId"),
                    "generatedAt": record.get("generatedAt"),
                    "seed": record.get("seed"),
                    "label": (record.get("planId") or "").strip() or None,
                }
            )
        entries.sort(key=lambda item: item.get("generatedAt") or "", reverse=True)
        return entries[: max(0, min(limit, 96))]


__all__ = [
    "STUDIO_MODELS",
    "StudioModel",
    "StudioRenderRequest",
    "StudioService",
    "MAX_RENDERS_PER_REQUEST",
]
