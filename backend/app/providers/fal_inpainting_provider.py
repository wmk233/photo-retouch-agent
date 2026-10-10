"""fal.ai Qwen-Image-Edit **inpainting** provider (native mask editing).

This is the P2.8 backend. The previous styling flow sent a whole photograph to a
whole-image edit model and pasted the original face back afterwards; the model
had never been told which pixels it was allowed to touch, so the pasted-back
face was the only thing standing between the user and a repainted identity.
Here the mask *is* the instruction:

    white (255) = repaint this          black (0) = keep these original pixels

Two different inputs are therefore sent on every call -- ``image_url`` (the
original photograph) and ``mask_url`` (the black/white edit mask in that photo's
own coordinates) -- plus the prompt the styling planner already produced. The
mask is never optional: constructing this provider without one is a programming
error and it refuses instead of quietly degrading to a whole-image edit.

What this module deliberately does *not* do:

* **No second job framework.** It speaks the fal queue API inline (submit, poll,
  fetch) and leaves scheduling to the existing job layer.
* **No retry of a billed request.** The submit POST is issued exactly once. Only
  the status ``GET`` is repeated, because polling is free; a failure is raised,
  never re-submitted.
* **No upload to third-party storage.** Both inputs travel as ``data:`` URIs in
  the request body, so the photograph is not parked in a bucket to be referenced.
* **No local model, no GPU.** It is an HTTP client.

Nothing here validates that the result kept the subject's face: that is
:mod:`app.domains.portrait.inpaint_verify`, which composites the untouched
pixels back from the original and reports its own, separate statuses.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import ipaddress
import json
import mimetypes
import time
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, NamedTuple

import httpx
import numpy as np
from PIL import Image, UnidentifiedImageError

from app.schemas.retouch import RetouchPlan

METADATA_DIRNAME = "_meta"

# The result CDN does not always label its payload precisely: real PNG bytes are
# regularly served as ``application/octet-stream``. Both declarations are
# acceptable; any other explicit type (``text/html`` sign-in page, ``application/
# json`` error envelope, ...) is refused up front. Decoding decides the rest.
GENERIC_BINARY_CONTENT_TYPES = frozenset(
    {"application/octet-stream", "binary/octet-stream"}
)

# Bytes are only trusted once Pillow fully decodes them into one of these formats.
SUPPORTED_IMAGE_FORMATS = frozenset({"PNG", "JPEG", "WEBP"})

REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
MAX_DOWNLOAD_REDIRECTS = 3

#: fal queue vocabulary. ``COMPLETED`` is the only terminal success.
COMPLETED_STATUSES = frozenset({"COMPLETED", "OK", "SUCCESS"})
FAILED_STATUSES = frozenset({"FAILED", "ERROR", "CANCELLED", "CANCELED"})
PENDING_STATUSES = frozenset({"IN_QUEUE", "IN_PROGRESS", "QUEUED", "RUNNING"})

#: Hard ceiling on status polls, so an unresponsive queue cannot loop forever.
MAX_STATUS_POLLS = 600

#: A mask that opens less than this share of the frame has nothing to repaint,
#: and one that opens more than the remainder is a whole-image edit wearing a
#: mask's clothes. Both are refused before anything is billed.
MIN_EDITABLE_SHARE = 0.001
MAX_EDITABLE_SHARE = 0.90


class FalInpaintingProviderError(RuntimeError):
    """Safe fal.ai error that never contains API credentials."""


class DownloadedImage(NamedTuple):
    """A validated picture fetched from the provider's result URL."""

    content: bytes
    content_type: str
    final_url: str
    status_code: int


class MaskCheck(NamedTuple):
    """What the mask looked like when it was about to be sent."""

    width: int
    height: int
    editable_pixels: int
    editable_share: float
    sha256: str


class FalInpaintingProvider:
    """``fal-ai/qwen-image-edit/inpaint`` behind the shared provider contract.

    The queue lifecycle is: ``POST {endpoint}`` -> poll ``status_url`` ->
    ``GET response_url``. All three are plain JSON over HTTPS.
    """

    provider_name = "fal-inpaint"
    model_name = "fal-ai/qwen-image-edit/inpaint"
    output_extension = ".png"

    #: Tells the job layer that this backend cannot run without an explicit
    #: black/white edit mask, so the workflow builds one instead of silently
    #: degrading to the whole-image path.
    requires_edit_mask = True

    def __init__(
        self,
        api_key: str,
        endpoint: str,
        model_name: str = "fal-ai/qwen-image-edit/inpaint",
        timeout_seconds: float = 180,
        download_limit_bytes: int = 30 * 1024 * 1024,
        poll_interval_seconds: float = 2.0,
        max_poll_seconds: float = 600,
        extra_parameters: dict[str, Any] | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._api_key = api_key
        self.endpoint = endpoint.rstrip("/")
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds
        self.download_limit_bytes = download_limit_bytes
        self.poll_interval_seconds = poll_interval_seconds
        self.max_poll_seconds = max_poll_seconds
        #: Pass-through for documented model knobs (``num_inference_steps``,
        #: ``guidance_scale``, ``strength``, ``negative_prompt``, ``seed``, ...).
        #: Nothing is hard-coded here that the model's own defaults already cover.
        self.extra_parameters = dict(extra_parameters or {})
        self._transport = transport

    # ------------------------------------------------------------------ public

    async def edit_image(
        self,
        source_path: Path,
        output_path: Path,
        plan: RetouchPlan,
        user_instruction: str,
        *,
        mask_path: Path | None = None,
    ) -> None:
        """Repaint the masked region of ``source_path`` and write the raw result.

        The bytes written to ``output_path`` are the model's own output, untouched:
        restoring the kept pixels is a separate, auditable step
        (:func:`app.domains.portrait.inpaint_verify.restore_kept_pixels`) so the
        two can never be confused with one another.
        """
        if not self._api_key.strip():
            raise FalInpaintingProviderError(
                "fal.ai API key is required for native mask inpainting; no photo "
                "was uploaded and no request was sent. Set FAL_KEY or provide "
                "X-Action-API-Key."
            )
        if mask_path is None:
            raise FalInpaintingProviderError(
                "Native mask inpainting requires an edit mask; refusing to fall "
                "back to a whole-image edit. Build the black/white mask (SCHP "
                "parsing + MediaPipe face guard) and pass it as mask_path."
            )
        mask_path = Path(mask_path)
        if not mask_path.is_file():
            raise FalInpaintingProviderError(
                f"Edit mask not found: {mask_path.name}. No request was sent."
            )
        if not Path(source_path).is_file():
            raise FalInpaintingProviderError(
                f"Source image not found: {Path(source_path).name}. No request was sent."
            )

        mask_check = self._inspect_mask(source_path, mask_path)
        payload = self._build_payload(source_path, mask_path, plan, user_instruction)

        headers = {
            "Authorization": f"Key {self._api_key}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(
            timeout=self.timeout_seconds,
            # Redirects are resolved by hand so every hop can be validated: these
            # URLs arrive from a third party and must not become an arbitrary
            # fetch target.
            follow_redirects=False,
            transport=self._transport,
        ) as client:
            submission = await self._submit(client, headers, payload)
            result = await self._await_completion(client, headers, submission)
            image_url = self._extract_image_url(result)
            self._validate_download_url(image_url)
            download = await self._download_image(client, image_url)

        self._store_download(download, output_path)
        self._write_metadata(
            output_path,
            plan=plan,
            submission=submission,
            result=result,
            mask_check=mask_check,
            content_type=download.content_type,
            source_host=httpx.URL(download.final_url).host,
        )

    # ----------------------------------------------------------------- payload

    def _build_payload(
        self,
        source_path: Path,
        mask_path: Path,
        plan: RetouchPlan,
        user_instruction: str,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "prompt": self._build_prompt(plan, user_instruction),
            # Both inputs are distinct fields and both are the real thing: the
            # photo in ``image_url``, the black/white mask in ``mask_url``.
            "image_url": self._encode_image(source_path),
            "mask_url": self._encode_image(mask_path),
            "output_format": "png",
            "num_images": 1,
            "sync_mode": False,
        }
        for key, value in self.extra_parameters.items():
            payload[key] = value
        return payload

    @staticmethod
    def _encode_image(path: Path) -> str:
        mime_type, _ = mimetypes.guess_type(path.name)
        if not mime_type or not mime_type.startswith("image/"):
            raise FalInpaintingProviderError(
                f"Unsupported image type for {path.name}: {mime_type!r}."
            )
        encoded = base64.b64encode(Path(path).read_bytes()).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

    @staticmethod
    def _build_prompt(plan: RetouchPlan, user_instruction: str) -> str:
        instructions = [
            plan.edit_prompt,
            "只重绘蒙版中的白色区域（发型与服装），黑色区域保持原样像素不变。",
            "必须保持同一个人：人物身份、面部几何、五官比例、笑容、年龄表现与姿态"
            "都不改变；不换脸、不替换成陌生人、不改变性别。",
            "不得新增或删除人物，不得添加文字或水印；避免塑料皮肤与明显 AI 生成痕迹。",
        ]
        if user_instruction.strip():
            instructions.append(f"用户补充要求：{user_instruction.strip()}")
        return "\n".join(instructions)

    # ------------------------------------------------------------------- mask

    def _inspect_mask(self, source_path: Path, mask_path: Path) -> MaskCheck:
        """Refuse a mask that cannot mean what an inpainting mask means.

        The mask has to be in the photo's own coordinates (same HxW), strictly
        black/white, and non-trivial: a mask that is all black asks for nothing
        and one that is almost all white is a whole-image edit in disguise.
        """
        try:
            with Image.open(source_path) as source_image:
                source_size = source_image.size
        except (UnidentifiedImageError, OSError) as exc:
            raise FalInpaintingProviderError(
                f"Source image could not be decoded: {source_path.name}."
            ) from exc
        try:
            with Image.open(mask_path) as mask_image:
                mask_image.load()
                mask_size = mask_image.size
                mask_array = np.asarray(mask_image.convert("L"))
        except (UnidentifiedImageError, OSError) as exc:
            raise FalInpaintingProviderError(
                f"Edit mask could not be decoded: {mask_path.name}."
            ) from exc

        if mask_size != source_size:
            raise FalInpaintingProviderError(
                "Edit mask is not in the photo's own coordinates: mask is "
                f"{mask_size[0]}x{mask_size[1]}, photo is "
                f"{source_size[0]}x{source_size[1]}. Refusing to send either."
            )
        levels = np.unique(mask_array)
        if not np.all(np.isin(levels, (0, 255))):
            raise FalInpaintingProviderError(
                "Edit mask must be strictly black/white (0 = keep, 255 = repaint); "
                f"it contains {len(levels)} grey level(s)."
            )
        total = int(mask_array.size)
        editable_pixels = int((mask_array == 255).sum())
        editable_share = editable_pixels / float(max(total, 1))
        if editable_share < MIN_EDITABLE_SHARE:
            raise FalInpaintingProviderError(
                "Edit mask opens almost nothing to repaint "
                f"({editable_share:.4%} editable); refusing to pay for a no-op edit."
            )
        if editable_share > MAX_EDITABLE_SHARE:
            raise FalInpaintingProviderError(
                "Edit mask opens almost the whole frame "
                f"({editable_share:.1%} editable); that is a whole-image edit, not "
                "a masked one -- use the SiliconFlow whole-image backend instead."
            )
        return MaskCheck(
            width=int(mask_size[0]),
            height=int(mask_size[1]),
            editable_pixels=editable_pixels,
            editable_share=round(editable_share, 6),
            sha256=hashlib.sha256(Path(mask_path).read_bytes()).hexdigest(),
        )

    # ------------------------------------------------------------------- queue

    async def _submit(
        self,
        client: httpx.AsyncClient,
        headers: dict[str, str],
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Issue the one and only billed request: the submit POST."""
        try:
            response = await client.post(self.endpoint, headers=headers, json=payload)
        except httpx.HTTPError as exc:
            # A transport failure is not retried: the request may well have been
            # received and queued, and a second POST would be a second charge.
            raise FalInpaintingProviderError(
                "fal.ai request failed in transit and was not retried: "
                f"{type(exc).__name__}. Check the fal queue before submitting again."
            ) from exc
        return self._parse_response(response, "submit")

    async def _await_completion(
        self,
        client: httpx.AsyncClient,
        headers: dict[str, str],
        submission: dict[str, Any],
    ) -> dict[str, Any]:
        """Poll the queue until the request is done, then fetch the result.

        Only free ``GET``s happen here. The request id is already paid for, so a
        timeout reports the id and stops instead of re-submitting.
        """
        request_id = submission.get("request_id")
        status_url = submission.get("status_url")
        response_url = submission.get("response_url")
        if not status_url:
            # A synchronous-style body already carrying the output is accepted.
            if self._looks_like_output(submission):
                return submission
            raise FalInpaintingProviderError(
                "fal.ai submit response carried no status_url, so the queued "
                "request cannot be followed."
            )
        status_url = self._validate_provider_url(str(status_url))
        response_url = self._validate_provider_url(str(response_url)) if response_url else None

        deadline = time.monotonic() + self.max_poll_seconds
        polls = 0
        while True:
            if polls >= MAX_STATUS_POLLS or time.monotonic() > deadline:
                raise FalInpaintingProviderError(
                    "fal.ai request "
                    f"{request_id or '(unknown id)'} was still running after "
                    f"{self.max_poll_seconds:.0f}s of polling; it was NOT "
                    "re-submitted. Check the fal queue for that request id before "
                    "paying for another run."
                )
            polls += 1
            response = await client.get(status_url, headers=headers)
            payload = self._parse_response(response, "status")
            status = str(payload.get("status", "")).upper()
            if status in FAILED_STATUSES:
                raise FalInpaintingProviderError(
                    f"fal.ai request {request_id or '(unknown id)'} ended as "
                    f"{status}: {self._error_detail(payload)}"
                )
            if status in COMPLETED_STATUSES or self._looks_like_output(payload):
                target = self._validate_provider_url(
                    str(payload.get("response_url") or response_url or "")
                )
                if not target:
                    if self._looks_like_output(payload):
                        return payload
                    raise FalInpaintingProviderError(
                        "fal.ai request completed without a response_url."
                    )
                result_response = await client.get(target, headers=headers)
                return self._parse_response(result_response, "result")
            if status and status not in PENDING_STATUSES:
                raise FalInpaintingProviderError(
                    f"fal.ai queue reported an unknown status {status!r} for "
                    f"request {request_id or '(unknown id)'}."
                )
            await asyncio.sleep(self.poll_interval_seconds)

    @staticmethod
    def _looks_like_output(payload: dict[str, Any]) -> bool:
        return isinstance(payload.get("images"), list) and bool(payload["images"])

    def _parse_response(self, response: httpx.Response, stage: str) -> dict[str, Any]:
        try:
            payload = response.json()
        except ValueError as exc:
            raise FalInpaintingProviderError(
                f"fal.ai {stage} returned invalid JSON (HTTP {response.status_code})."
            ) from exc
        if not isinstance(payload, dict):
            raise FalInpaintingProviderError(
                f"fal.ai {stage} returned an unexpected payload."
            )
        if response.is_error or payload.get("error"):
            raise FalInpaintingProviderError(
                f"fal.ai {stage} failed (HTTP {response.status_code}): "
                f"{self._error_detail(payload)}"
            )
        return payload

    def _error_detail(self, payload: dict[str, Any]) -> str:
        detail: Any = payload.get("detail") or payload.get("error") or payload.get(
            "message"
        )
        if isinstance(detail, dict):
            detail = detail.get("message") or detail.get("detail") or detail
        if isinstance(detail, list) and detail:
            first = detail[0]
            if isinstance(first, dict):
                detail = first.get("msg") or first
        return str(detail or "no detail").replace(self._api_key, "[redacted]")

    @staticmethod
    def _extract_image_url(payload: dict[str, Any]) -> str:
        entries = payload.get("images")
        if isinstance(entries, list):
            for entry in entries:
                if isinstance(entry, dict):
                    url = entry.get("url") or entry.get("imageUrl")
                    if url:
                        return str(url)
                elif isinstance(entry, str) and entry:
                    return entry
        data = payload.get("data")
        if isinstance(data, dict) and data.get("url"):
            return str(data["url"])
        raise FalInpaintingProviderError(
            "fal.ai result did not contain an output image."
        )

    # --------------------------------------------------------------- downloads

    async def _download_image(
        self, client: httpx.AsyncClient, image_url: str
    ) -> DownloadedImage:
        """Fetch the result URL and return the bytes only if they are a real picture."""
        request_url = self._validate_download_url(image_url)
        for attempt in range(MAX_DOWNLOAD_REDIRECTS + 1):
            async with client.stream("GET", request_url) as response:
                if response.status_code in REDIRECT_STATUSES:
                    location = response.headers.get("location")
                    if not location:
                        raise FalInpaintingProviderError(
                            "fal.ai output URL redirected without a target."
                        )
                    if attempt == MAX_DOWNLOAD_REDIRECTS:
                        raise FalInpaintingProviderError(
                            "fal.ai output URL redirected too many times."
                        )
                    request_url = self._validate_download_url(
                        str(httpx.URL(request_url).join(location))
                    )
                    continue
                if response.is_error:
                    raise FalInpaintingProviderError(
                        "Failed to download the fal.ai output image "
                        f"(HTTP {response.status_code})."
                    )
                content_type = response.headers.get("content-type", "")
                if not self._content_type_may_carry_an_image(content_type):
                    body = await self._read_limited(response)
                    raise FalInpaintingProviderError(
                        "fal.ai output URL did not return an image "
                        f"(content-type={content_type!r}, bytes={len(body)})."
                    )
                content = await self._read_limited(response)
                self._verify_image_bytes(content, content_type)
                return DownloadedImage(
                    content=content,
                    content_type=content_type,
                    final_url=request_url,
                    status_code=response.status_code,
                )
        raise FalInpaintingProviderError("fal.ai output URL could not be resolved.")

    async def _read_limited(self, response: httpx.Response) -> bytes:
        chunks: list[bytes] = []
        total = 0
        async for chunk in response.aiter_bytes():
            total += len(chunk)
            if total > self.download_limit_bytes:
                raise FalInpaintingProviderError(
                    "fal.ai output image exceeded the download limit."
                )
            chunks.append(chunk)
        return b"".join(chunks)

    @staticmethod
    def _content_type_may_carry_an_image(content_type: str) -> bool:
        declared = content_type.split(";", 1)[0].strip().lower()
        if not declared:
            return True
        return declared.startswith("image/") or declared in GENERIC_BINARY_CONTENT_TYPES

    @staticmethod
    def _verify_image_bytes(content: bytes, content_type: str = "") -> None:
        if not content:
            raise FalInpaintingProviderError(
                "fal.ai output image was empty "
                f"(content-type={content_type!r}, bytes=0)."
            )
        try:
            with Image.open(BytesIO(content)) as image:
                image_format = image.format
                image.load()
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise FalInpaintingProviderError(
                "fal.ai output image could not be decoded "
                f"(content-type={content_type!r}, bytes={len(content)})."
            ) from exc
        if image_format not in SUPPORTED_IMAGE_FORMATS:
            raise FalInpaintingProviderError(
                f"fal.ai output image format {image_format!r} is not supported."
            )

    @staticmethod
    def _store_download(download: DownloadedImage, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(download.content)

    # ------------------------------------------------------------------- URLs

    def _validate_provider_url(self, url: str) -> str:
        """Only follow queue URLs on the host this provider was configured with."""
        if not url:
            return ""
        validated = self._validate_download_url(url)
        expected = httpx.URL(self.endpoint).host
        if httpx.URL(validated).host != expected:
            raise FalInpaintingProviderError(
                "fal.ai queue returned a status URL on an unexpected host; "
                "refusing to follow it."
            )
        return validated

    @staticmethod
    def _validate_download_url(url: str) -> str:
        """Reject anything that is not a plain, public http(s) location (SSRF guard)."""
        parsed = httpx.URL(url)
        if parsed.scheme not in ("http", "https"):
            raise FalInpaintingProviderError(
                "fal.ai URL is not an http(s) address; refusing to fetch it."
            )
        host = parsed.host
        if not host:
            raise FalInpaintingProviderError("fal.ai URL has no host.")
        if parsed.userinfo:
            raise FalInpaintingProviderError(
                "fal.ai URL carries credentials; refusing to fetch it."
            )
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            if host.lower() == "localhost" or host.lower().endswith(".localhost"):
                raise FalInpaintingProviderError(
                    "fal.ai URL points at a local host; refusing to fetch it."
                )
        else:
            if (
                address.is_private
                or address.is_loopback
                or address.is_link_local
                or address.is_unspecified
                or address.is_reserved
            ):
                raise FalInpaintingProviderError(
                    "fal.ai URL points at a non-public address; refusing to fetch it."
                )
        return str(parsed)

    # --------------------------------------------------------------- metadata

    def _write_metadata(
        self,
        output_path: Path,
        *,
        plan: RetouchPlan,
        submission: dict[str, Any],
        result: dict[str, Any],
        mask_check: MaskCheck,
        content_type: str,
        source_host: str | None = None,
    ) -> None:
        """Persist non-secret provenance next to the generated image.

        The ``_meta`` subdirectory keeps image-id lookups, which glob
        ``outputs/<image_id>.*``, from mistaking metadata for a picture. Only the
        *host* of the result URL is recorded; the full URL carries short-lived
        signature parameters and is deliberately not persisted.
        """
        metadata_dir = output_path.parent / METADATA_DIRNAME
        metadata_dir.mkdir(parents=True, exist_ok=True)
        metadata = {
            "provider": self.provider_name,
            "model": self.model_name,
            "mode": "native-mask-inpaint",
            "planId": plan.plan_id,
            "intensity": plan.intensity,
            "outputFile": output_path.name,
            "generatedAt": datetime.now(timezone.utc).isoformat(),
            "requestId": str(submission.get("request_id") or "") or None,
            "endpoint": self.endpoint,
            "bytes": output_path.stat().st_size,
            "contentType": content_type or None,
            "sourceHost": source_host,
            "seed": result.get("seed"),
            "timings": result.get("timings"),
            "hasNsfwConcepts": result.get("has_nsfw_concepts"),
            "mask": {
                "width": mask_check.width,
                "height": mask_check.height,
                "editablePixels": mask_check.editable_pixels,
                "editableShare": mask_check.editable_share,
                "sha256": mask_check.sha256,
                "semantics": "white=repaint black=keep original pixels",
            },
            "inputs": ["image_url", "mask_url"],
            "extraParameters": sorted(self.extra_parameters),
            "note": "raw model output; kept pixels are restored by inpaint_verify",
        }
        (metadata_dir / f"{output_path.name}.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
