from __future__ import annotations
import base64
import ipaddress
import json
import mimetypes
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, NamedTuple, Sequence

import httpx
from PIL import Image, UnidentifiedImageError

from app.schemas.retouch import RetouchPlan

# SiliconFlow's Qwen-Image-Edit family rejects `image_size`. The constant below is
# deliberately absent from the request body and only documents that omission.
UNSUPPORTED_PARAMETERS = ("image_size",)

METADATA_DIRNAME = "_meta"

# A failed download must not cost a second billed generation. The short-lived signed
# result URL is therefore parked owner-only outside the outputs tree, so a download-only
# retry can pick it up; it is never logged and is deleted once recovery is done.
RECOVERY_DIRNAME = "recovery"

# The edit model answers with a short-lived CDN URL and that CDN does not always label
# its payload precisely: real PNG bytes are regularly served as
# ``application/octet-stream``. Both declarations are therefore acceptable; any other
# explicit type (a ``text/html`` sign-in page, an ``application/json`` error envelope,
# ...) is refused up front. Decoding decides the rest, so a mislabelled real image is
# accepted while a well-labelled fake one is still rejected.
GENERIC_BINARY_CONTENT_TYPES = frozenset({"application/octet-stream", "binary/octet-stream"})

# Bytes are only trusted once Pillow fully decodes them into one of these formats.
SUPPORTED_IMAGE_FORMATS = frozenset({"PNG", "JPEG", "WEBP"})

REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
MAX_DOWNLOAD_REDIRECTS = 3


class DownloadedImage(NamedTuple):
    """A validated picture fetched from the provider's result URL."""

    content: bytes
    content_type: str
    final_url: str
    status_code: int


class SiliconFlowProviderError(RuntimeError):
    """Safe SiliconFlow error that never contains API credentials."""


class SiliconFlowImageProvider:
    """SiliconFlow image-edit provider.

    Sends a JSON POST to ``{base_url}/images/generations`` with the flat body
    ``{model, prompt, image}``. ``image`` is the source photo encoded as a data URL,
    so the edit always starts from the real input image instead of a text prompt.
    Exactly one output image is requested per call; ``image2``/``image3`` are
    reserved for a future multi-reference feature and are unused by the UI.
    """

    provider_name = "siliconflow"
    output_extension = ".png"

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model_name: str,
        timeout_seconds: float = 180,
        download_limit_bytes: int = 30 * 1024 * 1024,
        reference_image_paths: Sequence[Path] = (),
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds
        self.download_limit_bytes = download_limit_bytes
        self.reference_image_paths = tuple(reference_image_paths)
        self._transport = transport

    @property
    def endpoint(self) -> str:
        return f"{self.base_url}/images/generations"

    async def edit_image(
        self,
        source_path: Path,
        output_path: Path,
        plan: RetouchPlan,
        user_instruction: str,
    ) -> None:
        payload = self._build_payload(source_path, plan, user_instruction)

        async with httpx.AsyncClient(
            timeout=self.timeout_seconds,
            # Redirects are resolved by hand so every hop can be validated: this URL
            # arrives from a third party and must not become an arbitrary fetch target.
            follow_redirects=False,
            transport=self._transport,
        ) as client:
            response = await client.post(
                self.endpoint,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            result = self._parse_response(response)
            image_url = self._extract_image_url(result)
            # An untrusted or non-public result URL is refused before anything is
            # parked, so only a legitimately fetched URL can ever be kept for retry.
            self._validate_download_url(image_url)
            try:
                download = await self._download_image(client, image_url)
            except SiliconFlowProviderError:
                # The generation was already billed, so keep the result URL available
                # for a download-only retry. The failure itself is still fatal here.
                self._stash_recovery_url(output_path, image_url)
                raise

        # The file is only written after the bytes have been decoded into a real picture.
        self._store_download(download, output_path)
        self._write_metadata(
            output_path,
            plan=plan,
            response=response,
            payload=result,
            content_type=download.content_type,
            source_host=httpx.URL(download.final_url).host,
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
            "image": self._encode_image(source_path),
        }
        # Optional multi-reference inputs: omitted entirely unless explicitly set,
        # because the single-image flow is the only one the UI drives today.
        for index, reference in enumerate(self.reference_image_paths[:2], start=2):
            payload[f"image{index}"] = self._encode_image(reference)
        for unsupported in UNSUPPORTED_PARAMETERS:
            payload.pop(unsupported, None)
        return payload

    @staticmethod
    def _encode_image(path: Path) -> str:
        mime_type, _ = mimetypes.guess_type(path.name)
        if not mime_type or not mime_type.startswith("image/"):
            raise SiliconFlowProviderError("Unsupported input image type.")
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

    @staticmethod
    def _build_prompt(plan: RetouchPlan, user_instruction: str) -> str:
        instructions = [
            plan.edit_prompt,
            "在原始照片的基础上编辑，必须保持同一个人：人物身份、面部几何、五官比例、"
            "年龄表现与姿态都不改变；不换脸、不替换成陌生人、不改变性别。",
            "不得新增或删除人物，不得添加文字或水印；避免塑料皮肤与明显 AI 生成痕迹。",
        ]
        if user_instruction.strip():
            instructions.append(f"用户补充要求：{user_instruction.strip()}")
        return "\n".join(instructions)

    def _parse_response(self, response: httpx.Response) -> dict[str, Any]:
        try:
            payload = response.json()
        except ValueError as exc:
            raise SiliconFlowProviderError(
                f"SiliconFlow API returned invalid JSON (HTTP {response.status_code})."
            ) from exc

        if response.is_error or payload.get("error"):
            error = payload.get("error") or {}
            if isinstance(error, dict):
                message = error.get("message") or error.get("code")
            else:
                message = str(error)
            message = message or payload.get("message") or "Request failed."
            message = str(message).replace(self._api_key, "[redacted]")
            raise SiliconFlowProviderError(
                f"SiliconFlow API call failed (HTTP {response.status_code}): {message}"
            )
        if not isinstance(payload, dict):
            raise SiliconFlowProviderError("SiliconFlow API returned an unexpected payload.")
        return payload

    @staticmethod
    def _extract_image_url(payload: dict[str, Any]) -> str:
        candidates: list[Any] = []
        for key in ("images", "data"):
            entries = payload.get(key)
            if isinstance(entries, list) and entries:
                candidates.append(entries[0])
        for entry in candidates:
            if not isinstance(entry, dict):
                continue
            url = entry.get("url") or entry.get("imageUrl")
            if url:
                return str(url)
        raise SiliconFlowProviderError(
            "SiliconFlow API response did not contain an output image."
        )

    async def _download_image(
        self, client: httpx.AsyncClient, image_url: str
    ) -> DownloadedImage:
        """Fetch the result URL and return the bytes only if they are a real picture.

        Every hop is revalidated, the body is read under a hard byte ceiling, and any
        failure aborts the job: there is no fallback to a mock or a text-to-image call.
        """
        request_url = self._validate_download_url(image_url)
        for attempt in range(MAX_DOWNLOAD_REDIRECTS + 1):
            async with client.stream("GET", request_url) as response:
                if response.status_code in REDIRECT_STATUSES:
                    location = response.headers.get("location")
                    if not location:
                        raise SiliconFlowProviderError(
                            "SiliconFlow output URL redirected without a target."
                        )
                    if attempt == MAX_DOWNLOAD_REDIRECTS:
                        raise SiliconFlowProviderError(
                            "SiliconFlow output URL redirected too many times."
                        )
                    request_url = self._validate_download_url(
                        str(httpx.URL(request_url).join(location))
                    )
                    continue
                if response.is_error:
                    raise SiliconFlowProviderError(
                        "Failed to download SiliconFlow output image "
                        f"(HTTP {response.status_code})."
                    )
                content_type = response.headers.get("content-type", "")
                if not self._content_type_may_carry_an_image(content_type):
                    # Read the (bounded) body only to report how much was returned, so a
                    # failed run stays diagnosable without another billed generation.
                    body = await self._read_limited(response)
                    raise SiliconFlowProviderError(
                        "SiliconFlow output URL did not return an image "
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
        raise SiliconFlowProviderError("SiliconFlow output URL could not be resolved.")

    async def _read_limited(self, response: httpx.Response) -> bytes:
        """Read a body while enforcing the download ceiling as it arrives."""
        chunks: list[bytes] = []
        total = 0
        async for chunk in response.aiter_bytes():
            total += len(chunk)
            if total > self.download_limit_bytes:
                raise SiliconFlowProviderError(
                    "SiliconFlow output image exceeded the download limit."
                )
            chunks.append(chunk)
        return b"".join(chunks)

    @staticmethod
    def _content_type_may_carry_an_image(content_type: str) -> bool:
        declared = content_type.split(";", 1)[0].strip().lower()
        if not declared:
            # Some CDNs omit the header entirely; decoding still has the final say.
            return True
        return declared.startswith("image/") or declared in GENERIC_BINARY_CONTENT_TYPES

    @staticmethod
    def _verify_image_bytes(content: bytes, content_type: str = "") -> None:
        """Fully decode the body; the declaration alone never counts as proof."""
        if not content:
            raise SiliconFlowProviderError(
                "SiliconFlow output image was empty "
                f"(content-type={content_type!r}, bytes=0)."
            )
        try:
            with Image.open(BytesIO(content)) as image:
                image_format = image.format
                # ``load()`` decodes the pixels: a truncated file that carries a valid
                # signature still fails here instead of reaching the output directory.
                image.load()
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise SiliconFlowProviderError(
                "SiliconFlow output image could not be decoded "
                f"(content-type={content_type!r}, bytes={len(content)})."
            ) from exc
        if image_format not in SUPPORTED_IMAGE_FORMATS:
            raise SiliconFlowProviderError(
                f"SiliconFlow output image format {image_format!r} is not supported."
            )

    @staticmethod
    def _store_download(download: DownloadedImage, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(download.content)

    def _stash_recovery_url(self, output_path: Path, image_url: str) -> None:
        """Park a failed download's result URL for a bounded, download-only retry.

        The URL is signed and short-lived, so the file is written owner-only, kept out
        of the outputs tree, never logged and meant to be removed after recovery. A
        failure to stash is swallowed: the original download error is what matters.
        """
        try:
            directory = output_path.parent.parent / RECOVERY_DIRNAME
            directory.mkdir(parents=True, exist_ok=True)
            target = directory / f"{output_path.stem}.url.json"
            target.write_text(
                json.dumps(
                    {
                        "provider": self.provider_name,
                        "model": self.model_name,
                        "outputFile": output_path.name,
                        "stashedAt": datetime.now(timezone.utc).isoformat(),
                        "note": "signed, short-lived result URL; delete after recovery",
                        "resultUrl": image_url,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            target.chmod(0o600)
        except OSError:
            pass

    @staticmethod
    def _validate_download_url(url: str) -> str:
        """Reject anything that is not a plain, public http(s) location.

        The URL is supplied by an upstream API response, so it is untrusted input: no
        other schemes, no embedded credentials, and no loopback, link-local or
        private-network target (SSRF guard).
        """
        parsed = httpx.URL(url)
        if parsed.scheme not in ("http", "https"):
            raise SiliconFlowProviderError(
                "SiliconFlow output URL is not an http(s) address; refusing to download."
            )
        host = parsed.host
        if not host:
            raise SiliconFlowProviderError("SiliconFlow output URL has no host.")
        if parsed.userinfo:
            raise SiliconFlowProviderError(
                "SiliconFlow output URL carries credentials; refusing to download."
            )
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            if host.lower() == "localhost" or host.lower().endswith(".localhost"):
                raise SiliconFlowProviderError(
                    "SiliconFlow output URL points at a local host; refusing to download."
                )
        else:
            if (
                address.is_private
                or address.is_loopback
                or address.is_link_local
                or address.is_unspecified
                or address.is_reserved
            ):
                raise SiliconFlowProviderError(
                    "SiliconFlow output URL points at a non-public address; "
                    "refusing to download."
                )
        return str(parsed)

    def _write_metadata(
        self,
        output_path: Path,
        plan: RetouchPlan,
        response: httpx.Response,
        payload: dict[str, Any],
        content_type: str,
        source_host: str | None = None,
    ) -> None:
        """Persist non-secret provenance next to the generated image.

        The file lives in a ``_meta`` subdirectory so image-id lookups, which glob
        ``outputs/<image_id>.*``, never mistake metadata for a picture.

        Only the host of the result URL is recorded: the full URL carries short-lived
        signature parameters and is deliberately not persisted.
        """
        metadata_dir = output_path.parent / METADATA_DIRNAME
        metadata_dir.mkdir(parents=True, exist_ok=True)
        request_id = (
            response.headers.get("x-request-id")
            or response.headers.get("x-silicon-request-id")
            or payload.get("requestId")
            or payload.get("id")
        )
        image_count = len(payload.get("images") or payload.get("data") or [1])
        metadata = {
            "provider": self.provider_name,
            "model": self.model_name,
            "planId": plan.plan_id,
            "intensity": plan.intensity,
            "outputFile": output_path.name,
            "generatedAt": datetime.now(timezone.utc).isoformat(),
            "requestId": str(request_id) if request_id else None,
            "imageCount": image_count,
            "bytes": output_path.stat().st_size,
            "contentType": content_type or None,
            "sourceHost": source_host,
            "seed": payload.get("seed"),
            "timings": payload.get("timings"),
            "usage": payload.get("usage"),
            "referenceImages": len(self.reference_image_paths),
        }
        (metadata_dir / f"{output_path.name}.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
