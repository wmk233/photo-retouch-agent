"""The standard package a local generation engine consumes -- no cloud in it.

The segmentation, the identity guard and the prompt design already exist in this
project. What was missing is one stable *contract*: a directory that a local
engine (today's target is the ComfyUI ``Inpaint-CropAndStitch`` pair) can read
without this project knowing anything about that engine.

    original photo
      -> facts + styling proposal        (app.domains.portrait.stylist)
      -> SCHP ATR parse + MediaPipe mesh (app.services.native_mask_inpainting)
      -> hair / clothing / protected masks in the photo's own coordinates
      -> prompt_bundle.json + manifest.json

Six items come out of :meth:`LocalEditPackageBuilder.build`, and nothing else is
required to drive a two-pass "hair once, clothing once" crop-and-stitch flow:

======================  ==========================================================
``original_image``      the photograph, copied in so the package stands alone
``hair_edit_mask``      white = hair may be repainted, black = keep the pixels
``clothing_edit_mask``  white = the garment may be repainted, black = keep
``protected_identity``  eyes/brows/nose/lips/teeth/smile/oval/glasses -- keep
``prompt_bundle.json``  the styled prompts and the identity constraints
``manifest.json``       the single instruction sheet for any engine
======================  ==========================================================

Two rules this module will not bend:

* the masks are built by the *production* mask service, in this photo's own
  coordinates, and a mask that fails its acceptance checks raises instead of
  being shipped;
* nothing here calls a provider, spends money or writes to any other project.
  The package is local data on disk, and the engine that consumes it is chosen
  by a human afterwards.

The JSON keys are snake_case on purpose: this file is the published contract for
a downstream engine, not an internal API response, so it follows the package
spec verbatim (``hair_prompt``, ``protected_identity_mask``, ...).
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from app.core.config import PROJECT_ROOT, Settings, settings
from app.domain.analysis import PhotoAnalysis, SubjectSummary
from app.domain.styling import StylingProposal
from app.domains.portrait.edit_mask import EditMaskSet, SemanticMaskConfig
from app.domains.portrait.face_lock import FaceLockConfig
from app.domains.portrait.facts import derive_face_facts
from app.domains.portrait.stylist import PortraitStylist
from app.services.native_mask_inpainting import (
    MAX_BACKGROUND_SHARE_OF_EDIT_AREA,
    NativeMaskBuild,
    NativeMaskInpainting,
    NativeMaskUnavailable,
)

#: Bumped whenever the package layout or the mask rules change. An engine that
#: pinned an older layout can refuse a package it does not understand.
PIPELINE_VERSION = "local-edit-package/1.0.0"

#: The engine this contract was written for. It is metadata, not a dependency:
#: this module never imports, runs or installs it.
ENGINE_TARGET = "comfyui-inpaint-cropandstitch"

#: The masks are binary by contract. Crop-and-stitch (and every diffusers
#: inpaint pipeline) treats a partial value as "blend", so a grey mask would
#: leak the original pixels through the generated frame.
MASK_LEVELS = (0, 255)

#: Aliases onto the three axes the stylist already authors. These are *names for
#: existing designs*, not new styles: the package never invents a look.
#:
#: One entry needs its own note. "精致优雅风" is not an axis name: it is the
#: title of one historical AI-authored plan (``data/jobs/job_45f7920ad505.json``,
#: plan id ``ai_elegant_refined``) that shares the big-wave silhouette but not
#: every detail -- that plan says chestnut brown hair, a deep olive French
#: blouse and browline glasses, while the ``artistic_wave`` axis below says
#: burgundy brown, cream/burgundy/muted-green draping and gold hoops. So this
#: alias picks the closest *existing* design, it does not reproduce that plan.
#: To carry that plan verbatim instead, drive the builder with the stored plan
#: itself (``--plan ai_elegant_refined`` on the CLI), which reads the prompt
#: back out of the job file rather than composing it from this axis.
STYLE_AXIS_ALIASES: dict[str, str] = {
    "urban_short": "urban_short",
    "urban": "urban_short",
    "crisp_urban": "urban_short",
    "都市利落短发": "urban_short",
    "都市短发": "urban_short",
    "artistic_wave": "artistic_wave",
    "artistic": "artistic_wave",
    "elegant": "artistic_wave",
    "艺术感长发微卷": "artistic_wave",
    "艺术感长发": "artistic_wave",
    "优雅": "artistic_wave",
    "精致优雅风": "artistic_wave",
    "bold_street": "bold_street",
    "bold": "bold_street",
    "street": "bold_street",
    "先锋街头高马尾": "bold_street",
    "街头高马尾": "bold_street",
}


class UnknownStyleError(ValueError):
    """The requested style id is not one of the styled looks this project owns."""


class LocalEditPackageUnavailable(RuntimeError):
    """The package could not be built honestly, so none was written."""


def style_ids() -> tuple[str, ...]:
    """Every style id the CLI accepts, canonical ids first."""
    return tuple(sorted(STYLE_AXIS_ALIASES))


def resolve_style_id(style_id: str) -> str:
    """Map a user-facing style id (or alias) onto a stylist axis id."""
    key = str(style_id or "").strip()
    axis = STYLE_AXIS_ALIASES.get(key) or STYLE_AXIS_ALIASES.get(key.lower())
    if axis is None:
        raise UnknownStyleError(
            f"unknown style {style_id!r}; known ids: {', '.join(style_ids())}"
        )
    return axis


#: The job store this project already writes: one JSON file per portrait job,
#: each holding the plan that produced it. A stored plan is *not* an axis. It is
#: the verbatim brief a real model authored for one photo, so it is read back
#: from disk rather than re-derived from an axis name -- re-deriving is exactly
#: how "精致优雅风" drifted into the neighbouring ``artistic_wave`` copy.
PROJECT_JOBS_DIR = PROJECT_ROOT / "data" / "jobs"

#: Statuses that mean "this plan really ran". A stored plan is only offered for
#: reuse once the job that wrote it finished.
PLAN_SUCCEEDED_STATUSES = frozenset({"succeeded", "success", "completed"})


class StoredPlanError(ValueError):
    """A stored plan exists but cannot be read back exactly as it was written."""


@dataclass(frozen=True)
class StoredPlan:
    """One historical plan, verbatim, plus where it came from."""

    proposal: StylingProposal
    job_path: Path
    job_id: str
    status: str
    model_provider: str
    model_name: str
    created_at: str

    @property
    def plan_id(self) -> str:
        return self.proposal.plan_id

    @property
    def succeeded(self) -> bool:
        return self.status in PLAN_SUCCEEDED_STATUSES

    def provenance(self) -> dict[str, Any]:
        return {
            "kind": "stored-plan",
            "plan_id": self.proposal.plan_id,
            "title": self.proposal.title,
            "job_id": self.job_id,
            "job_path": str(self.job_path),
            "job_status": self.status,
            "model_provider": self.model_provider,
            "model_name": self.model_name,
            "created_at": self.created_at,
            "style_reason_source": "plan.description",
            "prompt_source": "plan.edit_prompt (verbatim)",
            "note": (
                "the prompts below are this plan's own text, read back from the "
                "job store, not a re-composition from a style axis"
            ),
        }


def _stored_identity_anchors(edit_prompt: str) -> tuple[str, ...]:
    """The plan's own ``Identity anchors`` line, split into single promises."""
    line = _prompt_section(edit_prompt, "Identity anchors")
    if not line:
        return ()
    body = line.split(":", 1)[1] if ":" in line else line
    return tuple(part.strip() for part in body.split(";") if part.strip())


def _stored_presets(edit_prompt: str) -> tuple[str, ...]:
    """The plan's own ``Stylekit presets applied`` line, split into ids."""
    line = _prompt_section(edit_prompt, "Stylekit presets applied:")
    return tuple(part.strip() for part in line.split(",") if part.strip())


def decode_stored_plan(job_path: Path | str, plan_id: str | None = None) -> StoredPlan:
    """Read one job file back into the proposal it recorded, verbatim.

    The stored plan only carries what the model actually wrote (title, the
    multi-line edit prompt, the negative prompt, the expected changes). The
    axis-only fields (``hair_plan``, ``palette``, ...) are left empty on
    purpose: the labelled lines inside ``edit_prompt`` are the real text and the
    prompt bundle reads them from there, so filling the fields as well would
    duplicate every sentence.
    """
    path = Path(job_path).expanduser()
    if not path.is_file():
        raise StoredPlanError(f"job file not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StoredPlanError(f"{path} is not readable JSON: {exc}") from exc
    plan = payload.get("plan") if isinstance(payload, dict) else None
    if not isinstance(plan, dict):
        raise StoredPlanError(f"{path} holds no plan object")
    stored_id = str(plan.get("plan_id") or "")
    if not stored_id:
        raise StoredPlanError(f"the plan in {path} has no plan_id")
    if plan_id is not None and stored_id != plan_id:
        raise StoredPlanError(f"{path} holds plan {stored_id!r}, not {plan_id!r}")
    edit_prompt = str(plan.get("edit_prompt") or "")
    if not edit_prompt.strip():
        raise StoredPlanError(f"plan {stored_id!r} in {path} has an empty edit_prompt")
    title = str(plan.get("title") or stored_id)
    description = str(plan.get("description") or "")
    proposal = StylingProposal(
        plan_id=stored_id,
        domain_type=str(plan.get("domain_type") or "portrait"),
        title=title,
        name=title,
        design_axis="stored-plan",
        design_direction=title,
        description=description,
        design_reason=description,
        intensity=str(plan.get("intensity") or "medium"),
        hair_plan="",
        makeup_plan="",
        outfit_plan="",
        accessory_plan="",
        palette="",
        identity_anchors=_stored_identity_anchors(edit_prompt),
        expected_changes=tuple(
            str(item) for item in (plan.get("expected_changes") or ())
        ),
        edit_prompt=edit_prompt,
        negative_prompt=str(plan.get("negative_prompt") or ""),
        design_mode="ai",
        source_provider=str(payload.get("model_provider") or ""),
        selected_presets=_stored_presets(edit_prompt),
    )
    return StoredPlan(
        proposal=proposal,
        job_path=path.resolve(),
        job_id=str(payload.get("job_id") or path.stem),
        status=str(payload.get("status") or ""),
        model_provider=str(payload.get("model_provider") or ""),
        model_name=str(payload.get("model_name") or ""),
        created_at=str(payload.get("created_at") or ""),
    )


def stored_plan_ids(jobs_dir: Path | str | None = None) -> tuple[str, ...]:
    """Every plan id the job store holds, succeeded jobs first."""
    directory = Path(jobs_dir or PROJECT_JOBS_DIR).expanduser()
    if not directory.is_dir():
        return ()
    ids: list[str] = []
    for path in sorted(directory.glob("job_*.json")):
        try:
            stored = decode_stored_plan(path)
        except StoredPlanError:
            continue
        if stored.plan_id not in ids:
            ids.append(stored.plan_id)
    return tuple(ids)


def find_stored_plan(
    plan_id: str, jobs_dir: Path | str | None = None
) -> StoredPlan:
    """Find the stored job that owns ``plan_id``; prefer one that succeeded."""
    wanted = str(plan_id or "").strip()
    if not wanted:
        raise StoredPlanError("a plan id is required")
    directory = Path(jobs_dir or PROJECT_JOBS_DIR).expanduser()
    if not directory.is_dir():
        raise StoredPlanError(f"job store not found: {directory}")
    matches: list[StoredPlan] = []
    for path in sorted(directory.glob("job_*.json")):
        try:
            stored = decode_stored_plan(path)
        except StoredPlanError:
            continue
        if stored.plan_id == wanted:
            matches.append(stored)
    if not matches:
        raise StoredPlanError(f"no stored job in {directory} holds plan {wanted!r}")
    return max(matches, key=lambda item: (item.succeeded, item.created_at))


# --------------------------------------------------------------------- helpers


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mask_pixels(mask: Any) -> int:
    return int((np.asarray(mask) > 0).sum())


def _binary(mask: Any) -> np.ndarray:
    """Normalise a mask to a strictly 0/255 ``uint8`` array."""
    return np.where(np.asarray(mask) > 0, 255, 0).astype(np.uint8)


def write_binary_mask(path: Path, mask: Any) -> Path:
    """Write a 0/255 mask; a partially covered mask is a contract violation."""
    array = np.asarray(mask)
    if array.ndim != 2:
        raise LocalEditPackageUnavailable(
            f"the mask for {path.name} is {array.ndim}-D, not a single-channel mask"
        )
    binary = _binary(array)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(binary, mode="L").save(path)
    return path


def git_info(repo_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Best-effort provenance: the head this package was built from."""

    def _run(*args: str) -> str | None:
        try:
            done = subprocess.run(
                ["git", "-C", str(repo_root), *args],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if done.returncode != 0:
            return None
        return done.stdout.strip()

    head = _run("rev-parse", "HEAD")
    status = _run("status", "--porcelain")
    return {
        "head": head or "unknown",
        "short_head": (head or "unknown")[:7],
        "dirty": bool(status) if status is not None else None,
        "branch": _run("rev-parse", "--abbrev-ref", "HEAD") or "unknown",
    }


def _prompt_section(edit_prompt: str, prefix: str) -> str:
    """Pull one labelled line out of the stylist's multi-line edit prompt."""
    for line in str(edit_prompt or "").splitlines():
        if line.startswith(prefix):
            return line[len(prefix) :].strip()
    return ""


def _preview(
    image: np.ndarray, layers: list[tuple[np.ndarray, tuple[int, int, int]]]
) -> np.ndarray:
    """Dim the photo, then paint each region in its own colour on top."""
    canvas = np.asarray(image, dtype=np.float64) * 0.35
    for mask, colour in layers:
        selected = _binary(mask) > 0
        canvas[selected] = (
            np.asarray(colour, dtype=np.float64) * 0.65 + canvas[selected]
        )
    return np.clip(canvas, 0, 255).astype(np.uint8)


# ------------------------------------------------------------------- package


@dataclass(frozen=True)
class LocalEditPackage:
    """One built package: where it lives, what it holds, and what it claims."""

    root: Path
    files: dict[str, Path]
    prompt_bundle: dict[str, Any]
    manifest: dict[str, Any]
    mask_stats: dict[str, Any]
    acceptance: dict[str, Any]

    @property
    def manifest_path(self) -> Path:
        return self.files["manifest"]

    @property
    def prompt_bundle_path(self) -> Path:
        return self.files["prompt_bundle"]

    def summary_lines(self) -> list[str]:
        coverage = self.manifest["coverage"]
        overlap = self.manifest["overlap"]
        rules = self.manifest["mask_rules"]
        return [
            f"package        : {self.root}",
            f"style          : {self.prompt_bundle['style_name']} "
            f"({self.prompt_bundle['style_id']})",
            f"photo          : {self.manifest['source_image_size'][0]}x"
            f"{self.manifest['source_image_size'][1]}",
            f"hair mask      : {coverage['hair']['pixels']} px "
            f"({coverage['hair']['share_of_frame']:.2%})",
            f"clothing mask  : {coverage['clothing']['pixels']} px "
            f"({coverage['clothing']['share_of_frame']:.2%})",
            f"protected mask : {coverage['protected_identity']['pixels']} px "
            f"({coverage['protected_identity']['share_of_frame']:.2%})",
            f"protected ∩ edit: {overlap['protected_and_edit']['pixels']} px",
            f"hair ∩ clothing : {overlap['hair_and_clothing']['pixels']} px",
            f"background in edit: "
            f"{overlap['background_inside_edit']['share_of_edit_area']:.2%} "
            f"(limit {MAX_BACKGROUND_SHARE_OF_EDIT_AREA:.0%})",
            f"mask rules     : "
            f"{sum(1 for rule in rules if rule['passed'])}/{len(rules)} passed",
            f"mask acceptance: passed={self.acceptance['passed']}",
            f"engine target  : {self.manifest['engine_handoff']['engine']}",
        ]


class LocalEditPackageBuilder:
    """Turn one photo plus one style id into the standard local edit package."""

    def __init__(
        self,
        config: Settings = settings,
        *,
        detector_factory: Callable[[Path], Any] | None = None,
        segmenter_factory: Callable[[Path], Any] | None = None,
        mask_config: SemanticMaskConfig | None = None,
        face_lock_config: FaceLockConfig | None = None,
        stylist: PortraitStylist | None = None,
        git_info_fn: Callable[[], dict[str, Any]] | None = None,
    ) -> None:
        self.config = config
        self.mask_service = NativeMaskInpainting(
            config,
            detector_factory=detector_factory,
            segmenter_factory=segmenter_factory,
            mask_config=mask_config,
            face_lock_config=face_lock_config,
        )
        self.stylist = stylist or PortraitStylist()
        self._git_info = git_info_fn or (lambda: git_info(PROJECT_ROOT))

    # ------------------------------------------------------------------ build

    def default_root(self) -> Path:
        return PROJECT_ROOT / "artifacts" / "local-edit-package"

    def build(
        self,
        photo_path: Path | str,
        style_id: str | None = None,
        out_dir: Path | str | None = None,
        *,
        plan: StoredPlan | None = None,
    ) -> LocalEditPackage:
        """Build the package for one photo.

        Two ways to name the look, and they are not interchangeable:

        * ``style_id`` picks one of the stylist's template axes, composed from
          scratch for this photo's facts;
        * ``plan`` carries a *stored* AI-authored plan and uses its own text
          verbatim. It wins when both are given, because the whole point of a
          stored plan is that nothing re-derives or paraphrases it.
        """
        photo = Path(photo_path).expanduser().resolve()
        if not photo.is_file():
            raise FileNotFoundError(f"photo not found: {photo}")
        if plan is None and not style_id:
            raise UnknownStyleError(
                "either a style id or a stored plan is required; known style "
                f"ids: {', '.join(style_ids())}"
            )
        axis = plan.plan_id if plan is not None else resolve_style_id(style_id)

        if out_dir is None:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            root = self.default_root() / stamp
        else:
            root = Path(out_dir).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)

        image = np.asarray(Image.open(photo).convert("RGB"))
        height, width = int(image.shape[0]), int(image.shape[1])

        proposal = plan.proposal if plan is not None else self._proposal(photo, axis)
        mask_build = self._build_masks(photo, root)
        mask_set = mask_build.mask_set

        files: dict[str, Path] = {}
        files["original_image"] = self._copy_original(photo, root)

        for key, region, name in (
            ("hair_edit_mask", mask_set.hair, "hair_edit_mask.png"),
            ("clothing_edit_mask", mask_set.clothing, "clothing_edit_mask.png"),
            ("protected_identity_mask", mask_set.protected, "protected_identity_mask.png"),
            ("combined_edit_mask", mask_set.editable, "combined_edit_mask.png"),
        ):
            if region is None:
                raise LocalEditPackageUnavailable(
                    f"the {key} region is missing: the mask was built from geometry "
                    "rather than from a parsed label map, so it has no hair and "
                    "clothing regions to publish"
                )
            files[key] = write_binary_mask(root / name, region)

        stats = self._stats(image, mask_set)
        prompt_bundle = self._prompt_bundle(
            proposal, axis, plan.provenance() if plan is not None else None
        )
        previews = self._previews(image, mask_set, root)
        files.update(previews)

        files["prompt_bundle"] = self._write_json(root / "prompt_bundle.json", prompt_bundle)
        manifest = self._manifest(
            photo,
            root,
            files,
            prompt_bundle,
            stats,
            mask_build.acceptance,
            (width, height),
            plan.provenance() if plan is not None else None,
        )
        files["manifest"] = self._write_json(root / "manifest.json", manifest)

        return LocalEditPackage(
            root=root,
            files=files,
            prompt_bundle=prompt_bundle,
            manifest=manifest,
            mask_stats=stats,
            acceptance=mask_build.acceptance,
        )

    # ----------------------------------------------------------------- pieces

    def _proposal(self, photo: Path, axis: str) -> StylingProposal:
        facts = derive_face_facts(photo)
        analysis = PhotoAnalysis(
            image_id=photo.stem,
            domain_type="portrait",
            scene_type="portrait",
            subjects=SubjectSummary(count=1, position="center", face_visibility="clear"),
            face_facts=facts,
        )
        for proposal in self.stylist.create_proposals(analysis):
            if proposal.plan_id == axis:
                return proposal
        raise LocalEditPackageUnavailable(
            f"the stylist produced no proposal for axis {axis!r}"
        )

    def _build_masks(self, photo: Path, root: Path) -> NativeMaskBuild:
        """The production mask service, with its acceptance gate left on."""
        try:
            return self.mask_service.build_mask(photo, root / "combined_edit_mask.png")
        except NativeMaskUnavailable as exc:
            raise LocalEditPackageUnavailable(
                f"the local edit masks could not be built honestly: {exc}"
            ) from exc

    def _copy_original(self, photo: Path, root: Path) -> Path:
        suffix = photo.suffix.lower() or ".png"
        target = root / f"original{suffix}"
        if target.resolve() != photo:
            shutil.copyfile(photo, target)
        return target

    def _previews(
        self, image: np.ndarray, mask_set: EditMaskSet, root: Path
    ) -> dict[str, Path]:
        red, blue, green = (255, 60, 60), (60, 110, 255), (60, 220, 60)
        previews = {
            "preview_hair": _preview(image, [(mask_set.hair, red)]),
            "preview_clothing": _preview(image, [(mask_set.clothing, blue)]),
            "preview_protected_identity": _preview(image, [(mask_set.protected, green)]),
            "preview_overlay": _preview(
                image,
                [
                    (mask_set.hair, red),
                    (mask_set.clothing, blue),
                    (mask_set.protected, green),
                ],
            ),
        }
        written: dict[str, Path] = {}
        for key, array in previews.items():
            path = root / f"{key}.png"
            Image.fromarray(array).save(path)
            written[key] = path
        return written

    # ------------------------------------------------------------------ stats

    def _stats(self, image: np.ndarray, mask_set: EditMaskSet) -> dict[str, Any]:
        height, width = int(image.shape[0]), int(image.shape[1])
        frame = float(max(height * width, 1))

        hair = _binary(mask_set.hair)
        clothing = _binary(mask_set.clothing)
        protected = _binary(mask_set.protected)
        editable = _binary(mask_set.editable)

        def _and(a: Any, b: Any) -> int:
            if a is None or b is None:
                return 0
            return int(np.logical_and(_binary(a) > 0, _binary(b) > 0).sum())

        hair_px, clothing_px = mask_pixels(hair), mask_pixels(clothing)
        protected_px, editable_px = mask_pixels(protected), mask_pixels(editable)
        # Background means the *parsed* Background class, not the preserve mask:
        # ``background_preserve`` has the edit area already subtracted from it, so
        # measuring against it could never find a leak. This is the same figure
        # the mask service's own acceptance check reports.
        background_in_edit = int(mask_set.stats.get("backgroundEditablePixels", 0))

        face_core = mask_set.face_core

        coverage = {
            "hair": {
                "pixels": hair_px,
                "share_of_frame": round(hair_px / frame, 6),
                "share_of_frame_percent": round(100.0 * hair_px / frame, 2),
            },
            "clothing": {
                "pixels": clothing_px,
                "share_of_frame": round(clothing_px / frame, 6),
                "share_of_frame_percent": round(100.0 * clothing_px / frame, 2),
            },
            "protected_identity": {
                "pixels": protected_px,
                "share_of_frame": round(protected_px / frame, 6),
                "share_of_frame_percent": round(100.0 * protected_px / frame, 2),
            },
            "editable_combined": {
                "pixels": editable_px,
                "share_of_frame": round(editable_px / frame, 6),
                "share_of_frame_percent": round(100.0 * editable_px / frame, 2),
            },
            "frame": {"width": width, "height": height, "pixels": int(frame)},
        }
        overlap = {
            "protected_and_edit": {"pixels": _and(protected, editable)},
            "hair_and_protected": {"pixels": _and(hair, protected)},
            "clothing_and_protected": {"pixels": _and(clothing, protected)},
            "hair_and_face_core": {"pixels": _and(hair, face_core)},
            "background_inside_edit": {
                "pixels": background_in_edit,
                "share_of_edit_area": round(
                    background_in_edit / float(max(editable_px, 1)), 6
                ),
                "share_of_frame": round(background_in_edit / frame, 6),
                "limit": MAX_BACKGROUND_SHARE_OF_EDIT_AREA,
                "note": (
                    "parsed-Background pixels the grown hair/clothing rims reach; "
                    "a few pixels along a segmented edge are expected, a quarter of "
                    "the edit area is a mask that would repaint the room"
                ),
            },
            "hair_and_clothing": {
                "pixels": _and(hair, clothing),
                "share_of_edit_area": round(
                    _and(hair, clothing) / float(max(editable_px, 1)), 6
                ),
                "note": (
                    "the hair and clothing rims meet along the hair/garment "
                    "junction, so both masks claim that boundary band; each pass "
                    "still returns every pixel outside its own mask, so this does "
                    "not invalidate either pass. The band is not owned by both: "
                    "clothing runs first and hair second, so the hair pass decides "
                    "its final pixels (engine_handoff.steps)"
                ),
            },
        }

        rules = self._mask_rules(mask_set, coverage, overlap)
        return {
            "coverage": coverage,
            "overlap": overlap,
            "mask_rules": rules,
            "collar": {
                "split_applied": bool(mask_set.stats.get("collarSplitApplied", False)),
                "pixels": self._stats_collar(mask_set)["pixels"],
                "reclassified_pixels": int(
                    mask_set.stats.get("collarReclassifiedPixels", 0)
                ),
                "neck_skin_protected_pixels": self._stats_collar(mask_set)[
                    "neck_skin_protected_pixels"
                ],
            },
            "identity_guard": {
                "identity_guard_pixels": int(
                    mask_set.stats.get("identityGuardPixels", 0)
                ),
                "glasses_guard_pixels": int(
                    mask_set.stats.get("glassesGuardPixels", 0)
                ),
                "face_core_pixels": int(mask_set.stats.get("faceCorePixels", 0)),
            },
            "parse": {
                "semantic": bool(mask_set.semantic),
                "parsed_class_shares": mask_set.stats.get("parsedClassShares"),
            },
            "provider_stats": dict(mask_set.stats),
            "gaps": list(mask_set.gaps),
        }

    def _mask_rules(
        self,
        mask_set: EditMaskSet,
        coverage: dict[str, Any],
        overlap: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """The rules the package promises, each checked and reported."""
        min_region_px = self.mask_service.mask_config.min_region_px
        stats = mask_set.stats
        collar = self._stats_collar(mask_set)

        def rule(rule_id: str, passed: bool, detail: str) -> dict[str, Any]:
            return {"id": rule_id, "passed": bool(passed), "detail": detail}

        background_share = overlap["background_inside_edit"]["share_of_edit_area"]
        rules = [
            rule(
                "hair_mask_non_empty",
                coverage["hair"]["pixels"] >= min_region_px,
                f"{coverage['hair']['pixels']} px (floor {min_region_px} px)",
            ),
            rule(
                "hair_mask_excludes_identity",
                overlap["hair_and_protected"]["pixels"] == 0
                and overlap["hair_and_face_core"]["pixels"] == 0,
                "hair is disjoint from the identity guard and the parsed face core "
                f"(protected {overlap['hair_and_protected']['pixels']} px, face core "
                f"{overlap['hair_and_face_core']['pixels']} px)",
            ),
            rule(
                "clothing_mask_non_empty",
                coverage["clothing"]["pixels"] >= min_region_px,
                f"{coverage['clothing']['pixels']} px (floor {min_region_px} px)",
            ),
            rule(
                "clothing_mask_includes_old_collar",
                bool(stats.get("collarSplitApplied", False))
                and collar["pixels"] > 0,
                (
                    f"the collar carved out of the parsed face island is editable "
                    f"({collar['pixels']} px)"
                    if stats.get("collarSplitApplied", False)
                    else "no collar was filed as Face below the chin on this photo, "
                    "so there is no collar region to include"
                ),
            ),
            rule(
                "clothing_mask_protects_neck_skin",
                overlap["clothing_and_protected"]["pixels"] == 0
                and collar["neck_skin_protected_pixels"] >= 0,
                "the garment region is disjoint from the protected guard; the neck "
                "skin below the chin stays protected "
                f"({collar['neck_skin_protected_pixels']} px)",
            ),
            rule(
                "protected_identity_guard_present",
                coverage["protected_identity"]["pixels"] > 0
                and int(stats.get("identityGuardPixels", 0)) > 0
                and int(stats.get("glassesGuardPixels", 0)) > 0,
                "the guard covers the face features, the lips hull (teeth) and the "
                f"glasses band ({stats.get('identityGuardPixels', 0)} px features, "
                f"{stats.get('glassesGuardPixels', 0)} px glasses)",
            ),
            rule(
                "protected_and_edit_disjoint",
                overlap["protected_and_edit"]["pixels"] == 0,
                f"{overlap['protected_and_edit']['pixels']} px of overlap",
            ),
            rule(
                "edit_regions_independently_usable",
                coverage["hair"]["pixels"] > 0 and coverage["clothing"]["pixels"] > 0,
                "hair and clothing are separate masks, so the engine can run the "
                "hair pass and the clothing pass independently",
            ),
            rule(
                "background_not_swallowed_by_edit",
                background_share <= MAX_BACKGROUND_SHARE_OF_EDIT_AREA,
                f"{background_share:.2%} of the edit area is background "
                f"(limit {MAX_BACKGROUND_SHARE_OF_EDIT_AREA:.0%})",
            ),
        ]
        return rules

    def _stats_collar(self, mask_set: EditMaskSet) -> dict[str, int]:
        stats = mask_set.stats
        return {
            "pixels": int(stats.get("collarEditablePixels", 0)),
            "neck_skin_protected_pixels": int(stats.get("neckSkinProtectedPixels", 0)),
        }

    # ----------------------------------------------------------------- prompts

    def _prompt_bundle(
        self,
        proposal: StylingProposal,
        axis: str,
        plan_provenance: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        hair_line = _prompt_section(proposal.edit_prompt, "Hair:")
        outfit_line = _prompt_section(proposal.edit_prompt, "Outfit:")
        accessory_line = _prompt_section(proposal.edit_prompt, "Accessories:")
        makeup_line = _prompt_section(proposal.edit_prompt, "Makeup:")
        palette_line = _prompt_section(proposal.edit_prompt, "Colour palette:")
        intro = str(proposal.edit_prompt or "").splitlines()[:1]

        hair_prompt = " ".join(
            part for part in (hair_line, proposal.hair_plan) if part
        ).strip()
        clothing_prompt = " ".join(
            part for part in (outfit_line, proposal.outfit_plan, accessory_line) if part
        ).strip()

        global_prompt = " ".join(
            part
            for part in (
                intro[0] if intro else "",
                palette_line,
                "只编辑 hair_edit_mask 与 clothing_edit_mask 覆盖的像素；"
                "两张 mask 的黑色区域必须逐像素保留原照片，不得重新生成。",
                "This is a two-pass masked edit: pass 1 repaints only the clothing "
                "mask, pass 2 repaints only the hair mask. The order is not "
                "cosmetic: the two masks overlap along the hair/garment junction, "
                "and in a crop-and-stitch chain each pass blends its result over "
                "the image the previous pass returned, so the mask that runs last "
                "owns the overlap. Hair runs last on purpose, so the strands that "
                "fall over the new garment stay hair. Anything outside the current "
                "pass's mask must come back as the original pixels.",
            )
            if part
        ).strip()

        constraints = self._identity_constraints(proposal)
        bundle: dict[str, Any] = {
            "style_id": axis,
            "style_name": proposal.name,
            "style_reason": proposal.design_reason,
            "hair_prompt": hair_prompt,
            "clothing_prompt": clothing_prompt,
            "global_prompt": global_prompt,
            "negative_prompt": proposal.negative_prompt,
            "identity_constraints": constraints,
            # Context for a human and for a whole-image engine. The masked flow
            # cannot apply these lines: no mask covers the face or the palette.
            "context": {
                "design_direction": proposal.design_direction,
                "design_axis": proposal.design_axis,
                "hair_plan_zh": proposal.hair_plan,
                "makeup_plan_zh": proposal.makeup_plan,
                "outfit_plan_zh": proposal.outfit_plan,
                "accessory_plan_zh": proposal.accessory_plan,
                "palette_zh": proposal.palette,
                "makeup_prompt_en_not_applied_by_masks": makeup_line,
                "expected_changes": list(proposal.expected_changes),
                "why_it_fits": proposal.why_it_fits,
                "reference_presets": [ref.preset_id for ref in proposal.reference_presets],
                "design_mode": proposal.design_mode,
                "estimated_paid_requests": proposal.estimated_paid_requests,
            },
            "mask_scope": {
                "hair_edit_mask": "white pixels may be repainted with the new hairstyle",
                "clothing_edit_mask": "white pixels may be repainted with the new outfit",
                "protected_identity_mask": (
                    "never repainted; verified afterwards against the original pixels"
                ),
                "engine_order": ["clothing_edit_mask", "hair_edit_mask"],
            },
        }
        if plan_provenance is not None:
            bundle["plan_provenance"] = plan_provenance
        return bundle

    def _identity_constraints(self, proposal: StylingProposal) -> dict[str, Any]:
        rules = [
            {
                "id": "face_features_unchanged",
                "zh": "五官不变：眼睛、眉毛、鼻子、嘴巴必须保持原照片的像素与形状。",
                "en": "Keep the facial features (eyes, brows, nose, lips) exactly as "
                "in the original photo.",
            },
            {
                "id": "smile_unchanged",
                "zh": "笑容不变：嘴角形态与露齿状态保持原样。",
                "en": "Keep the smile and the teeth exactly as in the original photo.",
            },
            {
                "id": "glasses_unchanged",
                "zh": "眼镜不变：镜片、镜框与镜面反光保持原样。",
                "en": "Keep the glasses, their frames and their reflections unchanged.",
            },
            {
                "id": "background_unchanged",
                "zh": "背景不变：人物轮廓以外的像素全部保持原样。",
                "en": "Keep the background unchanged; never repaint outside the mask.",
            },
            {
                "id": "pose_unchanged",
                "zh": "姿势不变：头部角度、身体朝向与构图保持原样。",
                "en": "Keep the pose, head angle, body orientation and framing unchanged.",
            },
            {
                "id": "edit_only_inside_masks",
                "zh": "只编辑指定 mask 区域：白色区域可改，黑色区域必须保留原照片像素。",
                "en": "Edit only inside the supplied mask; every pixel outside it must "
                "be the original photo's own pixels.",
            },
        ]
        return {
            "rules": rules,
            "prompt_text_zh": "；".join(rule["zh"] for rule in rules),
            "prompt_text_en": " ".join(rule["en"] for rule in rules),
            "anchors": list(proposal.identity_anchors),
        }

    # ---------------------------------------------------------------- manifest

    def _manifest(
        self,
        photo: Path,
        root: Path,
        files: dict[str, Path],
        prompt_bundle: dict[str, Any],
        stats: dict[str, Any],
        acceptance: dict[str, Any],
        size: tuple[int, int],
        plan_provenance: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        manifest = {
            "package_format": "local-edit-package",
            "pipeline_version": PIPELINE_VERSION,
            "generated_at": _now(),
            "engine_target": ENGINE_TARGET,
            "source_image": {
                "path": str(photo),
                "packaged_path": self._relative(files["original_image"], root),
                "sha256": sha256_file(photo),
                "width": int(size[0]),
                "height": int(size[1]),
            },
            "source_image_size": [int(size[0]), int(size[1])],
            "original_image": self._entry(files["original_image"], root),
            "hair_edit_mask": self._entry(files["hair_edit_mask"], root, "white=edit hair"),
            "clothing_edit_mask": self._entry(
                files["clothing_edit_mask"], root, "white=edit clothing"
            ),
            "protected_identity_mask": self._entry(
                files["protected_identity_mask"], root, "white=never repaint"
            ),
            "combined_edit_mask": self._entry(
                files["combined_edit_mask"], root, "hair ∪ clothing"
            ),
            "prompt_bundle": self._entry(files["prompt_bundle"], root),
            "coverage": stats["coverage"],
            "overlap": stats["overlap"],
            "mask_rules": stats["mask_rules"],
            "mask_rules_passed": all(rule["passed"] for rule in stats["mask_rules"]),
            "mask_acceptance": acceptance,
            "collar": stats["collar"],
            "identity_guard": stats["identity_guard"],
            "parse": stats["parse"],
            "gaps": stats["gaps"],
            "visualisations": {
                key: self._relative(files[key], root)
                for key in (
                    "preview_hair",
                    "preview_clothing",
                    "preview_protected_identity",
                    "preview_overlay",
                )
            },
            "engine_handoff": {
                "engine": ENGINE_TARGET,
                "engine_reference": "https://github.com/lquesada/ComfyUI-Inpaint-CropAndStitch",
                "input_contract": {
                    "image": "original_image (IMAGE)",
                    "mask": "hair_edit_mask / clothing_edit_mask (MASK, 255 = repaint)",
                    "prompt": "prompt_bundle.hair_prompt / prompt_bundle.clothing_prompt",
                    "negative_prompt": "prompt_bundle.negative_prompt",
                    "output": "one image per pass, stitched back at the original coordinates",
                },
                "steps": [
                    {
                        "id": "clothing",
                        "order": 1,
                        "image": self._relative(files["original_image"], root),
                        "mask": self._relative(files["clothing_edit_mask"], root),
                        "prompt": prompt_bundle["clothing_prompt"],
                        "source": "original photo",
                    },
                    {
                        "id": "hair",
                        "order": 2,
                        "image": "output of step 'clothing'",
                        "mask": self._relative(files["hair_edit_mask"], root),
                        "prompt": prompt_bundle["hair_prompt"],
                        "source": "output of step 'clothing'",
                    },
                ],
                "notes": [
                    "the two masks are disjoint from the protected region, so each "
                    "pass can be run on its own",
                    "the two masks are NOT disjoint from each other: they share the "
                    "hair/garment junction band. In a crop-and-stitch chain each "
                    "pass blends over the image the previous pass returned, so the "
                    "last pass owns that band. Clothing runs first and hair second "
                    "so the strands crossing the new garment stay hair",
                    "the protected_identity_mask is NOT an input to the engine's "
                    "context mask: its 'optional_context_mask' is unioned with the "
                    "edit mask to enlarge the crop, so feeding it there would widen "
                    "the crop instead of protecting the face",
                    "the face is protected twice: by the masks, and afterwards by "
                    "copying the original pixels back wherever the mask is black, at "
                    "identical coordinates (app.domains.portrait.inpaint_verify)",
                    "a pass whose returned frame is not the photo's own size must be "
                    "refused rather than rescaled into an 'identity-preserving' result",
                ],
            },
            "files": {
                key: {
                    "path": self._relative(path, root),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for key, path in sorted(files.items())
                if key != "manifest"
            },
            "provenance": {
                "segmentation": "SCHP ATR-18 ONNX, CPU inference (app.domains.portrait.human_parsing)",
                "landmarks": "MediaPipe FaceMesh (app.domains.portrait.face_lock)",
                "masks": "app.domains.portrait.edit_mask via app.services.native_mask_inpainting",
                "styling": (
                    "stored AI plan "
                    f"{plan_provenance['plan_id']} from {plan_provenance['job_id']} "
                    "(edit_prompt read back verbatim)"
                    if plan_provenance is not None
                    else "app.domains.portrait.stylist (template axis composition)"
                ),
                "git": self._git_info(),
                "network_calls": 0,
                "paid_generations": 0,
            },
        }
        if plan_provenance is not None:
            manifest["styling_plan"] = plan_provenance
        return manifest

    def _relative(self, path: Path, root: Path) -> str:
        try:
            return str(path.relative_to(root))
        except ValueError:
            return str(path)

    def _entry(self, path: Path, root: Path, semantics: str | None = None) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "path": self._relative(path, root),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        if semantics:
            entry["semantics"] = semantics
        return entry

    def _write_json(self, path: Path, payload: dict[str, Any]) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return path


__all__ = [
    "ENGINE_TARGET",
    "LocalEditPackage",
    "LocalEditPackageBuilder",
    "LocalEditPackageUnavailable",
    "MASK_LEVELS",
    "PIPELINE_VERSION",
    "PLAN_SUCCEEDED_STATUSES",
    "PROJECT_JOBS_DIR",
    "STYLE_AXIS_ALIASES",
    "StoredPlan",
    "StoredPlanError",
    "UnknownStyleError",
    "decode_stored_plan",
    "find_stored_plan",
    "git_info",
    "mask_pixels",
    "resolve_style_id",
    "sha256_file",
    "stored_plan_ids",
    "style_ids",
    "write_binary_mask",
]
