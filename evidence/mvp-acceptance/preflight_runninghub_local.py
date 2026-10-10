"""FREE, offline preflight for the real RunningHub crop-and-stitch path.

Runs every step the provider performs *before* its single billed call:
the local SCHP + MediaPipe mask build, the prompt build from the user-selected
styling proposal, the workflow build, the ComfyUI node validation and the
preflight gate. Nothing is uploaded and no task is created -> zero paid calls.

    .venv/bin/python evidence/mvp-acceptance/preflight_runninghub_local.py
"""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "backend"))

API = "http://127.0.0.1:8000/api"
PHOTO = REPO / "data/inputs/IMG_3651.jpg"

checks: list[tuple[str, bool, str]] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    checks.append((label, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {label}{' - ' + detail if detail else ''}")


def _post(path: str, payload: dict, headers: dict | None = None):
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        API + path,
        data=body,
        headers={"Content-Type": "application/json", **(headers or {})},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _upload(path: Path) -> dict:
    boundary = "----preflightboundary"
    payload = path.read_bytes()
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{path.name}"\r\n'
        "Content-Type: image/jpeg\r\n\r\n"
    ).encode("utf-8") + payload + f"\r\n--{boundary}--\r\n".encode("utf-8")
    req = urllib.request.Request(
        API + "/photos/upload",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> int:
    from app.core.config import settings
    from app.infrastructure import runninghub_crop_stitch as rh
    from app.providers import runninghub_crop_stitch_provider as rh_provider
    from app.schemas.retouch import RetouchPlan
    from app.services.native_mask_inpainting import NativeMaskInpainting

    photo = _upload(PHOTO)
    check("1 upload original photo", bool(photo.get("imageId")), f"id={photo.get('imageId')}")

    analysis = _post(
        "/photos/analyze",
        {"imageId": photo["imageId"], "domainType": "auto"},
        {"X-Agent-Provider": "local", "X-Action-Provider": "mock"},
    )
    facts = analysis.get("faceFacts") or analysis.get("face_facts") or {}
    check(
        "2 free local analysis (0 model calls)",
        bool(facts),
        f"faces={facts.get('faceCount')} domain={analysis.get('domainType') or analysis.get('domain_type')}",
    )

    proposals = _post("/styling/proposals", {"analysis": analysis, "mode": "template"})
    check(
        "3 three free template looks",
        isinstance(proposals, list) and len(proposals) == 3,
        ", ".join(p.get("planId", "?") for p in proposals),
    )

    proposal = proposals[1]
    plan = RetouchPlan.model_validate(proposal)
    check("4 selected proposal carries a real editPrompt", bool(plan.edit_prompt), plan.plan_id)

    work = REPO / "evidence" / "mvp-acceptance" / "preflight-work"
    (work / "package").mkdir(parents=True, exist_ok=True)
    build = NativeMaskInpainting().build_mask(PHOTO, work / "edit_mask.png")
    stats = build.stats
    check(
        "5 SCHP + MediaPipe mask build",
        build.mask_set.hair is not None and build.mask_set.clothing is not None,
        f"editableShare={stats.get('editableShare')} collarSplit={stats.get('collarSplitApplied')} "
        f"gaps={list(build.mask_set.gaps)}",
    )

    provider = rh_provider.RunningHubCropStitchProvider(
        None,
        workflow_id=settings.runninghub_workflow_id,
        instance_type=settings.runninghub_instance_type,
        seed=settings.runninghub_seed,
        checkpoint=settings.runninghub_checkpoint,
        filename_prefix="preflight",
        object_info_path=settings.runninghub_object_info_path,
    )
    bundle, prompts = provider._build_prompts(plan, plan.plan_id)
    check(
        "6 prompts built from the SELECTED proposal (no template swap)",
        prompts["style_id"] == plan.plan_id,
        f"style_id={prompts['style_id']}",
    )
    check(
        "7 hair prompt reflects the plan's hair target",
        bool(prompts["hair"]) and prompts["hair"] != prompts["clothing"],
        prompts["hair"].split(",")[0][:80],
    )
    check(
        "8 clothing prompt reflects the plan's outfit target",
        bool(prompts["clothing"]),
        prompts["clothing"].split(",")[0][:80],
    )

    frame = provider._write_package(work / "package", PHOTO, build.mask_set)
    package = work / "package"
    workflow = rh.build_workflow(
        package,
        {kind: f"__UPLOAD__{kind}" for kind in rh_provider.STAGED_INPUTS},
        prompts,
        settings.runninghub_seed,
        filename_prefix="preflight",
        checkpoint=settings.runninghub_checkpoint,
    )
    object_info = rh_provider._load_object_info(provider._object_info_path)
    errors = rh.validate_workflow(workflow, object_info)
    check("9 workflow validated against the live node table", not errors,
          f"objectInfo={'yes' if object_info else 'structural-only'} errors={errors}")

    gate_errors, gate = rh.collect_gate_checks(
        workflow,
        object_info,
        prompts,
        bundle,
        package,
        expected_style_id=plan.plan_id,
        checkpoint=settings.runninghub_checkpoint,
    )
    check(
        "10 preflight gate accepts the selected plan",
        not gate_errors,
        "; ".join(gate_errors) if gate_errors else f"denoise={gate.get('denoise')} frame={frame[0]}x{frame[1]}",
    )
    check(
        "11 no billed request was made this run",
        provider.paid_requests == 0 and provider.last_task_id is None,
        f"paidRequests={provider.paid_requests} taskId={provider.last_task_id}",
    )

    (REPO / "evidence" / "mvp-acceptance" / "preflight-runninghub-local.json").write_text(
        json.dumps(
            {
                "photoId": photo.get("imageId"),
                "planId": plan.plan_id,
                "hairPrompt": prompts["hair"],
                "clothingPrompt": prompts["clothing"],
                "negativePrompt": prompts["negative"],
                "gate": gate,
                "gateErrors": gate_errors,
                "validateErrors": errors,
                "paidRequests": provider.paid_requests,
                "checks": [{"label": l, "ok": o, "detail": d} for l, o, d in checks],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    failed = [c for c in checks if not c[1]]
    print("\nRESULT:", "ALL FREE PREFLIGHT STEPS PASS" if not failed else f"{len(failed)} STEP(S) FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
