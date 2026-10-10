"""Free end-to-end acceptance walk of the Beauty Lab MVP against a live server.

Runs the exact flow the UI runs, but pins the *free* providers so it can never
create a paid request:

    X-Agent-Provider: local   (rule-based planner brain, no API key)
    X-Action-Provider: mock   (Pillow local retoucher, no billing)

Steps mirror the UI: upload -> analyze -> template proposals -> create job ->
poll -> list history. Prints a PASS/FAIL summary per step.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8000"
PHOTO = Path("data/inputs/IMG_3651.jpg")
FREE_HEADERS = {"X-Agent-Provider": "local", "X-Action-Provider": "mock"}

FAILURES: list[str] = []


def step(label: str, ok: bool, detail: str = "") -> None:
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] {label}{(' — ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(label)


def post_json(path: str, payload: dict, headers: dict | None = None):
    data = json.dumps(payload).encode()
    hdrs = {"Content-Type": "application/json", **(headers or {})}
    req = urllib.request.Request(BASE + path, data=data, headers=hdrs, method="POST")
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.status, json.loads(resp.read().decode())


def get_json(path: str):
    req = urllib.request.Request(BASE + path, method="GET")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.status, json.loads(resp.read().decode())


def upload(path: Path):
    boundary = "----beautylab" + str(int(time.time()))
    body = b""
    body += f"--{boundary}\r\n".encode()
    body += (
        f'Content-Disposition: form-data; name="file"; filename="{path.name}"\r\n'
        "Content-Type: image/jpeg\r\n\r\n"
    ).encode()
    body += path.read_bytes()
    body += f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        BASE + "/api/photos/upload",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.status, json.loads(resp.read().decode())


def main() -> int:
    print("== AI Beauty Style Lab — free mock end-to-end ==")
    print(f"photo: {PHOTO}")

    status, asset = upload(PHOTO)
    image_id = asset.get("imageId") or asset.get("id")
    step("1 upload photo", status == 200 and bool(image_id), f"imageId={image_id}")

    status, analysis = post_json(
        "/api/photos/analyze",
        {"imageId": image_id, "domainType": "portrait"},
        FREE_HEADERS,
    )
    step(
        "2 analyze photo (brain=local)",
        status == 200 and analysis.get("domainType") == "portrait",
        f"scene={analysis.get('sceneType')}",
    )

    status, proposals = post_json(
        "/api/styling/proposals",
        {"analysis": analysis, "mode": "template"},
        {"X-Agent-Provider": "local"},
    )
    plan_ids = [p.get("planId") for p in proposals] if isinstance(proposals, list) else []
    modes = {p.get("designMode") for p in proposals} if isinstance(proposals, list) else set()
    prompts = {p.get("planId"): bool(p.get("editPrompt")) for p in proposals} if isinstance(proposals, list) else {}
    step(
        "3 template proposals (3 looks, free)",
        status == 200 and len(proposals) == 3 and modes == {"template"},
        f"plans={plan_ids} designModes={sorted(modes)}",
    )
    step(
        "3b each look carries its own editPrompt",
        all(prompts.values()) and len(prompts) == 3,
        f"unique prompts={len(set(p.get('editPrompt') for p in proposals))}",
    )

    chosen = proposals[1]
    print(f"   -> selecting plan: {chosen.get('planId')} ({chosen.get('title')})")

    status, created = post_json(
        "/api/retouch/jobs",
        {
            "sourceImageId": image_id,
            "plan": chosen,
            "userInstruction": "",
        },
        FREE_HEADERS,
    )
    job_id = created.get("jobId")
    step("4 create job (action=mock)", status == 202 and bool(job_id), f"jobId={job_id}")

    job = {}
    deadline = time.time() + 300
    stages: list[str] = []
    while time.time() < deadline:
        _, job = get_json(f"/api/retouch/jobs/{job_id}")
        st = job.get("stage") or job.get("status")
        if not stages or stages[-1] != st:
            stages.append(st)
        if job.get("status") in ("succeeded", "failed"):
            break
        time.sleep(1.0)

    step(
        "5 job reached a terminal status",
        job.get("status") in ("succeeded", "failed"),
        f"status={job.get('status')} stage={job.get('stage')}",
    )
    print(f"   -> observed stages: {stages}")
    step(
        "6 job succeeded with an output image",
        job.get("status") == "succeeded" and bool(job.get("outputUrls")),
        f"verificationLevel={job.get('verificationLevel')} paidRequests={job.get('paidRequests')}",
    )
    step(
        "7 chosen plan is stored on the job (no hard-coded style)",
        job.get("planId") == chosen.get("planId"),
        f"job.planId={job.get('planId')}",
    )
    step(
        "8 mock run billed 0 paid requests",
        job.get("paidRequests") in (0, None),
        f"paidRequests={job.get('paidRequests')}",
    )

    status, history = get_json("/api/retouch/jobs?limit=5")
    done = [j for j in history if j.get("outputUrls")] if isinstance(history, list) else []
    step(
        "9 history endpoint lists finished results",
        status == 200 and any(j.get("jobId") == job_id for j in history),
        f"total={len(history)} withOutput={len(done)}",
    )

    if job.get("outputUrls"):
        print(f"   -> output url: {job['outputUrls'][0]}")

    print()
    if FAILURES:
        print(f"RESULT: FAIL ({len(FAILURES)} step(s)) -> {FAILURES}")
        return 1
    print("RESULT: ALL FREE STEPS PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
