"""Thin CLI over :mod:`app.infrastructure.runninghub_crop_stitch`.

The graph, the prompt builder and every gate check live in the shared module so
the web app and this CLI cannot drift apart. This file only keeps the command
line usable: it re-exports the pieces other tools import (``tools/…`` scripts
and ``evidence/…`` scripts import this module by name) and drives one run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import httpx

from app.infrastructure.runninghub_crop_stitch import (
    BASE_URL,
    CHECKPOINT,
    COLLAR_CLAUSE,
    COLLAR_NEGATIVE_TERMS,
    CREATE_PATH,
    CROP_DEFAULTS,
    CURRENT_HAIR_CLAUSE,
    DEFAULT_MASK_RULE,
    DENOISE_BAND,
    EXPECTED_STYLE_ID,
    FORBIDDEN_POSITIVE_PATTERN,
    GLASSES_CLAUSE,
    MASK_RULE_ID,
    NECKLACE_CLAUSE,
    NEGATIVE_PROMPT,
    OUTPUTS_PATH,
    PASS_DENOISE,
    PROTECTED_CORE_ERODE_PX,
    REQUIRED_NEGATIVE_TERMS,
    SAMPLER,
    STATUS_PATH,
    TUNED_STYLE_ID,
    UPLOAD_PATH,
    RunningHubError,
    _crop_box,
    _mask_pipeline_np,
    _post,
    _read_key,
    _workflow_diff,
    build_prompts,
    build_workflow,
    collect_gate_checks,
    create_task,
    download,
    extract_plan_targets,
    fetch_outputs,
    pick_image_output,
    poll_task,
    run_task,
    task_status,
    upload,
    validate_checkpoint,
    validate_workflow,
)

__all__ = [
    "BASE_URL",
    "CHECKPOINT",
    "COLLAR_CLAUSE",
    "COLLAR_NEGATIVE_TERMS",
    "CREATE_PATH",
    "CROP_DEFAULTS",
    "CURRENT_HAIR_CLAUSE",
    "DEFAULT_MASK_RULE",
    "DENOISE_BAND",
    "EXPECTED_STYLE_ID",
    "FORBIDDEN_POSITIVE_PATTERN",
    "GLASSES_CLAUSE",
    "MASK_RULE_ID",
    "NECKLACE_CLAUSE",
    "NEGATIVE_PROMPT",
    "OUTPUTS_PATH",
    "PASS_DENOISE",
    "PROTECTED_CORE_ERODE_PX",
    "REQUIRED_NEGATIVE_TERMS",
    "RunningHubError",
    "SAMPLER",
    "STATUS_PATH",
    "TUNED_STYLE_ID",
    "UPLOAD_PATH",
    "_crop_box",
    "_mask_pipeline_np",
    "_post",
    "_read_key",
    "_workflow_diff",
    "build_prompts",
    "build_workflow",
    "collect_gate_checks",
    "create_task",
    "download",
    "extract_plan_targets",
    "fetch_outputs",
    "main",
    "pick_image_output",
    "poll_task",
    "run_task",
    "task_status",
    "upload",
    "validate_checkpoint",
    "validate_workflow",
]

STAGED_INPUTS = {
    "original": "original.jpg",
    "clothing_mask": "clothing_edit_mask.png",
    "hair_mask": "hair_edit_mask.png",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--package", type=Path, required=True, help="local-edit-package directory")
    parser.add_argument("--out", type=Path, default=Path("artifacts/p29-fix-run"))
    parser.add_argument("--compare-to", type=Path, default=Path("artifacts/p29-run/workflow.json"),
                        help="previous workflow to diff the node/connection changes against")
    parser.add_argument("--key-file", type=Path, default=Path.home() / ".runninghub_key")
    parser.add_argument("--object-info", type=Path, default=Path("evidence/p29-live-run/object_info_live.json"))
    parser.add_argument("--seed", type=int, default=20261008)
    parser.add_argument("--instance-type", default="default")
    # ``workflowId`` is a required (non-null) field on /task/openapi/create even
    # when the graph is supplied inline via ``workflow``; the API documents that
    # ``workflow`` takes precedence over it. Without it the call is rejected with
    # code=301 "must not be null" before any task is created.
    parser.add_argument("--workflow-id", default="1")
    parser.add_argument("--poll-seconds", type=int, default=900)
    parser.add_argument("--dry-run", action="store_true", help="build + validate only (default)")
    parser.add_argument("--upload", action="store_true", help="upload the three input files")
    parser.add_argument("--submit", action="store_true", help="create the one paid task")
    parser.add_argument("--i-understand-this-costs-money", action="store_true")
    args = parser.parse_args(argv)

    package = args.package
    bundle = json.loads((package / "prompt_bundle.json").read_text())
    prompts = build_prompts(bundle)
    for key, label in (("glasses_clause_dropped", GLASSES_CLAUSE),
                       ("necklace_clause_dropped", NECKLACE_CLAUSE),
                       ("current_hair_clause_dropped", CURRENT_HAIR_CLAUSE)):
        if not prompts[key]:
            print(f"WARNING: clause {label!r} not found in the plan prompt; nothing dropped",
                  file=sys.stderr)

    for name in STAGED_INPUTS.values():
        if not (package / name).is_file():
            raise SystemExit(f"package is missing {name}")

    args.out.mkdir(parents=True, exist_ok=True)
    workflow = build_workflow(package, {k: f"__UPLOAD__{k}" for k in STAGED_INPUTS}, prompts, args.seed)

    object_info = None
    if args.object_info.is_file():
        object_info = json.loads(args.object_info.read_text())
        print(f"validating against {args.object_info} ({len(object_info)} installed nodes)")
    else:
        print(f"WARNING: {args.object_info} not found; structural checks only", file=sys.stderr)
    errors = validate_workflow(workflow, object_info)
    if errors:
        for err in errors:
            print(f"  INVALID  {err}", file=sys.stderr)
        return 1
    print(f"workflow OK: {len(workflow)} nodes against the live node table")

    gate_errors, gate = collect_gate_checks(workflow, object_info, prompts, bundle, package)
    gate["validate_workflow_errors"] = errors
    if args.compare_to.is_file():
        gate["workflow_diff_vs"] = str(args.compare_to)
        gate["workflow_diff"] = _workflow_diff(json.loads(args.compare_to.read_text()), workflow)
    (args.out / "gate_report.json").write_text(json.dumps(gate, ensure_ascii=False, indent=2))
    print(f"style {gate['style_id']} ({gate['style_name']}), denoise {gate['denoise']}")
    for note in gate["checkpoint_notes"]:
        print(f"  checkpoint: {note}")
    for kind, geom in gate.get("mask_geometry", {}).items():
        print(f"  {kind:8} painted={geom['painted_pixels']:>7} ring={geom['soft_ring_pixels']:>6} "
              f"over_protected={geom['painted_over_protected']:>5} "
              f"over_facial_core={geom['painted_over_facial_core']} "
              f"crop={geom['crop']['w']}x{geom['crop']['h']} -> "
              f"{geom['crop']['target_w']}x{geom['crop']['target_h']} "
              f"(scale {geom['scale_x']},{geom['scale_y']})")
    if gate.get("workflow_diff"):
        print("  workflow diff vs previous run:")
        for line in gate["workflow_diff"]:
            print(f"    {line}")
    if gate_errors:
        for err in gate_errors:
            print(f"  GATE FAILED  {err}", file=sys.stderr)
        print(f"gate report -> {args.out / 'gate_report.json'}", file=sys.stderr)
        return 1
    print(f"gate passed -> {args.out / 'gate_report.json'}")

    if not (args.upload or args.submit):
        (args.out / "workflow.json").write_text(json.dumps(workflow, indent=2))
        print(f"dry run complete -> {args.out / 'workflow.json'}")
        return 0

    key = _read_key(args.key_file)
    client = httpx.Client(base_url=BASE_URL, timeout=60.0)
    print("uploading inputs")
    filenames = {k: upload(client, key, package / v) for k, v in STAGED_INPUTS.items()}
    workflow = build_workflow(package, filenames, prompts, args.seed)
    errors = validate_workflow(workflow, object_info)
    if errors:
        for err in errors:
            print(f"  INVALID  {err}", file=sys.stderr)
        return 1
    (args.out / "workflow.json").write_text(json.dumps(workflow, indent=2))
    print(f"workflow written -> {args.out / 'workflow.json'}")

    if not args.submit:
        print("upload stage complete; no task created")
        return 0
    if not args.i_understand_this_costs_money:
        raise SystemExit("--submit needs --i-understand-this-costs-money (this creates one billed task)")

    created = create_task(
        client, key, workflow, workflow_id=args.workflow_id, instance_type=args.instance_type
    )
    task_id = created["data"]["taskId"]
    print(f"task created: {task_id} status={created['data'].get('taskStatus')}")
    if created["data"].get("promptTips"):
        print(f"promptTips: {created['data']['promptTips']}")
    (args.out / "task.json").write_text(json.dumps({"taskId": task_id, "create": created}, indent=2))

    status = poll_task(
        client, key, task_id,
        timeout_seconds=args.poll_seconds,
        on_status=lambda s: print(f"  {s}"),
    )
    if status != "SUCCESS":
        print(f"task did not succeed (status={status}); no result downloaded", file=sys.stderr)
        return 1

    outputs = fetch_outputs(client, key, task_id)
    (args.out / "outputs.json").write_text(json.dumps(outputs, indent=2))
    for index, item in enumerate(outputs):
        url = item.get("fileUrl")
        if not url:
            continue
        suffix = Path(url.split("?")[0]).suffix or ".png"
        target = args.out / f"result_{index}{suffix}"
        target.write_bytes(download(client, url))
        print(f"  result -> {target} ({target.stat().st_size} B) consumeCoins={item.get('consumeCoins')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
