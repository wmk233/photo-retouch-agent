# P2.9 last paid run -- acceptance artefacts (saved offline)

Round scope: freeze the evidence of the **one** real RunningHub task the last
round submitted, with its charged amount verified against the account balance,
and commit only safe metadata, the acceptance report and the measuring scripts.
**No new paid task this round: 0.** No photo of a person, no API key and no
credential are committed.

Base revision: `16c41b6`. Photo: `data/inputs/IMG_3651.jpg` (904x1190).
Plan: `ai_elegant_refined` (`job_45f7920ad505.json`), verbatim.
Checkpoint: `juggernautXL_juggXIByRundiffusion.safetensors`, seed `20261008`.
Package submitted: `artifacts/local-edit-package/20261009T-neckfix`
(hair 187,091 / clothing 302,487 / protected 75,762 px).

## A. One task, two stage images

`task.json` records a single task, `2108330225116082177`. `outputs.json` holds
two entries, but they are **not two tasks**: they are the two `SaveImage` stage
records of the same graph --

| node | stage | file |
|---|---|---|
| `20` | clothing-pass stitch (`p29_elegant_clothing_pass_*`) | `clothing_pass.png` |
| `19` | final stitch (`p29_elegant_*`) | `final.png` |

Each entry carries `taskCostTime "34"` and `consumeMoney "0.038"`.

## B. What was actually charged

The two `consumeMoney` fields must **not** be added: they are two line items of
one task and each is rounded to three decimals, so `0.038 + 0.038` invents a
milli-yuan that was never billed. The charge is verified against the account
instead.

    POST https://www.runninghub.cn/uc/openapi/accountStatus   (read-only)
    before this task:  remainMoney 19.955 CNY
    after  this task:  remainMoney 19.880 CNY

Real charge for the whole run (both saved images, both sampling passes):
**0.075 CNY**.

## C. Redaction

`task.json` is committed redacted: the `netWssUrl` (which carried a signed
`Rh-Comfy-Auth=eyJ...` websocket token) and `clientId` were removed in place and
a `redacted: true` note added. Only the task id, the create response and the
note remain. No key, no token, no balance credential is in any committed file.

## D. What is committed here

Safe to publish (scripts + measurements + text report):

- `task.json` (redacted), `outputs.json`, `gate_report.json`,
  `acceptance_report.json`
- `acceptance_checks.py`, `ghost_measure.py`, `throat_diag.py`,
  `throat_zoom.py`, `make_visuals.py`, `mask_panels.py`

Deliberately **not** committed (all `*.png` are git-ignored, and the frame is a
real person): `final.png`, `clothing_pass.png`, `triptych.png`,
`neck_8x_orig_vs_final.png` and the rest of the zoom renders. They stay on disk
as the working evidence the collar-fix round reads; the report and the mask
metrics they produced are what travel with the code.

## E. Deliverables produced for this round's acceptance

From these two renders (see `p29-collar-fix-report.md`):

1. last round's original output -- `evidence/p29-submit-run/final.png`,
   `clothing_pass.png` (on disk);
2. neck 8x zoom -- `evidence/p29-collar-fix/collar_band_orig_final_cloth_8x.png`;
3. collar-edit mask before/after -- `evidence/p29-collar-fix/collar_edit_mask_before_4x.png`,
   `collar_edit_mask_after_4x.png`, `collar_edit_mask_delta_4x.png`;
4. neck-skin protection proof -- `evidence/p29-collar-fix/collar_fix_verify.json`;
5. clothing-pass vs final pale-source comparison --
   `evidence/p29-collar-fix/ghost_source_clothing_vs_final_3x.png`
   (`ghost_measure.json`);
6. corrected workflow + prompts -- `artifacts/p29-collar-fix-run/gate_report.json`.

Offline only: no provider was called, no key was read, no task was submitted.
Paid tasks this round: 0.
