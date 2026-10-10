# P2.9 — first real RunningHub CropAndStitch run (1 paid task)

Result: **FAILED — do not use the output image.** One paid task was created and
consumed; the round stops here per the single-task rule.

## Run record

| item | value |
| --- | --- |
| task id | `2108222523123662850` |
| execution | inline `workflow` (full graph JSON) on `/task/openapi/create` |
| status | `SUCCESS` |
| graph | 19 nodes, 2 passes: clothing -> stitch -> hair -> stitch |
| output | `artifacts/p29-run/result_0.png` (downloaded from `/task/openapi/outputs`, nodeId 19) |
| output size | 904x1190 (identical to the source photo — size contract holds) |
| formal charge | `consumeMoney = "0.045"` CNY, `taskCostTime = 40` s |
| balance | `remainMoney` 20.000 -> 19.955 CNY; `remainCoins` unchanged at 100 |
| paid tasks used | 1 of 1 |

Source package: `artifacts/local-edit-package/20261008T153609Z`
(`--plan ai_elegant_refined`, real SCHP masks, real plan text).

## What the run proves

1. RunningHub accepts a **fully inline workflow** (no saved workflowId content is
   used; `workflowId` only has to be non-null — omitting it returns
   `code=301 must not be null` without creating a task).
2. `InpaintCropImproved` / `InpaintStitchImproved` execute and return an image.
3. The **crop -> inpaint -> stitch coordinate contract holds**: the returned
   image is byte-comparable to the source at 904x1190, and
   `mean|result-original| = 0.90` over every pixel outside the two edit masks.
4. 99.44 % of all changed pixels (`|diff| > 25`, 372 900 px) fall **inside**
   `hair_edit_mask` or `clothing_edit_mask`. The composite is correct.
5. The protected identity core is untouched: for protected pixels >= 8 px away
   from any edit pixel, `max|diff| = 11`, `mean|diff| = 1.34` (JPEG decode
   rounding), i.e. eyes / brows / nose / mouth / teeth / glasses pixels survive.

## Why the output is rejected

The composite is faithful, but the **generated content is a different person**.
The sampler re-imagined the whole head, and because both masks hug the face —
the hair footprint contains hair strands that overlap the jaw/cheek, and the
clothing footprint runs up to the neck — that invented person is deposited
right beside, and partly under, the surviving original face. The result reads
as two people stacked on top of each other.

Observed in `artifacts/p29-run/result_0.png`:

* a second woman's jaw, lips, teeth and a second pair of glasses;
* hair replaced by a dark mass with bald/highlight patches over the forehead;
* the garment replaced by a dark green knit with mesh shoulders plus gold
  necklaces (content the prompt never asked for);
* a soft rectangular patch in the background of the crop region.

None of that originates outside the masks — it is what the model painted *inside*
them at `denoise = 1.0` (`VAEEncodeForInpaint`, `grow_mask_by = 0`), where the
crop contains the entire head and nothing anchors the face against being
repainted.

## Acceptance state

| check | state |
| --- | --- |
| `MASK_VALID` | PASSED — masks built by SCHP, 9/9 rules |
| size contract | PASSED — 904x1190 |
| `PIXEL_IDENTITY_PASSED` | PASSED — protected core `max|diff| = 11` |
| `VISUAL_ACCEPTANCE_PENDING` | — |
| `VISUAL_ACCEPTANCE_FAILED` | **set** — face/identity visually broken |

A programmatically restored facial pixel core is **not** visual acceptance.

## Single blocker

The inpainting pass replaced the subject instead of restyling hair and clothing:
at full denoise over a head-sized crop the model re-invents the face, and the
masks (which necessarily touch the jaw and neck) composite that invention onto
the original. Fixing it needs a second paid task, so the round stops here.

## Non-billable fixes already applied this round

* validator: accept ComfyUI's single-element COMBO form and `image_upload` inputs
* upload endpoint: `/openapi/v2/upload/binary` returns `errorCode 1000` for this
  key; `/task/openapi/upload` with the `apiKey` form field works
* create payload: `workflowId` must be non-null even with an inline `workflow`
