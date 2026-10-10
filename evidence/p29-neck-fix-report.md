# P2.9 neck-ghost / hairline / collar revision (offline)

Round scope: fix the two real RunningHub renders offline -- the pale ghost on
the chin and neck, the hard double hairline on the forehead and the deep V
neckline -- without a second paid task. **Paid tasks this round: 0.**

Base revision: `443f1e2`. Photo: `data/inputs/IMG_3651.jpg` (904x1190).
Plan: `ai_elegant_refined` (`job_45f7920ad505.json`), verbatim, not an axis.

Everything below is derived from the shipped package
`artifacts/local-edit-package/20261008T153609Z`, the cached SCHP label map
`evidence/p29-neck-fix/p29_labels.png` and the two real renders
`artifacts/p29-run/result_0.png` (task 2108222523123662850, second face) and
`artifacts/p29-fix-run/result_0.png` (task 2108231348232871937, pale ghost).

## A. Where the pale ghost and the second face come from

Rule used throughout: a ghost pixel is `max|channel delta| > 25` **and** the
render brighter than the original by more than 15. In the neck band
`y 560-1000, x 240-820` that is 37,922 px.

| attribution | share of ghost px |
|---|---|
| inside the shipped **hair** edit mask | **74.3 %** |
| inside the shipped clothing edit mask | 53.9 % |
| inside the protected identity mask | **0.33 %** |
| SCHP class hair | 58.8 % |
| SCHP class clothing | 32.0 % |
| SCHP class face | 4.6 % |

The guard is not the cause: it covers three pixels in a thousand of the ghost.
The ghost lives inside the region the run was *allowed* to repaint.

Three real defects put it there:

1. **The hair mask reached far over the garment.** On the shipped mask the hair
   region overlapped 48,654 px of SCHP-parsed clothing and 45,043 px of parsed
   background; the spill ran a median 14.3 px and up to 32.7 px beyond the
   parsed hair, because `hair_grow_ratio = 0.12` is a 30.8 px rim at this face
   width (256.7 px). A rim that large is right on the background side of the
   silhouette and wrong on the collar side: it handed the hair pass a band of
   collar to repaint.
2. **Both passes repainted the same seam.** The clothing mask sat on 23,225 px
   of SCHP-parsed hair, so the collar and the hair over it were painted once by
   the clothing pass and again by the hair pass, which runs second and
   therefore has the last word (`mask_scope.engine_order`, unchanged).
3. **The dark throat was carved out and given to the garment.** The collar
   split removed 13,470 px below the face oval; all of it lies below the oval
   (0 px over it), 85.7 % of it is brighter than 0.65x the subject's own skin
   luminance and only 10.6 % of it is within 60 RGB of skin, so it is the white
   knit collar, not the neck -- correct on the whole, but the shadow just under
   the jaw falls in the same chroma-rejected bucket and went with it. 237 px of
   genuinely dark throat were being handed to the clothing pass.

The same mechanism explains the first render: with the whole neck band editable
by both passes at `denoise` 0.75/0.70, the hair pass had a collar worth of
cloth and bare neck to repaint and drew a second head, a second pair of glasses
and a second mouth into it.

What the existing artefacts **cannot** separate is the per-pass contribution:
both renders are single stitched images, so the ghost can be attributed to
*regions* but not split between the clothing and hair pass. The new
`SaveImage` on node `20` records the clothing-pass stitch in the next run
precisely so the split becomes observable (see E).

## B. What changed in the masks

Four knobs on `SemanticMaskConfig` (`backend/app/domains/portrait/edit_mask.py`),
all defaulted to the fixed behaviour:

- `hair_rim_over_clothing_ratio = 0.02` -- the hair rim is split into an
  asymmetric pair: wide (0.12) over background, 0.02 over parsed clothing.
- `clothing_hair_clearance_ratio = 0.0` -- the garment stops exactly on the
  parse's own hair boundary, so cloth and hair tile the seam instead of both
  painting it. The hair pass owns the strands and their overhang.
- `hair_guard_release_erode_ratio = 0.0` -- the parsed fringe is eroded (at
  zero, not at all) and subtracted from the face guard, so the subject's own
  fringe is repainted with the new hairstyle.
- `collar_skin_luma_ratio = 0.35` -- below-chin non-skin that is also darker
  than 0.35x the subject's skin luminance is a shadowed throat and stays
  protected.

Plus a `_luma` helper using the same Rec.601 coefficients as
`face_lock._ycc`, so the shadow test agrees with the skin test that rejected the
pixel. Release only removes parsed-**hair** pixels: `protected` is
`maximum(maximum(features, glasses), protected_face)`, and no class map puts one
pixel in both `Hair` and `Face`, so no skin can be freed.

Measured on the real photo and the cached SCHP map (`mask_revision.json`):

| | shipped | fixed |
|---|---|---|
| hair px | 224,961 | **187,091** |
| clothing px | 325,145 | **302,487** |
| protected px | 81,741 | **75,762** |
| editable px | 470,242 | 476,221 |
| hair n clothing | **79,864** | **13,357** |
| hair over parsed clothing | 48,654 | **8,307** |
| clothing over parsed hair | **23,225** | **804** |
| hair over parsed face | 4,379 | **640** |
| protected n parsed hair | 7,507 | **1,289** |
| background inside edit | 13.06 % | 12.90 % |
| hair n protected | 0 | **0** |
| clothing n protected | 0 | **0** |
| editable n face core | 0 | **0** |
| neck skin protected | 4,792 | **5,029** |

The 13,357 px of remaining hair n clothing overlap is the strand-on-cloth band
the user asked to keep: it is not deleted, it is *owned* -- the hair pass runs
second and keeps final edit rights, so the strands press the cloth.

## C. Hairline

Mechanism (`mechanism.json`, `hairline_stats.json`): the protected island is the
MediaPipe face oval union the features union a glasses band, dilated. The
subject's own dark fringe lies inside that oval, so unioning the oval in without
carving the hair back out **froze 7,507 px of her own fringe as original
pixels** (74.5 % of them unchanged to within 2 levels, mean luminance 32 -
dark hair, not skin). The new brown waves were generated outside the island, so
the forehead carried two edges: the frozen old hairline and the new one.

Releasing the fringe drops that to 1,289 px (the panel's direct count labels
1,291; the 2 px gap is antialiasing at the boundary). The frozen band was never
the cause of the *hard* edge seen on the render -- of the 24,965 px that changed
in the hairline band, 24,737 were already inside the hair mask and only 228
outside, so blend expansion is numerically excluded. The fix removes the second
edge, not the blend.

Hairline band `y 60-420, x 200-720`, 3x: `hairline_mask_zoom3x.png`
(5 panels: original, shipped hair/protect, shipped frozen fringe, fixed
hair/protect, fixed frozen fringe).

## D. Collar prompt

Source: `ai_elegant_refined` verbatim from `job_45f7920ad505.json`. Clothing
pass text is unchanged except for one appended clause and negative terms
(`backend/tools/runninghub_crop_stitch_run.py`):

- positive, appended: `a higher natural collar closed near the collarbone with
  fine pleats and soft vintage folds`
- negative, appended: `deep v neck, plunging neckline, exposed chest,
  bare cleavage, sheer mesh shoulder, statement necklace`

`AIStylingPlanner` is untouched. The gate hard-fails if either the clause or any
negative term is missing.

## E. The new workflow

`artifacts/p29-neck-fix-run/workflow.json`, built from the rebuilt package
`artifacts/local-edit-package/20261009T-neckfix` (hair 187,091 / clothing
302,487 / protected 75,762 / hair n clothing 13,357 / background 12.90 %).
Gate: **`errors: []`**, 20 nodes against the live 19,188-node table.

Unchanged and re-verified: `LoadImage` -> `InpaintCropImproved` ->
`InpaintModelConditioning` -> `KSampler` -> `VAEDecode` -> `InpaintStitchImproved`
for each pass, clothing first, hair second, pass 2 fed by node `10`'s stitch.
No `EmptyLatentImage`, no `VAEEncodeForInpaint`; `denoise` 0.75 clothing /
0.70 hair; crop `mask_expand_pixels=0`, `mask_blend_pixels=2`,
`context_from_mask_extend_factor=1.2`, target 1024x1024.

Note the honest description of the conditioning: the checkpoint
(`juggernautXL_juggXIByRundiffusion.safetensors`) has no inpainting UNet input,
so `InpaintModelConditioning`'s concat channel is empty. What actually carries
the original pixels is the encoded crop plus `noise_mask`, at `denoise < 1.0`.
The empty concat is not a dedicated inpaint condition. The gate reports this as
a note, not as a pass.

New for this round: **node `20` `SaveImage`** records the clothing-pass stitch
(node `10`); node `19` remains the final image. Two stage artefacts, no extra
sampling, no extra task. The gate now fails if any `SaveImage` other than those
two exists.

Mask geometry in the new workflow: hair painted 189,900 / ring 9,767 /
over protected 475 / over facial core 0; clothing painted 305,270 / ring 8,078 /
over protected 541 / over facial core 0. The 475/541 px are the 2 px feather
ring clipping the protected boundary -- unchanged from the previously accepted
run (494/549) and not facial core.

## F. Strict per-pixel identity of the core

The earlier reports quoted "identical within le2 / le11", which is a tolerance
band, not preservation. Byte-exact (`max|delta| == 0`) against the original, on
the last render (`pixel_identity_strict.json`):

| region | px | byte identical | le2 | >25 | max delta |
|---|---|---|---|---|---|
| protected identity (fixed) | 75,762 | 2.0 % | 93.1 % | 538 | 146 |
| facial core (fixed) | 73,331 | 1.7 % | 93.7 % | 427 | 105 |
| protected minus facial core | 2,431 | 12.0 % | 75.9 % | 111 | 146 |
| outside **both** stitch crops | 93,090 | 11.5 % | **99.8 %** | **0** | **3** |

Conclusion, stated the way the evidence supports it on the last render:
outside the two crop boxes nothing is touched beyond 3 levels and nothing is
above 25 -- the pipeline is not damaging the untouched frame. Inside the crop
boxes every pixel is resampled once (the crop is scaled 1162 -> 1024 and back),
which is why the protected island is only 93.1 % within 2 levels rather than
byte-identical. **The protected identity pixels are therefore inside tolerance,
not byte-identical**, and the earlier `max|delta|=11` claim must be read as a
tolerance, not as preservation. The first render was worse on the same
measurement (>25 for 680 px, max 165).

## G. Tests

`backend` full suite: **346 passed** (baseline this round 338; +8).

- `tests/test_semantic_edit_mask.py` (30) -- new: the clothing mask never
  reaches back over parsed hair while the hair mask still overlaps parsed
  cloth; the released fringe leaves a single edge and the frozen band strictly
  shrinks; the synthetic glasses band is a band, not the whole oval.
- `tests/test_runninghub_crop_stitch_workflow.py` (20) -- new: the collar
  clause and its negative terms are enforced; the stage records expose exactly
  one clothing-pass image and one final image; the gate still passes with no
  API key.

Offline only: no provider was called, no key was read, no task was submitted.
Paid tasks this round: 0.
