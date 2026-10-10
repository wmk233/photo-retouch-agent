"""Sample-only collar repair mask for the P2.9 best final (task 2108330225116082177).

Why a separate script (brief: do NOT touch the general tools):
  * ``edit_mask.py`` is the product's general mask builder.
  * ``tools/build_collar_repair_mask.py`` is the committed round tool; its numbers
    are the frozen evidence of the previous adjudication.
This script exists for THIS sample only. It adds NO new colour threshold: every
tone test below is copied verbatim from the committed round tool. The only thing
it adds is structural connectivity to the already-adjudicated defect -- which is
exactly what the brief asks for (real collar outline / continuous folds /
adjacent-cloth structure), instead of a new skin-vs-cloth brightness cut.

white (255) = a residual old-collar pixel a single local inpaint may repaint
black (0)   = keep the best final's own pixel
"""
import json, sys
from pathlib import Path
import numpy as np, cv2
from PIL import Image, ImageDraw

REPO = Path("/Users/liuxiuyu/AI-beauty-style-lab/repos/photo-retouch-agent")
WORK = Path(__file__).resolve().parent

# --- thresholds copied verbatim from tools/build_collar_repair_mask.py (no new tuning) ---
SOURCE_KNIT_LUMA_MIN, SOURCE_KNIT_CHROMA_MAX = 140.0, 50.0   # "the original pixel was pale knit"
SURVIVOR_UNCHANGED_PX = 10                                    # "the render left this pixel alone"
OLIVE_HUE_MIN, OLIVE_HUE_MAX, OLIVE_SAT_MIN = 45.0, 115.0, 0.12  # the correct new blouse
VISIBLE_PALE_LUMA_MIN, VISIBLE_PALE_CHROMA_MAX = 140.0, 55.0  # the r9 "visible residual" probe
MIN_BLOB_PX = 100

d = np.load(WORK / "_masks.npy", allow_pickle=True).item()
orig = np.asarray(Image.open(REPO / "data/inputs/IMG_3651.jpg").convert("RGB")).astype(np.int16)
fin = np.asarray(Image.open(REPO / "evidence/p29-submit-run/final.png").convert("RGB")).astype(np.int16)
H, W = orig.shape[:2]
skin, hair, background = d["skin"], d["hair"], d["background"]
oval, protected, olive = d["oval"], d["protected"], d["olive"]
ol, oc, fl, fc, change = d["ol"], d["oc"], d["fl"], d["fc"], d["change"]
window = d["window"]

committed = np.asarray(
    Image.open(REPO / "evidence/p29-collar-repair/collar_repair_mask.png").convert("L")) > 127

# 1. old pale knit the render did not repaint   (committed tone rule + committed unchanged test)
pale_knit_unchanged = (ol > SOURCE_KNIT_LUMA_MIN) & (oc < SOURCE_KNIT_CHROMA_MAX) & (change < SURVIVOR_UNCHANGED_PX)
# 2. never release these, whatever the tone test says
forbidden = oval | hair | background | olive
# 3. candidate growth pixels: old knit, unchanged, not forbidden
cand = pale_knit_unchanged & ~forbidden
# 4. STRUCTURE: keep only the 8-connected components of `cand` that touch the committed release
n, lab, stats, _ = cv2.connectedComponentsWithStats((cand * 255).astype(np.uint8), 8)
grow = np.zeros((H, W), bool)
grown_blobs = []
for i in range(1, n):
    comp = lab == i
    if (comp & committed).any():
        grow |= comp
        grown_blobs.append({"px": int(stats[i, cv2.CC_STAT_AREA]),
                            "bbox": [int(stats[i, 0]), int(stats[i, 1]), int(stats[i, 2]), int(stats[i, 3])]})
    elif comp.sum() >= MIN_BLOB_PX:
        grown_blobs.append({"px": int(comp.sum()), "bbox": [int(stats[i, 0]), int(stats[i, 1]),
                            int(stats[i, 2]), int(stats[i, 3])], "not_connected_to_defect": True})

sample = committed | grow
sample = cv2.dilate(sample.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0   # 1px rim, closes AA holes
sample = sample & ~forbidden
Image.fromarray((sample * 255).astype(np.uint8)).save(WORK / "sample_collar_repair_mask.png")

# --- what the mask covers, and what stays protected ---
visible_pale = (fl > VISIBLE_PALE_LUMA_MIN) & (fc < VISIBLE_PALE_CHROMA_MAX) & ~olive & window
vp_outside_committed = visible_pale & ~committed
unconfirmed = visible_pale & ~sample
def blobs(m, floor=1):
    nn, _, st, _ = cv2.connectedComponentsWithStats((m * 255).astype(np.uint8), 8)
    return sorted([{"px": int(st[i, cv2.CC_STAT_AREA]), "bbox": [int(st[i, 0]), int(st[i, 1]),
                    int(st[i, 2]), int(st[i, 3])]} for i in range(1, nn) if st[i, cv2.CC_STAT_AREA] >= floor],
                  key=lambda b: -b["px"])

blob1067 = np.load(WORK / "_blob.npy")
stats_out = {
    "tool": "evidence/p29-collar-residual-mask/build_sample_mask.py (SAMPLE ONLY; general tools untouched)",
    "baseline_final": "evidence/p29-submit-run/final.png",
    "original": "data/inputs/IMG_3651.jpg",
    "rules_copied_verbatim_from_round_tool": {
        "source_knit": "original luma > 140 and chroma < 50",
        "render_left_it_alone": "max channel change < 10",
        "never_release": "face oval / hair / background / new olive blouse",
        "no_new_threshold": True,
    },
    "rule_added_this_sample_only": (
        "8-connectivity: a candidate pixel is released only if it lies in the same 4/8-connected "
        "blob of 'old pale knit the render did not repaint' as the already-adjudicated release"
    ),
    "committed_release_px": int(committed.sum()),
    "sample_release_px": int(sample.sum()),
    "growth_px": int(grow.sum()),
    "growth_blobs": sorted(grown_blobs, key=lambda b: -b["px"])[:10],
    "sample_release_bbox": [int(np.where(sample.any(0))[0].min()), int(np.where(sample.any(1))[0].min()),
                            int(np.where(sample.any(0))[0].max()), int(np.where(sample.any(1))[0].max())],
    "release_over_forbidden_px": int((sample & forbidden).sum()),
    "release_over_skin_model_px": int((sample & skin).sum()),
    "release_over_face_oval_px": int((sample & oval).sum()),
    "release_over_protected_px": int((sample & protected).sum()),
    "visible_residual_px": int(visible_pale.sum()),
    "visible_residual_outside_committed_px": int(vp_outside_committed.sum()),
    "visible_residual_now_covered_px": int((visible_pale & sample).sum()),
    "unconfirmed_px": int(unconfirmed.sum()),
    "unconfirmed_blobs": blobs(unconfirmed, MIN_BLOB_PX)[:12],
    "block_1067_px": int(blob1067.sum()),
    "block_1067_released_px": int((blob1067 & sample).sum()),
    "block_1067_released_pct": round(100 * float((blob1067 & sample).sum()) / max(int(blob1067.sum()), 1), 1),
    "block_1067_unconfirmed_px": int((blob1067 & unconfirmed).sum()),
}
np.save(WORK / "_sample.npy", sample)
np.save(WORK / "_unconfirmed.npy", unconfirmed)
np.save(WORK / "_committed.npy", committed)
(WORK / "sample_mask_stats.json").write_text(json.dumps(stats_out, indent=1), encoding="utf-8")
print(json.dumps(stats_out, indent=1))
