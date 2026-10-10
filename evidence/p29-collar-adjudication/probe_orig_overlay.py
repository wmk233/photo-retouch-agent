from __future__ import annotations
import numpy as np, cv2
from PIL import Image, ImageDraw

REPO = "/Users/liuxiuyu/AI-beauty-style-lab/repos/photo-retouch-agent"
orig = np.asarray(Image.open(f"{REPO}/data/inputs/IMG_3651.jpg").convert("RGB"))
rel = np.asarray(Image.open(f"{REPO}/evidence/p29-collar-repair/collar_repair_mask.png")) > 127
gp  = np.asarray(Image.open(f"{REPO}/evidence/p29-collar-repair/collar_repair_guarded_pale.png")) > 127

# local-contrast (unsharp) version of original to reveal knit fibre vs smooth skin
blur = cv2.GaussianBlur(orig, (0,0), 3.0)
sharp = cv2.addWeighted(orig, 1.7, blur, -0.7, 0)

ov = orig.copy()
ov[rel] = (255,0,0); ov[gp] = (255,215,0)
ov2 = sharp.copy()
ov2[rel] = (255,0,0); ov2[gp] = (255,215,0)

X0,Y0,X1,Y1 = 400, 545, 615, 775
Z = 6
def cu(a):
    c = a[Y0:Y1, X0:X1]; im = Image.fromarray(c.astype(np.uint8))
    return im.resize((im.width*Z, im.height*Z), Image.NEAREST)
panels = [("ORIG", cu(orig)), ("ORIG+mask", cu(ov)), ("ORIG sharp+mask", cu(ov2))]
W = panels[0][1].width; H = panels[0][1].height
canvas = Image.new("RGB", (W*len(panels)+8*(len(panels)-1), H+20), (20,20,20))
d = ImageDraw.Draw(canvas)
for i,(t,im) in enumerate(panels):
    x = i*(W+8); canvas.paste(im,(x,20)); d.text((x+4,4), t, fill=(255,255,255))
canvas.save(f"{REPO}/evidence/p29-collar-adjudication/orig_overlay_sharp_6x.png")
print("saved", canvas.size)
