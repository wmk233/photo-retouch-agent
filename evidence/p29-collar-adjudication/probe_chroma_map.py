from __future__ import annotations
import numpy as np, cv2
from PIL import Image, ImageDraw
REPO="/Users/liuxiuyu/AI-beauty-style-lab/repos/photo-retouch-agent"
final=np.asarray(Image.open(f"{REPO}/evidence/p29-submit-run/final.png").convert("RGB")).astype(np.float32)
orig=np.asarray(Image.open(f"{REPO}/data/inputs/IMG_3651.jpg").convert("RGB")).astype(np.float32)
gp=np.asarray(Image.open(f"{REPO}/evidence/p29-collar-repair/collar_repair_guarded_pale.png"))>127
rel=np.asarray(Image.open(f"{REPO}/evidence/p29-collar-repair/collar_repair_mask.png"))>127
def chroma(a): return a.max(2)-a.min(2)
cf=chroma(final); co=chroma(orig)
v=np.clip((cf/90.0*255),0,255).astype(np.uint8)
heat=cv2.applyColorMap(v, cv2.COLORMAP_JET); heat=cv2.cvtColor(heat,cv2.COLOR_BGR2RGB)
base=Image.fromarray(final.astype(np.uint8)).convert("RGB")
ov=base.copy(); ov.paste(Image.fromarray(heat), (0,0), Image.fromarray((gp*255).astype(np.uint8)))
# histogram of final chroma within guarded band
h,_=np.histogram(cf[gp],bins=np.arange(0,121,10))
print("final chroma histogram in guarded band (0-120 step10):")
for i,c in enumerate(h): print(f"  {i*10:3d}-{i*10+10:3d}: {c}")
print("orig chroma mean guarded", co[gp].mean())
# also hist of orig chroma
h2,_=np.histogram(co[gp],bins=np.arange(0,121,10))
print("orig chroma histogram in guarded band:")
for i,c in enumerate(h2): print(f"  {i*10:3d}-{i*10+10:3d}: {c}")

X0,Y0,X1,Y1=420,575,600,760; Z=8
def cu(a):
    c=np.asarray(a)[Y0:Y1,X0:X1]; im=Image.fromarray(c.astype(np.uint8))
    return im.resize((im.width*Z,im.height*Z),Image.NEAREST)
p1=cu(base); p2=cu(ov)
W=p1.width;H=p1.height
canvas=Image.new("RGB",(W*2+10,H+22),(20,20,20));d=ImageDraw.Draw(canvas)
d.text((4,4),"FINAL + guarded (chroma heat on guarded)",fill=(255,255,255))
canvas.paste(p1,(0,22)); canvas.paste(p2,(W+10,22))
canvas.save(f"{REPO}/evidence/p29-collar-adjudication/chroma_heat_guarded_8x.png")
print("saved",canvas.size)
