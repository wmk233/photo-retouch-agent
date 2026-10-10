import numpy as np
from PIL import Image, ImageDraw
REPO = "/Users/liuxiuyu/AI-beauty-style-lab/repos/photo-retouch-agent"
orig = np.asarray(Image.open(f"{REPO}/data/inputs/IMG_3651.jpg").convert("RGB"))
fin  = np.asarray(Image.open(f"{REPO}/evidence/p29-submit-run/final.png").convert("RGB"))
def up(a,k): 
    im=Image.fromarray(a); return im.resize((im.size[0]*k, im.size[1]*k), Image.NEAREST)
def lab(img, text):
    d=ImageDraw.Draw(img); d.rectangle([0,0,img.size[0],26],fill=(0,0,0)); d.text((6,7),text,fill=(255,240,120)); return img
x0,y0,x1,y1,k = 495,565,590,705,10
o=lab(up(orig[y0:y1,x0:x1],k), f"ORIGINAL {k}x x{x0}-{x1} y{y0}-{y1}")
f=lab(up(fin[y0:y1,x0:x1],k),  f"FINAL {k}x x{x0}-{x1} y{y0}-{y1}")
out=Image.new("RGB",(o.size[0]+f.size[0]+16, max(o.size[1],f.size[1])),(20,20,20))
out.paste(o,(0,0)); out.paste(f,(o.size[0]+16,0))
out.save(f"{REPO}/evidence/p29-collar-residual-mask/block_zoom_{k}x.png")
print("wrote", out.size)
