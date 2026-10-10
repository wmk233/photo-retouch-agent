import numpy as np, cv2
from PIL import Image, ImageDraw
REPO="/Users/liuxiuyu/AI-beauty-style-lab/repos/photo-retouch-agent"
orig=np.asarray(Image.open(f"{REPO}/data/inputs/IMG_3651.jpg").convert("RGB"))
blob=np.load(f"{REPO}/evidence/p29-collar-residual-mask/_blob.npy")
x0,y0,x1,y1,k=498,570,596,700,11
sub=orig[y0:y1,x0:x1]
# high-pass (local contrast) on luma, amplified
g=cv2.cvtColor(sub,cv2.COLOR_RGB2GRAY).astype(np.float32)
lo=cv2.GaussianBlur(g,(0,0),3.0)
hp=g-lo
vis=np.clip(128+hp*4.0,0,255).astype(np.uint8)
vis=cv2.cvtColor(vis,cv2.COLOR_GRAY2RGB)
def lab(img,t):
    d=ImageDraw.Draw(img); d.rectangle([0,0,img.size[0],28],fill=(0,0,0)); d.text((6,8),t,fill=(255,240,120)); return img
def up(a): 
    im=Image.fromarray(a); return im.resize((im.size[0]*k,im.size[1]*k),Image.NEAREST)
a=lab(up(sub),f"ORIGINAL {k}x raw")
b=lab(up(vis),f"ORIGINAL {k}x high-pass x4 (texture only)")
out=Image.new("RGB",(a.size[0]*2+16,a.size[1]),(20,20,20)); out.paste(a,(0,0)); out.paste(b,(a.size[0]+16,0))
out.save(f"{REPO}/evidence/p29-collar-residual-mask/texture_zoom_{k}x.png"); print(out.size)
# texture stats
def stats(m, name):
    mm=m[y0:y1,x0:x1]
    if mm.sum()==0: print(name,"empty"); return
    l=hp[mm]
    print(f"{name:24s} n={int(mm.sum()):5d} hp_std={l.std():6.2f} hp_absmean={np.abs(l).mean():6.2f}")
d=np.load(f"{REPO}/evidence/p29-collar-residual-mask/_masks.npy",allow_pickle=True).item()
skin=d['skin']; olive=d['olive']; ol=d['ol']; oc=d['oc']
inner=cv2.erode(blob.astype(np.uint8),np.ones((3,3),np.uint8))>0
stats(inner,"blob interior")
neck=skin&~blob&(ol<170)
stats(neck,"orig skin-model (non-blob)")
knit=(ol>150)&(oc<45)&~blob
stats(knit,"orig bright knit (non-blob)")
olv=(ol<130)&ocv if False else None
olive_n=(oc>25)
stats(olive&(np.arange(orig.shape[0])[:,None]>560), "final olive? (orig px)")
