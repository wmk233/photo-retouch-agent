import numpy as np, cv2
from PIL import Image, ImageDraw
REPO="/Users/liuxiuyu/AI-beauty-style-lab/repos/photo-retouch-agent"
blob=np.load(f"{REPO}/evidence/p29-collar-residual-mask/_blob.npy")
orig=np.asarray(Image.open(f"{REPO}/data/inputs/IMG_3651.jpg").convert("RGB"))
fin=np.asarray(Image.open(f"{REPO}/evidence/p29-submit-run/final.png").convert("RGB"))
x0,y0,x1,y1,k=490,560,600,710,9
def up(a): 
    im=Image.fromarray(a); return im.resize((im.size[0]*k,im.size[1]*k),Image.NEAREST)
def outline(img, m):
    mm=(m[y0:y1,x0:x1]*255).astype(np.uint8)
    cnts,_=cv2.findContours(mm,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    big=max(cnts,key=cv2.contourArea)
    pts=(big.reshape(-1,2)*k + k/2).astype(np.int32)
    im=np.array(img); cv2.polylines(im,[pts],True,(255,0,0),3); return Image.fromarray(im)
def lab(img,t):
    d=ImageDraw.Draw(img); d.rectangle([0,0,img.size[0],28],fill=(0,0,0)); d.text((6,8),t,fill=(255,240,120)); return img
o=lab(outline(up(orig[y0:y1,x0:x1]),blob), f"ORIGINAL {k}x  RED=1067px residual blob")
f=lab(outline(up(fin[y0:y1,x0:x1]),blob),  f"FINAL {k}x  RED=1067px residual blob")
out=Image.new("RGB",(o.size[0]*2+16,max(o.size[1],f.size[1])),(20,20,20))
out.paste(o,(0,0)); out.paste(f,(o.size[0]+16,0))
out.save(f"{REPO}/evidence/p29-collar-residual-mask/blob_outline_9x.png"); print(out.size)
