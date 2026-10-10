import numpy as np, cv2
from PIL import Image, ImageDraw
REPO="/Users/liuxiuyu/AI-beauty-style-lab/repos/photo-retouch-agent"
orig=np.asarray(Image.open(f"{REPO}/data/inputs/IMG_3651.jpg").convert("RGB"))
fin =np.asarray(Image.open(f"{REPO}/evidence/p29-submit-run/final.png").convert("RGB"))
blob=np.load(f"{REPO}/evidence/p29-collar-residual-mask/_blob.npy")
def edges(img):
    g=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY).astype(np.float32)
    g=cv2.GaussianBlur(g,(0,0),0.8)
    gx=cv2.Sobel(g,cv2.CV_32F,1,0,ksize=3); gy=cv2.Sobel(g,cv2.CV_32F,0,1,ksize=3)
    return np.hypot(gx,gy)
x0,y0,x1,y1,k=490,558,600,712,9
def panel(img,tag):
    e=edges(img)[y0:y1,x0:x1]
    e=np.clip(e/ np.percentile(e,99.0) ,0,1)
    rgb=img[y0:y1,x0:x1].astype(np.float32).copy()
    m=e>0.45
    rgb[m]=rgb[m]*0.25+np.array([255,30,30])*0.75
    im=Image.fromarray(rgb.astype(np.uint8)).resize(((x1-x0)*k,(y1-y0)*k),Image.NEAREST)
    mm=(blob[y0:y1,x0:x1]*255).astype(np.uint8)
    c,_=cv2.findContours(mm,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    pts=(max(c,key=cv2.contourArea).reshape(-1,2)*k+k/2).astype(np.int32)
    a=np.array(im); cv2.polylines(a,[pts],True,(0,255,255),2); im=Image.fromarray(a)
    d=ImageDraw.Draw(im); d.rectangle([0,0,im.size[0],28],fill=(0,0,0)); d.text((6,8),tag,fill=(255,240,120))
    return im
o=panel(orig,f"ORIGINAL {k}x edges(red) + blob(cyan)")
f=panel(fin ,f"FINAL {k}x edges(red) + blob(cyan)")
out=Image.new("RGB",(o.size[0]*2+16,o.size[1]),(20,20,20)); out.paste(o,(0,0)); out.paste(f,(o.size[0]+16,0))
out.save(f"{REPO}/evidence/p29-collar-residual-mask/edge_overlay_9x.png"); print(out.size)
