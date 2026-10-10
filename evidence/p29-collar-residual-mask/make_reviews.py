"""The five visual-review artifacts the brief asks for (sample-only round)."""
import numpy as np, cv2
from PIL import Image, ImageDraw
REPO="/Users/liuxiuyu/AI-beauty-style-lab/repos/photo-retouch-agent"
WORK=REPO+"/evidence/p29-collar-residual-mask"
orig=np.asarray(Image.open(f"{REPO}/data/inputs/IMG_3651.jpg").convert("RGB"))
fin =np.asarray(Image.open(f"{REPO}/evidence/p29-submit-run/final.png").convert("RGB"))
d=np.load(f"{WORK}/_masks.npy",allow_pickle=True).item()
blob=np.load(f"{WORK}/_blob.npy")
sample=np.load(f"{WORK}/_sample.npy"); unconf=np.load(f"{WORK}/_unconfirmed.npy")
comm=np.load(f"{WORK}/_committed.npy")
x0,y0,x1,y1,k=478,552,606,722,8
crop=lambda a: a[y0:y1,x0:x1].astype(np.float32)
def up(arr): 
    im=Image.fromarray(arr.astype(np.uint8)); return im.resize((im.size[0]*k,im.size[1]*k),Image.NEAREST)
def paint(base, mask, color, alpha):
    m=crop(mask)[:,:,None].astype(np.float32); out=base.copy()
    for ch in range(3): out[:,:,ch]=np.where(m[:,:,0]>0, base[:,:,ch]*(1-alpha)+color[ch]*alpha, out[:,:,ch])
    return out
def title(im,t):
    dd=ImageDraw.Draw(im); dd.rectangle([0,0,im.size[0],28],fill=(0,0,0)); dd.text((6,8),t,fill=(255,240,120)); return im
def contour(im,mask,color,thick=2):
    mm=(crop(mask)*255).astype(np.uint8)
    c,_=cv2.findContours(mm,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    a=np.array(im); cv2.drawContours(a,[(p.reshape(-1,2)*k+k//2).astype(np.int32) for p in c],-1,color,thick)
    return Image.fromarray(a)
def legend(im,items):
    dd=ImageDraw.Draw(im); y=34
    for txt,col in items:
        dd.rectangle([6,y,20,y+14],fill=col); dd.text((26,y+1),txt,fill=(255,255,255)); y+=20
    return im
fo, ff = crop(orig), crop(fin)
# 1
i1=contour(up(fo),blob,(0,255,255),2); title(i1,f"1. ORIGINAL neck {k}x   cyan = the 1067px block, bbox[514,578,51,111]"); i1.save(f"{WORK}/review_1_original_neck_8x.png")
# 2
i2=contour(up(ff),blob,(0,255,255),2); title(i2,f"2. BEST FINAL neck {k}x   cyan = same block, left untouched by the render"); i2.save(f"{WORK}/review_2_final_neck_8x.png")
# 3
a3=paint(ff,comm,(0,140,255),0.45); a3=paint(a3,sample&~comm,(0,255,0),0.55)
i3=contour(up(a3),blob,(255,0,255),1)
i3=legend(title(i3,"3. CANDIDATE old-cloth mask on the FINAL"),[("blue = already-released residual (committed)",(0,140,255)),("green = added by this sample-only mask",(0,255,0)),("magenta = the 1067px block outline",(255,0,255))])
i3.save(f"{WORK}/review_3_candidate_mask_overlay.png")
# 4
prot=(~sample)&(d['window']|blob)
a4=paint(ff,sample,(0,220,0),0.40); a4=paint(a4,prot,(220,0,0),0.40)
i4=legend(title(up(a4),"4. EDIT-ALLOWED (green) vs PROTECTED (red) in the collar band"),[("may be repainted by the local inpaint",(0,220,0)),("must keep the final pixel",(220,0,0))])
i4.save(f"{WORK}/review_4_edit_vs_protect.png")
# 5
i5=legend(title(up(paint(ff,unconf,(255,0,255),0.75)),"5. UNCONFIRMED residual pixels (magenta)"),[("visibly pale in the final, not provably old knit",(255,0,255))])
dd=ImageDraw.Draw(i5)
for b in [{"px":668,"bbox":[526,617,39,67]},{"px":251,"bbox":[472,704,17,25]},{"px":170,"bbox":[526,726,15,16]}]:
    bx,by,bw,bh=b["bbox"]
    dd.rectangle([(bx-x0)*k,(by-y0)*k,(bx+bw-x0)*k,(by+bh-y0)*k],outline=(255,255,0),width=2)
    dd.text(((bx-x0)*k+3,(by-y0)*k+3),f'{b["px"]}px',fill=(255,255,0))
i5.save(f"{WORK}/review_5_unconfirmed.png")
print("wrote 5 review images", i1.size)
