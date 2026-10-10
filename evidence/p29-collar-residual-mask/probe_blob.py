import numpy as np, cv2
from PIL import Image
REPO="/Users/liuxiuyu/AI-beauty-style-lab/repos/photo-retouch-agent"
d=np.load(f"{REPO}/evidence/p29-collar-residual-mask/_masks.npy",allow_pickle=True).item()
orig=np.asarray(Image.open(f"{REPO}/data/inputs/IMG_3651.jpg").convert("RGB"))
fin=np.asarray(Image.open(f"{REPO}/evidence/p29-submit-run/final.png").convert("RGB"))
H,W=orig.shape[:2]
fl,fc,ol,oc,olive=d['fl'],d['fc'],d['ol'],d['oc'],d['olive']
band=np.zeros((H,W),bool); band[560:760,400:640]=True
vis=(fl>140)&(fc<55)&~olive&band
# outside edit region: what was released+carved
rel=np.asarray(Image.open(f"{REPO}/evidence/p29-collar-repair/collar_repair_mask.png").convert("L"))>127
vis_out=vis&~rel
n,lbl,stats,_=cv2.connectedComponentsWithStats((vis_out*255).astype(np.uint8),8)
print("blobs (px,bbox) top6:",[(int(stats[i,4]),[int(stats[i,0]),int(stats[i,1]),int(stats[i,2]),int(stats[i,3])]) for i in np.argsort(-stats[1:,4])[:6]+1])
big=1+int(np.argmax(stats[1:,4]))
blob=lbl==big
print("largest blob px",int(blob.sum()),"bbox",stats[big,:4].tolist())
ys,xs=np.where(blob); print("y range",ys.min(),ys.max(),"x range",xs.min(),xs.max())
# also compute residual-vs-original knit stats restricted to the blob
print("blob: orig skin-model",int((blob&d['skin']).sum()),"orig knit(l140c50)",int((blob&(ol>140)&(oc<50)).sum()),
      "orig bright-neutral",int((blob&(ol>150)&(oc<45)).sum()),"unchanged<10",int((blob&(d['change']<10)).sum()),"/",int(blob.sum()))
print("blob median orig RGB",np.median(orig[blob],0).tolist(),"final RGB",np.median(fin[blob],0).tolist())
print("blob mean orig luma",float(ol[blob].mean()),"chroma",float(oc[blob].mean()))
np.save(f"{REPO}/evidence/p29-collar-residual-mask/_blob.npy", blob)
