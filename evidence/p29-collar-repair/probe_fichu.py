from pathlib import Path
import numpy as np
from PIL import Image
ROOT = Path(__file__).resolve().parents[2]
final = np.asarray(Image.open(ROOT/"evidence/p29-submit-run/final.png").convert("RGB")).astype(np.float32)
orig = np.asarray(Image.open(ROOT/"data/inputs/IMG_3651.jpg").convert("RGB")).astype(np.float32)
def s(name,y0,y1,x0,x1,img=final):
    a=img[y0:y1,x0:x1].reshape(-1,3)
    l=0.299*a[:,0]+0.587*a[:,1]+0.114*a[:,2]; c=a.max(1)-a.min(1); gb=a[:,1]-a[:,2]
    print(f"{name:34s} RGB={a.mean(0).round(0)} L={l.mean():6.1f} p10L={np.percentile(l,10):5.1f} C={c.mean():5.1f} p90C={np.percentile(c,90):5.1f} G-B={gb.mean():5.1f} p10GB={np.percentile(gb,10):5.1f}")
print("FINAL")
s("fichu cream (595-715, 478-568)", 595,715,478,568)
s("fichu upper (585-640, 490-570)", 585,640,490,570)
s("olive collar left (640-740, 415-470)", 640,740,415,470)
s("olive collar right (620-720, 570-660)", 620,720,570,660)
s("throat skin (560-595, 460-540)", 560,595,460,540)
s("hair left (640-760, 340-420)", 640,760,340,420)
print("ORIGINAL")
s("orig fichu (595-715, 478-568)", 595,715,478,568,orig)
s("orig throat skin (560-595, 460-540)", 560,595,460,540,orig)
s("orig knit chest (900-1000, 350-550)", 900,1000,350,550,orig)
