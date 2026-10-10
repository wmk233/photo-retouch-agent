from pathlib import Path
from PIL import Image, ImageDraw
REPO = Path(__file__).resolve().parents[2]; OUT = Path(__file__).resolve().parent
final = Image.open(REPO/'evidence/p29-submit-run/final.png').convert('RGB')
orig  = Image.open(REPO/'data/inputs/IMG_3651.jpg').convert('RGB')
x0,y0,x1,y1 = 415,575,595,745
sc=10; W,H=(x1-x0)*sc,(y1-y0)*sc
def p(im,lab):
    q=im.crop((x0,y0,x1,y1)).resize((W,H),Image.LANCZOS)
    d=ImageDraw.Draw(q); d.rectangle([0,0,len(lab)*7+10,16],fill=(0,0,0)); d.text((4,3),lab,fill=(255,255,0))
    return q
sheet=Image.new('RGB',(W*2+6,H),(20,20,20))
sheet.paste(p(orig,'ORIGINAL x415-595 y575-745  x10'),(0,0))
sheet.paste(p(final,'FINAL x415-595 y575-745  x10'),(W+6,0))
sheet.save(OUT/'clean_orig_final_10x.png'); print('wrote', sheet.size)
