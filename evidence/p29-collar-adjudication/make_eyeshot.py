from pathlib import Path
from PIL import Image, ImageDraw
REPO = Path(__file__).resolve().parents[2]; OUT = Path(__file__).resolve().parent
final = Image.open(REPO/'evidence/p29-submit-run/final.png').convert('RGB')
orig  = Image.open(REPO/'data/inputs/IMG_3651.jpg').convert('RGB')
x0,y0,x1,y1 = 420,570,600,700   # neckline only
a = final.crop((x0,y0,x1,y1))   # 1x
b = a.resize((a.width*2, a.height*2), Image.LANCZOS)   # 2x
c = orig.crop((x0,y0,x1,y1)).resize((a.width*2,a.height*2), Image.LANCZOS)
sheet = Image.new('RGB', (a.width + b.width + c.width + 12, max(a.height, b.height)), (30,30,30))
sheet.paste(a, (0,0)); sheet.paste(b, (a.width+6, 0)); sheet.paste(c, (a.width+b.width+12, 0))
d = ImageDraw.Draw(sheet)
for x,lab in ((0,'FINAL 1x'), (a.width+6,'FINAL 2x'), (a.width+b.width+12,'ORIGINAL 2x')):
    d.rectangle([x,0,x+len(lab)*7+10,15], fill=(0,0,0)); d.text((x+4,3),lab,fill=(255,255,0))
sheet.save(OUT/'eyeshot_neckline.png'); print('wrote', sheet.size)
