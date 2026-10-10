from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "evidence/p29-submit-run"
SRC = [
    ("ORIGINAL", ROOT / "data/inputs/IMG_3651.jpg"),
    ("LAST FAILED (pale ghost)", ROOT / "artifacts/p29-fix-run/result_0.png"),
    ("THIS RUN (2108330225116082177)", OUT / "final.png"),
]
X0, Y0, X1, Y1 = 300, 470, 680, 840
S = 3
tiles = []
for label, path in SRC:
    im = Image.open(path).convert("RGB").crop((X0, Y0, X1, Y1))
    im = im.resize((im.width * S, im.height * S), Image.NEAREST)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, im.width - 1, 26], fill=(0, 0, 0))
    d.text((6, 7), label, fill=(255, 255, 255))
    tiles.append(im)
w = sum(t.width for t in tiles) + 12 * (len(tiles) - 1)
h = max(t.height for t in tiles)
canvas = Image.new("RGB", (w, h), (24, 24, 24))
x = 0
for t in tiles:
    canvas.paste(t, (x, 0))
    x += t.width + 12
canvas.save(OUT / "throat_zoom3x.png")
print("saved", OUT / "throat_zoom3x.png", canvas.size)
