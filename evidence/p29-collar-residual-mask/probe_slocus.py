import numpy as np
from PIL import Image
REPO="/Users/liuxiuyu/AI-beauty-style-lab/repos/photo-retouch-agent"
WORK=REPO+"/evidence/p29-collar-residual-mask"
d=np.load(f"{WORK}/_masks.npy",allow_pickle=True).item()
blob=np.load(f"{WORK}/_blob.npy"); sample=np.load(f"{WORK}/_sample.npy")
skin=d['skin']; ol=d['ol']; oc=d['oc']
H,W=ol.shape
print("== the subject's OWN skin: does it ever look like the unconfirmed band (luma~185, chroma~50)? ==")
s=skin
print(f"skin model px {int(s.sum())}")
print(f"  skin luma   p10/p50/p90 = {np.percentile(ol[s],[10,50,90]).round(0)}")
print(f"  skin chroma p10/p50/p90 = {np.percentile(oc[s],[10,50,90]).round(1)}")
hi = s & (ol>170)
print(f"  skin with luma>170: {int(hi.sum())}px  chroma p10/p50/p90 = {np.percentile(oc[hi],[10,50,90]).round(1)}  luma p50={np.median(ol[hi]):.0f}")
lo = s & (oc<60)
print(f"  skin with chroma<60: {int(lo.sum())}px  luma p50={np.median(ol[lo]) if lo.sum() else float('nan')}")
print()
print("== the unconfirmed band and the block ==")
band = blob & ~sample
print(f"unreleased block px {int(band.sum())}  luma p10/p50/p90 = {np.percentile(ol[band],[10,50,90]).round(0)}  chroma p10/p50/p90 = {np.percentile(oc[band],[10,50,90]).round(1)}")
print()
print("== overlap: how much of the unreleased block is inside the subject's own skin locus? ==")
print(f"  unreleased block & skin model                {int((band&skin).sum())} / {int(band.sum())}")
print(f"  unreleased block & (luma>170 & chroma<60)    {int((band&(ol>170)&(oc<60)).sum())}")
print(f"  the skin model's own (luma>170 & chroma<60)  {int((skin&(ol>170)&(oc<60)).sum())}  <- skin that actually looks like the band")
print()
print("== where is the subject's skin locus in this crop? luma-vs-chroma scatter of pure skin ==")
for lo_,hi_ in [(120,150),(150,170),(170,190),(190,215)]:
    m=s&(ol>=lo_)&(ol<hi_)
    if m.sum()>30: print(f"  skin luma {lo_}-{hi_}: n={int(m.sum()):6d} chroma p50={np.median(oc[m]):5.1f}  (band is 50-72)")
