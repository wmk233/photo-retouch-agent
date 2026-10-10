from __future__ import annotations
import numpy as np, cv2
from pathlib import Path
from PIL import Image
exec(open("probe_rule3.py").read().split("# overlay image")[0])
def stat(m,tag):
    print(f"{tag:34s} n={int(m.sum()):5d}  ORIG L={np.percentile(oL[m],50):5.1f} C={np.percentile(oC[m],50):5.1f} H={np.percentile(hC[m],50):5.1f} | "
          f"FINAL L={np.percentile(fL[m],50):5.1f} C={np.percentile(fC[m],50):5.1f} " if m.any() else tag)
# recompute final hue for reporting
def HSf(a):
    a=a.astype(np.float32)/255.0; mx,mn=a.max(2),a.min(2); d=mx-mn
    sat=np.where(mx>1e-6,d/np.maximum(mx,1e-6),0.0)
    r,g,b=a[...,0],a[...,1],a[...,2]; h=np.zeros_like(mx); m=d>1e-6
    i=(mx==r)&m; h[i]=(60*((g[i]-b[i])/d[i])+360)%360
    i=(mx==g)&m; h[i]=60*((b[i]-r[i])/d[i])+120
    i=(mx==b)&m; h[i]=60*((r[i]-g[i])/d[i])+240
    return h, np.where(m,sat,0.0)
fh,fs=HSf(f)
print("                 region                      ORIG med(L,C,H)         FINAL med(L,C,H)")
for m,tag in ((release,"RELEASE (new)"),(keep,"KEEP total")):
    print(f"  {tag:22s} n={int(m.sum()):5d}  O(L={np.median(oL[m]):5.1f} C={np.median(oC[m]):5.1f} H={np.median(hC[m]):5.1f})  F(L={np.median(fL[m]):5.1f} C={np.median(fC[m]):5.1f} H={np.median(fh[m]):5.1f})")
# named regions
regs={
 "keep oC>=60 (skin-evidenced)": keep&(oC>=60),
 "keep oL<=145 (dim)":           keep&(oL<=145)&(oC<60),
 "keep not-in-mass":             keep&~mass&(oC<60)&(oL>145),
 "keep near-skin<=2px":          keep&(dist<=2)&(oC<60)&(oL>145)&mass,
 "release":                      release,
 "OLD release":                  rel,
 "neck skin (final, warm C>=60)":(fC>=60)&np.pad(np.ones((120,140),bool),((590,1190-710),(430,904-570))),
 "blouse olive (final)":         (fh>=45)&(fh<=115)&(fs>0.12),
}
for tag,m in regs.items():
    m=m&True
    print(f"  {tag:30s} n={int(m.sum()):6d}  O(L={np.median(oL[m]):6.1f} C={np.median(oC[m]):6.1f} H={np.median(hC[m]):6.1f})  F(L={np.median(fL[m]):6.1f} C={np.median(fC[m]):6.1f} H={np.median(fh[m]):6.1f})")
# the kept blob 482 @437,601,91,67
b=np.zeros_like(keep); b[601:601+67,437:437+91]=True; b&=keep
print("\nkept blob[437,601,91,67] n=",int(b.sum()))
print("  ORIG L/C/H med",np.median(oL[b]),np.median(oC[b]),np.median(hC[b]))
print("  FINAL L/C/H med",np.median(fL[b]),np.median(fC[b]),np.median(fh[b]))
print("  d(orig-final) med",np.median(np.linalg.norm(o.astype(np.float32)-f.astype(np.float32),axis=2)[b]))
