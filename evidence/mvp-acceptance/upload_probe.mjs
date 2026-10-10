// Reproduces the REAL user upload path: a genuine mouse click on the upload
// control + the OS file chooser (intercepted), then verifies the photo actually
// renders and what network calls the selection triggers.
import { spawn } from "node:child_process";
import { mkdtempSync, writeFileSync, mkdirSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
const CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const APP = "http://127.0.0.1:8000/";
const PORT = 9231;
const PHOTO = process.argv[2] || "data/inputs/IMG_3651.jpg";
const OUT = "evidence/mvp-acceptance";
const PROFILE = mkdtempSync(join(tmpdir(), "probe-chrome-"));
mkdirSync(OUT, { recursive: true });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const chrome = spawn(CHROME, [
  "--headless=new", "--disable-gpu", "--no-first-run", "--window-size=1440,1100",
  `--remote-debugging-port=${PORT}`, `--user-data-dir=${PROFILE}`, "about:blank",
], { stdio: "ignore" });

let targets = null;
for (let i = 0; i < 60; i++) {
  try { const r = await fetch(`http://127.0.0.1:${PORT}/json/list`); targets = await r.json();
    if (targets.some((t) => t.type === "page")) break; } catch {}
  await sleep(250);
}
const page = targets.find((t) => t.type === "page");
const ws = new WebSocket(page.webSocketDebuggerUrl);
let id = 0; const pend = new Map(); const events = []; const net = [];
ws.addEventListener("message", (ev) => {
  const m = JSON.parse(ev.data);
  if (m.method) events.push(m);
  if (m.method === "Network.responseReceived") {
    const { status, url } = m.params.response;
    if (url.includes("/api/") || status >= 400) net.push(`${status} ${m.params.response.requestHeaders ? "" : ""}${url}`);
  }
  if (m.id && pend.has(m.id)) { const { resolve, reject } = pend.get(m.id); pend.delete(m.id);
    m.error ? reject(new Error(JSON.stringify(m.error))) : resolve(m.result); }
});
await new Promise((r) => ws.addEventListener("open", r));
const send = (method, params = {}) => { const i = ++id;
  return new Promise((res, rej) => { pend.set(i, { resolve: res, reject: rej }); ws.send(JSON.stringify({ id: i, method, params })); }); };
const ev = async (e) => { const r = await send("Runtime.evaluate", { expression: e, returnByValue: true, awaitPromise: true });
  if (r.exceptionDetails) return "EXC:" + (r.exceptionDetails.exception?.description || r.exceptionDetails.text); return r.result.value; };

await send("Page.enable"); await send("Runtime.enable"); await send("DOM.enable"); await send("Network.enable");
await send("Page.setInterceptFileChooserDialog", { enabled: true });
await send("Emulation.setDeviceMetricsOverride", { width: 1440, height: 1100, deviceScaleFactor: 1, mobile: false });
await send("Page.navigate", { url: APP });
for (let i = 0; i < 40; i++) { if (await ev("!!document.querySelector('#uploadButton')")) break; await sleep(300); }
await sleep(1500);

const before = await ev(`({
  beforeSrc: document.querySelector('#beforeImage').getAttribute('src'),
  naturalW: document.querySelector('#beforeImage').naturalWidth,
  docName: document.querySelector('#documentName').textContent,
  uploadVisible: !!document.querySelector('#uploadButton')?.getClientRects().length,
  uploadText: document.querySelector('#uploadButton')?.innerText.replace(/\\n/g,' ').trim(),
  detectionStatus: document.querySelector('.detection-card .success-label')?.textContent,
})`);
console.log("[initial]", JSON.stringify(before));

// Real mouse click on the upload button.
const box = await ev(`(() => { const r = document.querySelector('#uploadButton').getBoundingClientRect();
  return { x: r.x + r.width/2, y: r.y + r.height/2, w: r.width, h: r.height }; })()`);
console.log("[uploadButton box]", JSON.stringify(box));
const cx = Math.round(box.x), cy = Math.round(box.y);
await send("Input.dispatchMouseEvent", { type: "mousePressed", x: cx, y: cy, button: "left", clickCount: 1 });
await send("Input.dispatchMouseEvent", { type: "mouseReleased", x: cx, y: cy, button: "left", clickCount: 1 });
await sleep(1200);
const chooser = events.find((m) => m.method === "Page.fileChooserOpened");
console.log("[fileChooserOpened]", chooser ? JSON.stringify({ mode: chooser.params.mode, backendNodeId: chooser.params.backendNodeId }) : "NO EVENT (chooser did not open)");

if (chooser) {
  await send("DOM.setFileInputFiles", { backendNodeId: chooser.params.backendNodeId, files: [join(process.cwd(), PHOTO)] });
}
await sleep(2500);

const after = await ev(`({
  beforeSrcHead: (document.querySelector('#beforeImage').getAttribute('src')||'').slice(0,12),
  naturalW: document.querySelector('#beforeImage').naturalWidth,
  naturalH: document.querySelector('#beforeImage').naturalHeight,
  complete: document.querySelector('#beforeImage').complete,
  docName: document.querySelector('#documentName').textContent,
  detectionStatus: document.querySelector('.detection-card .success-label')?.textContent,
  faceCount: document.querySelectorAll('.face-stats div')[0]?.textContent,
  canvasStatus: document.querySelector('#canvasStatus')?.innerText.replace(/\\n/g,' ').trim(),
})`);
console.log("[after select]", JSON.stringify(after));
console.log("[network >=400 or /api/]", [...new Set(net)].join(" || ") || "(none)");
const errs = events.filter((m) => m.method === "Runtime.exceptionThrown").map((m) => m.params.exceptionDetails.exception?.description || m.params.exceptionDetails.text);
console.log("[exceptions]", errs.join(" || ") || "(none)");

const shot = await send("Page.captureScreenshot", { format: "png" });
writeFileSync(join(OUT, "probe-after-upload.png"), Buffer.from(shot.data, "base64"));
console.log("[screenshot]", join(OUT, "probe-after-upload.png"));
ws.close(); chrome.kill(); process.exit(0);
