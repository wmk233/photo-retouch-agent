// Real-browser acceptance for the photo upload path.
// Drives actual mouse clicks + the OS file chooser (intercepted) in headless
// Chrome, then checks what the page really renders and what the API really
// returned. Not a mock: it uses data/inputs/IMG_3651.jpg and a real PNG.
import { spawn } from "node:child_process";
import { mkdtempSync, writeFileSync, mkdirSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const APP = "http://127.0.0.1:8000/";
const PORT = 9233;
const OUT = "evidence/mvp-acceptance";
const JPG = join(process.cwd(), process.argv[2] || "data/inputs/IMG_3651.jpg");
const PNG = join(process.cwd(), process.argv[3] || "evidence/mvp-acceptance/test-portrait.png");
const PROFILE = mkdtempSync(join(tmpdir(), "verify-chrome-"));
mkdirSync(OUT, { recursive: true });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const log = (tag, value) => console.log(tag, typeof value === "string" ? value : JSON.stringify(value));

const chrome = spawn(CHROME, [
  "--headless=new", "--disable-gpu", "--no-first-run", "--window-size=1440,1100",
  `--remote-debugging-port=${PORT}`, `--user-data-dir=${PROFILE}`, "about:blank",
], { stdio: "ignore" });

let targets = null;
for (let i = 0; i < 60; i++) {
  try {
    const r = await fetch(`http://127.0.0.1:${PORT}/json/list`);
    targets = await r.json();
    if (targets.some((t) => t.type === "page")) break;
  } catch {}
  await sleep(250);
}
const page = targets.find((t) => t.type === "page");
const ws = new WebSocket(page.webSocketDebuggerUrl);
let id = 0;
const pend = new Map();
const events = [];
const requests = new Map();
const responses = [];
ws.addEventListener("message", (ev) => {
  const m = JSON.parse(ev.data);
  if (m.method) events.push(m);
  if (m.method === "Network.requestWillBeSent") {
    requests.set(m.params.requestId, { method: m.params.request.method, url: m.params.request.url });
  }
  if (m.method === "Network.responseReceived") {
    const info = requests.get(m.params.requestId) || { method: "?", url: m.params.response.url };
    responses.push({ ...info, status: m.params.response.status, requestId: m.params.requestId });
  }
  if (m.id && pend.has(m.id)) {
    const { resolve, reject } = pend.get(m.id);
    pend.delete(m.id);
    m.error ? reject(new Error(JSON.stringify(m.error))) : resolve(m.result);
  }
});
await new Promise((r) => ws.addEventListener("open", r));
const send = (method, params = {}) => {
  const i = ++id;
  return new Promise((res, rej) => { pend.set(i, { resolve: res, reject: rej }); ws.send(JSON.stringify({ id: i, method, params })); });
};
const ev = async (expression) => {
  const r = await send("Runtime.evaluate", { expression, returnByValue: true, awaitPromise: true });
  if (r.exceptionDetails) return "EXC:" + (r.exceptionDetails.exception?.description || r.exceptionDetails.text);
  return r.result.value;
};
const shot = async (name) => {
  const r = await send("Page.captureScreenshot", { format: "png" });
  const path = join(OUT, name);
  writeFileSync(path, Buffer.from(r.data, "base64"));
  log("[screenshot]", path);
};
const clickSelector = async (selector) => {
  const box = await ev(`(() => { const el = document.querySelector(${JSON.stringify(selector)});
    if (!el) return null; const r = el.getBoundingClientRect();
    return { x: r.x + r.width / 2, y: r.y + r.height / 2, w: r.width, h: r.height }; })()`);
  if (!box || typeof box === "string") return { ok: false, box };
  const x = Math.round(box.x), y = Math.round(box.y);
  await send("Input.dispatchMouseEvent", { type: "mouseMoved", x, y });
  await send("Input.dispatchMouseEvent", { type: "mousePressed", x, y, button: "left", clickCount: 1 });
  await send("Input.dispatchMouseEvent", { type: "mouseReleased", x, y, button: "left", clickCount: 1 });
  return { ok: true, box };
};

async function chooseFileVia(selector, path) {
  const seen = () => events.filter((m) => m.method === "Page.fileChooserOpened");
  const before = seen().length;
  const click = await clickSelector(selector);
  let chooser = null;
  for (let i = 0; i < 40; i++) {
    if (seen().length > before) { chooser = seen()[seen().length - 1]; break; }
    await sleep(150);
  }
  if (!chooser) return { opened: false, click };
  await send("DOM.setFileInputFiles", { backendNodeId: chooser.params.backendNodeId, files: [path] });
  return { opened: true, mode: chooser.params.mode, click };
}

const pageState = () => ev(`({
  docName: document.querySelector('#documentName')?.textContent,
  docMeta: document.querySelector('#documentMeta')?.textContent,
  beforeSrc: (document.querySelector('#beforeImage')?.getAttribute('src')||'').slice(0,12),
  naturalW: document.querySelector('#beforeImage')?.naturalWidth,
  naturalH: document.querySelector('#beforeImage')?.naturalHeight,
  frame: (() => { const r = document.querySelector('#imageWorkbench')?.getBoundingClientRect();
    return r ? { w: Math.round(r.width), h: Math.round(r.height) } : null; })(),
  objectFit: getComputedStyle(document.querySelector('#beforeImage')).objectFit,
  dataLoaded: document.querySelector('#imageWorkbench')?.dataset.loaded,
  emptyVisible: !!document.querySelector('#workbenchEmpty')?.getClientRects().length,
  detectionLabel: document.querySelector('#detectionLabel')?.textContent,
  faceStats: [...document.querySelectorAll('.face-stats div')].map(d => d.textContent.replace(/\\s+/g,' ').trim()),
  portraitName: document.querySelector('#portraitName')?.textContent,
  portraitMeta: document.querySelector('#portraitMeta')?.textContent,
  canvasStatus: document.querySelector('#canvasStatus')?.textContent.replace(/\\s+/g,' ').trim(),
  footer: document.querySelector('#footerStatus')?.textContent,
})`);

const uploadResponse = async () => {
  const hit = [...responses].reverse().find((r) => r.url.includes("/api/photos/upload"));
  if (!hit) return null;
  let body = null;
  try {
    const raw = await send("Network.getResponseBody", { requestId: hit.requestId });
    body = JSON.parse(raw.base64Encoded ? Buffer.from(raw.body, "base64").toString() : raw.body);
  } catch (error) { body = { error: String(error.message) }; }
  return { status: hit.status, body };
};

const paidCalls = () => [...new Set(
  responses
    .filter((r) => /analyze|proposals|retouch\/jobs|styling|runninghub/i.test(r.url))
    .map((r) => `${r.method} ${r.status} ${r.url.replace('http://127.0.0.1:8000', '')}`),
)];

await send("Page.enable");
await send("Runtime.enable");
await send("DOM.enable");
await send("Network.enable");
await send("Page.setInterceptFileChooserDialog", { enabled: true });
await send("Emulation.setDeviceMetricsOverride", { width: 1440, height: 1100, deviceScaleFactor: 1, mobile: false });
await send("Page.navigate", { url: APP });
for (let i = 0; i < 40; i++) { if (await ev("!!document.querySelector('#uploadButton')")) break; await sleep(300); }
await sleep(1500);

log("[1 initial]", await pageState());

log("[2 click central frame]", await chooseFileVia("#imageWorkbench", JPG));
await sleep(2500);
log("[3 after JPG]", await pageState());
log("[4 upload api]", await uploadResponse());
await shot("verify-01-jpg-uploaded.png");

// --- switch to a real PNG through the rail "+ 换图" control ---------------
log("[5 click rail upload]", await chooseFileVia("#uploadButton", PNG));
await sleep(2500);
log("[6 after PNG]", await pageState());
log("[7 upload api]", await uploadResponse());

// --- the two uploads must be distinct backend records --------------------
const ids = responses
  .filter((r) => r.url.includes("/api/photos/upload"))
  .map((r) => r.requestId)
  .filter((v, i, a) => a.indexOf(v) === i);
log("[8 upload request count]", ids.length);
await shot("verify-02-png-uploaded.png");

// --- AI styling entrance: must be reachable and free in template mode ----
await clickSelector('[data-mode="ai"]');
await sleep(600);
log("[9 ai tab]", await ev(`({
  mode: document.querySelector('#modeNote')?.textContent.replace(/\\s+/g,' ').trim(),
  aiPanelVisible: !!document.querySelector('#aiPanel')?.getClientRects().length,
  designButton: !!document.querySelector('#stylingDesignButton'),
  templateButton: !!document.querySelector('#stylingTemplateButton'),
})`));
await clickSelector("#stylingTemplateButton");
for (let i = 0; i < 60; i++) {
  if (await ev("document.querySelector('#stylingGrid')?.children.length")) break;
  await sleep(500);
}
log("[10 three looks]", await ev(`({
  cards: document.querySelector('#stylingGrid')?.children.length,
  titles: [...document.querySelectorAll('#stylingGrid > *')].map(c => (c.querySelector('strong,h3,h4')?.textContent || c.textContent).replace(/\\s+/g,' ').trim().slice(0, 40)),
  feedback: document.querySelector('#stylingFeedback')?.textContent,
  detectionLabel: document.querySelector('#detectionLabel')?.textContent,
  faceStats: [...document.querySelectorAll('.face-stats div')].map(d => d.textContent.replace(/\\s+/g,' ').trim()),
})`));
log("[11 paid/vision api calls]", paidCalls());
await shot("verify-03-three-looks.png");

// --- what does a refresh do? --------------------------------------------
await send("Page.navigate", { url: APP });
for (let i = 0; i < 40; i++) { if (await ev("!!document.querySelector('#uploadButton')")) break; await sleep(300); }
await sleep(1200);
log("[12 after reload]", await pageState());

const errs = events
  .filter((m) => m.method === "Runtime.exceptionThrown")
  .map((m) => m.params.exceptionDetails.exception?.description || m.params.exceptionDetails.text);
log("[13 console exceptions]", errs.length ? errs : "(none)");
log("[14 all /api/ responses]", [...new Set(responses.filter((r) => r.url.includes("/api/")).map((r) => `${r.method} ${r.status} ${r.url.replace("http://127.0.0.1:8000", "")}`))]);

ws.close();
chrome.kill();
process.exit(0);
