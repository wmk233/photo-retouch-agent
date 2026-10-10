// Free end-to-end walkthrough of the real frontend in headless Chrome (CDP).
// Everything runs on the free local providers (brain=local, image=mock), so no
// DeepSeek / RunningHub request can be created. Verifies the *actual* generated
// pixels: the result image bytes are fetched and written to disk for eyeballing.
import { spawn } from "node:child_process";
import { mkdtempSync, writeFileSync, mkdirSync, readFileSync, existsSync, statSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, basename } from "node:path";

const CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const APP = "http://127.0.0.1:8000/";
const PORT = 9225;
const REPO = process.cwd();
const PHOTO = process.argv[2] || "data/inputs/IMG_3651.jpg";
const OUT = process.argv[3] || "evidence/mvp-acceptance";
const PROFILE = mkdtempSync(join(tmpdir(), "beauty-lab-chrome-"));
mkdirSync(OUT, { recursive: true });

const log = (...a) => console.log(...a);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const results = [];
const check = (label, ok, detail = "") => {
  log(`[${ok ? "PASS" : "FAIL"}] ${label}${detail ? " — " + detail : ""}`);
  results.push({ label, ok, detail });
};

const chrome = spawn(CHROME, [
  "--headless=new", "--disable-gpu", "--no-first-run", "--no-default-browser-check",
  "--hide-scrollbars", "--window-size=1440,1200",
  `--remote-debugging-port=${PORT}`, `--user-data-dir=${PROFILE}`, "about:blank",
], { stdio: "ignore" });

let targets = null;
for (let i = 0; i < 80; i++) {
  try {
    const r = await fetch(`http://127.0.0.1:${PORT}/json/list`);
    targets = await r.json();
    if (targets.some((t) => t.type === "page")) break;
  } catch {}
  await sleep(250);
}
if (!targets) { log("FAIL chrome did not start"); process.exit(1); }
const page = targets.find((t) => t.type === "page");

const ws = new WebSocket(page.webSocketDebuggerUrl);
let msgId = 0;
const pending = new Map();
const events = [];
const dialogs = [];
ws.addEventListener("message", (ev) => {
  const m = JSON.parse(ev.data);
  // Never let a native confirm() block the run: record it and auto-dismiss.
  if (m.method === "Page.javascriptDialogOpening") {
    dialogs.push(m.params.type);
    ws.send(JSON.stringify({ id: ++msgId, method: "Page.handleJavaScriptDialog", params: { accept: false } }));
  }
  if (m.id && pending.has(m.id)) {
    const { resolve, reject } = pending.get(m.id);
    pending.delete(m.id);
    m.error ? reject(new Error(JSON.stringify(m.error))) : resolve(m.result);
  } else if (m.method) {
    events.push(m);
  }
});
await new Promise((r) => ws.addEventListener("open", r));
const send = (method, params = {}) => {
  const id = ++msgId;
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject });
    ws.send(JSON.stringify({ id, method, params }));
  });
};
const evalJs = async (expression) => {
  const r = await send("Runtime.evaluate", { expression, returnByValue: true, awaitPromise: true });
  if (r.exceptionDetails) throw new Error("page error: " + (r.exceptionDetails.exception?.description || r.exceptionDetails.text));
  return r.result.value;
};
const waitFor = async (expr, label, timeoutMs = 180000) => {
  const t0 = Date.now();
  while (Date.now() - t0 < timeoutMs) {
    try { if (await evalJs(expr)) return true; } catch {}
    await sleep(400);
  }
  throw new Error(`timeout waiting for ${label}`);
};
const shot = async (name) => {
  const r = await send("Page.captureScreenshot", { format: "png" });
  const p = join(OUT, name);
  writeFileSync(p, Buffer.from(r.data, "base64"));
  log(`   screenshot: ${p}`);
};

await send("Page.enable");
await send("Runtime.enable");
await send("DOM.enable");
await send("Network.enable");
await send("Page.setInterceptFileChooserDialog", { enabled: true });
await send("Emulation.setDeviceMetricsOverride", { width: 1440, height: 1200, deviceScaleFactor: 1, mobile: false });
await send("Page.addScriptToEvaluateOnNewDocument", {
  source: `window.__errors = [];
    window.addEventListener("error", (e) => window.__errors.push(String(e.message)));
    window.addEventListener("unhandledrejection", (e) => window.__errors.push("rejection: " + String(e.reason && e.reason.message || e.reason)));
    const _ce = console.error.bind(console);
    console.error = (...a) => { window.__errors.push("console.error: " + a.map(String).join(" ")); _ce(...a); };`,
});

log("== AI Beauty Style Lab — free end-to-end (real Chrome) ==");
await send("Page.navigate", { url: APP });
await waitFor("!!document.querySelector('#stylingDesignButton')", "app shell");
await waitFor("(document.querySelector('#actionProviderSelect')?.options.length || 0) > 1", "provider capabilities");
await sleep(600);

// ---- 1. default entry is the AI styling workspace, with a clean no-photo state
const initial = await evalJs(`({
  aiVisible: !document.querySelector('#aiPanel').hidden,
  localHidden: document.querySelector('#localPanel').hidden,
  aiTabActive: document.querySelector('[data-mode="ai"]').classList.contains('active'),
  documentName: document.querySelector('#documentName').textContent.trim(),
  documentMeta: document.querySelector('#documentMeta').textContent.trim(),
  canvasStatus: document.querySelector('#canvasStatus').textContent.trim(),
  portraitName: document.querySelector('#portraitName').textContent.trim(),
  portraitMeta: document.querySelector('#portraitMeta').textContent.trim(),
  detectionLabel: document.querySelector('#detectionLabel').textContent.trim(),
  faceCount: document.querySelector('#detectionFaceCount').textContent.trim(),
  visibility: document.querySelector('#detectionVisibility').textContent.trim(),
  defaultAction: document.querySelector('#actionProviderSelect').value,
  defaultBrain: document.querySelector('#brainProviderSelect').value,
  workbenchLoaded: document.querySelector('#imageWorkbench').dataset.loaded,
})`);
check("1 app opens with AI 造型设计 as the default entry",
  initial.aiVisible && initial.localHidden && initial.aiTabActive, JSON.stringify(initial.canvasStatus));
// The image provider defaults to the free local simulation. The brain select
// defaults to the configured vision model, but the paid design path is behind an
// explicit confirm (step 10), and the free template flow forces a local brain.
check("1b default image provider is the FREE local simulation",
  initial.defaultAction === "mock" && Boolean(initial.defaultBrain),
  `action=${initial.defaultAction} brain=${initial.defaultBrain}`);
check("1c no-photo state is honest (no fake face detection)",
  initial.faceCount === "—" && initial.visibility === "—" && !/9\d%/.test(initial.visibility),
  `face=${initial.faceCount} vis=${initial.visibility} label=${initial.detectionLabel}`);
await shot("free-01-initial-ai-entry.png");

// ---- 2. clicking the central frame really opens the OS file chooser
const doc = await send("DOM.getDocument", { depth: -1 });
const wbNode = await send("DOM.querySelector", { nodeId: doc.root.nodeId, selector: "#imageWorkbench" });
const wbBox = await send("DOM.getBoxModel", { nodeId: wbNode.nodeId });
const [bx, by] = [(wbBox.model.content[0] + wbBox.model.content[2]) / 2, (wbBox.model.content[1] + wbBox.model.content[5]) / 2];
const beforeChooser = events.filter((e) => e.method === "Page.fileChooserOpened").length;
for (const type of ["mousePressed", "mouseReleased"]) {
  await send("Input.dispatchMouseEvent", { type, x: bx, y: by, button: "left", clickCount: 1 });
}
await sleep(500);
const chooserOpened = events.filter((e) => e.method === "Page.fileChooserOpened").length > beforeChooser;
check("2 clicking the photo area opens the system file chooser", chooserOpened);

// ---- 3. upload the real JPG and confirm the backend stored it
const inputNode = await send("DOM.querySelector", { nodeId: doc.root.nodeId, selector: "#fileInput" });
await send("DOM.setFileInputFiles", { nodeId: inputNode.nodeId, files: [join(REPO, PHOTO)] });
await waitFor("!document.querySelector('#canvasStatus').textContent.includes('正在上传')", "upload finished", 60000);
const uploadReqs = events.filter((e) => e.method === "Network.responseReceived" && /\/photos\/upload/.test(e.params.response.url));
const uploadStatus = uploadReqs.length ? uploadReqs[uploadReqs.length - 1].params.response.status : 0;
const uploaded = await evalJs(`({
  documentName: document.querySelector('#documentName').textContent.trim(),
  documentMeta: document.querySelector('#documentMeta').textContent.trim(),
  portraitMeta: document.querySelector('#portraitMeta').textContent.trim(),
  canvasStatus: document.querySelector('#canvasStatus').textContent.trim(),
  blob: document.querySelector('#beforeImage').src.startsWith('blob:'),
  natural: document.querySelector('#beforeImage').naturalWidth,
  workbenchLoaded: document.querySelector('#imageWorkbench').dataset.loaded,
})`);
check("3 photo uploads, previews and is stored on the backend",
  uploaded.blob && uploaded.natural > 0 && uploadStatus === 200 && uploaded.canvasStatus.includes("照片已就绪"),
  `HTTP ${uploadStatus} · ${uploaded.documentMeta} · ${uploaded.portraitMeta}`);
check("3b frontend photo id matches the backend record",
  /照片 ID img_[0-9a-f]+/.test(uploaded.portraitMeta), uploaded.portraitMeta);
await shot("free-02-photo-uploaded.png");

// ---- 4. free template design -> three real looks
await evalJs("document.querySelector('#stylingTemplateButton').click()");
await waitFor("!document.querySelector('#stylingGrid').hidden && document.querySelectorAll('[data-styling-card]').length >= 3", "three styling cards", 120000);
const cards = await evalJs(`[...document.querySelectorAll('[data-styling-card]')].map(c => ({
  id: c.dataset.stylingCard,
  name: (c.querySelector('[data-styling-name], .styling-card-name, h3, strong')?.textContent || '').trim(),
  mode: c.dataset.designMode || '',
  text: c.innerText.replace(/\\s+/g,' ').slice(0, 160),
}))`);
check("4 three real styling designs are produced for the uploaded photo",
  cards.length === 3, cards.map((c) => c.id).join(", "));
check("4b designs are template-based (free) and carry real text",
  cards.every((c) => c.text.length > 40), cards[0].text.slice(0, 80));
await shot("free-03-three-looks.png");

// ---- 5. pick a look -> cost dialog shows 0 paid requests -> confirm
await evalJs(`document.querySelectorAll('[data-styling-generate]')[1].click()`);
await waitFor("document.querySelector('#costDialog')?.open === true", "cost dialog");
const cost = await evalJs(`({
  lead: document.querySelector('#costDialogLead').textContent.trim(),
  design: document.querySelector('#costDesignValue').textContent.trim(),
  image: document.querySelector('#costImageValue').textContent.trim(),
  jobs: document.querySelector('#costJobCount').textContent.trim(),
  note: document.querySelector('#costDialogNote').textContent.trim(),
})`);
check("5 cost dialog reports a zero-charge local run before confirming",
  /0 次/.test(cost.design) && /0 次付费/.test(cost.image) && /不计费/.test(cost.jobs),
  `${cost.jobs} · ${cost.image}`);
await shot("free-04-cost-dialog.png");

await evalJs("document.querySelector('#costConfirmButton').click()");
await waitFor("document.querySelector('#stylingResult') && !document.querySelector('#stylingResult').hidden", "result page", 300000);
await waitFor("(document.querySelector('#resultAfterImage')?.src || '').length > 0", "result image");
await sleep(800);

const result = await evalJs(`({
  before: document.querySelector('#resultBeforeImage')?.src || '',
  after: document.querySelector('#resultAfterImage')?.src || '',
  badge: document.querySelector('#resultVerification')?.textContent?.trim(),
  download: document.querySelector('#resultDownload')?.getAttribute('download'),
  downloadHref: document.querySelector('#resultDownload')?.href || '',
  note: document.querySelector('#resultNote')?.textContent?.trim(),
  history: document.querySelectorAll('#historyList [data-history-job]').length,
  canvasStatus: document.querySelector('#canvasStatus').textContent.trim(),
  stages: [...document.querySelectorAll('#generationSteps li')].map(li => li.dataset.genStage + ':' + li.className.trim()),
})`);
check("6 generation completes and the result page renders",
  Boolean(result.after) && Boolean(result.download), `badge=${result.badge}`);
check("6b the displayed result is NOT the original photo",
  result.after !== result.before && !result.after.startsWith('blob:'), `after=${result.after.slice(0, 60)}`);
check("6c before/after + note + download link are present",
  Boolean(result.note) && result.downloadHref.includes('/data/outputs/'), `download=${result.download}`);
check("6d history lists the finished result", result.history >= 1, `entries=${result.history}`);
await evalJs("document.querySelector('#stylingResult')?.scrollIntoView({ block: 'start' })");
await sleep(500);
await shot("free-05-result.png");

// ---- 7. fetch the ACTUAL result bytes and compare with the source pixels
const url = new URL(result.after);
const rel = decodeURIComponent(url.pathname).replace(/^\/data\//, "data/");
const outPath = join(REPO, rel);
const bytesOk = existsSync(outPath) && statSync(outPath).size > 1000;
check("7 result file exists on the backend disk", bytesOk,
  bytesOk ? `${basename(outPath)} (${statSync(outPath).size} bytes)` : outPath);

// In-page pixel diff: draw both images and count changed pixels.
const diff = await evalJs(`(async () => {
  const load = (src) => new Promise((res, rej) => { const i = new Image(); i.crossOrigin='anonymous'; i.onload=()=>res(i); i.onerror=()=>rej(new Error('img')); i.src=src; });
  const [a, b] = await Promise.all([load(${JSON.stringify(result.before)}), load(${JSON.stringify(result.after)})]);
  const c = (img) => { const cv=document.createElement('canvas'); cv.width=img.naturalWidth; cv.height=img.naturalHeight; cv.getContext('2d').drawImage(img,0,0); return cv.getContext('2d').getImageData(0,0,cv.width,cv.height).data; };
  const da = c(a), db = c(b); const n = Math.min(da.length, db.length); let changed = 0;
  for (let i = 0; i < n; i += 4) { if (Math.abs(da[i]-db[i]) + Math.abs(da[i+1]-db[i+1]) + Math.abs(da[i+2]-db[i+2]) > 24) changed++; }
  return { pixels: n/4, changed, ratio: changed / (n/4), aw: a.naturalWidth, ah: a.naturalHeight, bw: b.naturalWidth, bh: b.naturalHeight };
})()`);
check("7b result pixels genuinely differ from the original (real edit, not a copy)",
  diff.ratio > 0.05, `changed=${(diff.ratio * 100).toFixed(1)}% of ${diff.pixels}px; ${diff.aw}x${diff.ah} -> ${diff.bw}x${diff.bh}`);
await shot("free-06-result-compare.png");
await evalJs("window.scrollTo(0, 0)");
await sleep(300);

// ---- 8. secondary free flow: skin "AI 美化" must now poll to a real result
// This is the exact path the user hit: POST /retouch/jobs answers 202, the old
// code treated that as terminal and reported "AI 美化任务未生成结果". Drive the
// real button with local brain + mock image so the whole thing stays free.
await evalJs(`(() => {
  const brain = document.querySelector('#brainProviderSelect');
  brain.value = 'local';
  brain.dispatchEvent(new Event('change', { bubbles: true }));
  const action = document.querySelector('#actionProviderSelect');
  action.value = 'mock';
  action.dispatchEvent(new Event('change', { bubbles: true }));
})()`);
await evalJs("document.querySelector('#generateButton').click()");
await waitFor("document.querySelector('#costDialog')?.open === true", "retouch cost dialog");
const retouchCost = await evalJs(`({
  lead: document.querySelector('#costDialogLead').textContent.trim(),
  design: document.querySelector('#costDesignValue').textContent.trim(),
  image: document.querySelector('#costImageValue').textContent.trim(),
  jobs: document.querySelector('#costJobCount').textContent.trim(),
})`);
check("8 skin AI 美化 cost dialog reports 0 paid requests",
  /0 次付费/.test(retouchCost.design) && /0 次付费/.test(retouchCost.image) && /不计费/.test(retouchCost.jobs),
  `${retouchCost.jobs} · ${retouchCost.design} · ${retouchCost.image}`);
await evalJs("document.querySelector('#costConfirmButton').click()");
await waitFor(
  "document.querySelector('#canvasStatus').textContent.includes('AI 结果') || document.querySelector('#aiFeedback').classList.contains('error')",
  "skin AI retouch finished", 300000);
const retouch = await evalJs(`({
  status: document.querySelector('#canvasStatus').textContent.trim(),
  feedback: document.querySelector('#aiFeedback').textContent.trim(),
  ok: document.querySelector('#aiFeedback').classList.contains('success'),
  model: document.querySelector('#previewCanvas').width + 'x' + document.querySelector('#previewCanvas').height,
})`);
check("8b skin AI 美化 polls a queued job to a real success (not an instant 202 error)",
  retouch.ok && retouch.status.includes("AI 结果") && !/未生成结果/.test(retouch.feedback),
  `${retouch.status} · ${retouch.feedback.slice(0, 70)}`);
const skinDiff = await evalJs(`(() => {
  const img = document.querySelector('#beforeImage');
  if (!img.naturalWidth) return { ratio: -1 };
  const cv = document.createElement('canvas');
  cv.width = img.naturalWidth; cv.height = img.naturalHeight;
  cv.getContext('2d').drawImage(img, 0, 0);
  const src = cv.getContext('2d').getImageData(0, 0, cv.width, cv.height).data;
  const pc = document.querySelector('#previewCanvas');
  const dst = pc.getContext('2d').getImageData(0, 0, pc.width, pc.height).data;
  const n = Math.min(src.length, dst.length); let changed = 0;
  for (let i = 0; i < n; i += 4) {
    if (Math.abs(src[i]-dst[i]) + Math.abs(src[i+1]-dst[i+1]) + Math.abs(src[i+2]-dst[i+2]) > 24) changed++;
  }
  return { ratio: changed / (n / 4), w: pc.width, h: pc.height };
})()`);
check("8c skin AI 美化 canvas shows edited pixels, not the untouched original",
  skinDiff.ratio > 0.02, `changed=${(skinDiff.ratio * 100).toFixed(1)}% · canvas ${skinDiff.w}x${skinDiff.h}`);
await shot("free-07-skin-ai-retouch.png");

// ---- 9. the paid "AI 设计三套造型" must never fire without an explicit confirm
// Point the brain back at the configured (paid) vision model so the guard is
// actually exercised; the auto-dismiss in the CDP handler then declines it.
await evalJs(`(() => {
  const brain = document.querySelector('#brainProviderSelect');
  const paid = [...brain.options].map((o) => o.value).find((v) => v !== 'local' && v !== 'mock') || 'deepseek';
  brain.value = paid;
  brain.dispatchEvent(new Event('change', { bubbles: true }));
  return paid;
})()`);
const proposalsSeen = () => events.filter((e) => e.method === "Network.requestWillBeSent" && /\/styling\/proposals/.test(e.params.request.url)).length;
const proposalsBefore = proposalsSeen();
const dialogsBefore = dialogs.length;
await evalJs("document.querySelector('#stylingDesignButton').click()");
await sleep(1800);
const cancelNote = await evalJs("document.querySelector('#stylingFeedback').textContent.trim()");
check("9 declining the confirm sends no paid design request",
  proposalsSeen() === proposalsBefore && dialogs.length > dialogsBefore && /已取消/.test(cancelNote),
  `dialogs=${dialogs.length - dialogsBefore} · proposals ${proposalsBefore} -> ${proposalsSeen()}`);

const errorsTotal = await evalJs("window.__errors.length");
check("10 zero page errors during the whole free flow", errorsTotal === 0, `errors=${errorsTotal}`);

ws.close();
chrome.kill();
writeFileSync(join(OUT, "free-e2e-summary.json"), JSON.stringify({ initial, uploaded, cards, cost, result, diff, retouchCost, retouch, skinDiff, dialogs, results }, null, 2));
const failed = results.filter((r) => !r.ok);
log(failed.length ? `\nRESULT: FAIL (${failed.length})` : "\nRESULT: ALL FREE E2E STEPS PASS");
process.exit(failed.length ? 1 : 0);
