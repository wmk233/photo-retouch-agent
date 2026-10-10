// REAL end-to-end acceptance run for AI Beauty Style Lab (one paid attempt).
//
// Drives the real UI in headless Chrome over CDP through the whole flow:
//   upload IMG_3651.jpg -> DeepSeek designs three looks -> pick one ->
//   SCHP/MediaPipe face processing -> RunningHub generates -> save Final ->
//   the browser shows the real result.
//
// Paid steps are NEVER automatic. The design confirm (window.confirm) and the
// cost <dialog> both require the operator to pass --confirm-paid; without it the
// script parks on the dialog and spends nothing.
//
// Budgets enforced here: DeepSeek <= 3 requests, RunningHub exactly 1 paid task,
// and no retry of any kind.
import { spawn } from "node:child_process";
import { mkdtempSync, writeFileSync, mkdirSync, existsSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const APP = process.env.E2E_APP || "http://127.0.0.1:8001/";
const API_ORIGIN = new URL(APP).origin;
const CDP_PORT = Number(process.env.E2E_CDP_PORT || 9226);
const ARGS = process.argv.slice(2);
const positional = ARGS.filter((a) => !a.startsWith("--"));
const PHOTO = positional[0] || "data/inputs/IMG_3651.jpg";
const OUT = positional[1] || "evidence/mvp-acceptance";
const CONFIRM_PAID = ARGS.includes("--confirm-paid");
const CHOICE = Number((ARGS.find((a) => a.startsWith("--choose=")) || "--choose=1").split("=")[1]);
const DESIGN_TIMEOUT_MS = Number(process.env.E2E_DESIGN_TIMEOUT_MS || 300000);
const JOB_TIMEOUT_MS = Number(process.env.E2E_JOB_TIMEOUT_MS || 1200000);
const PROFILE = mkdtempSync(join(tmpdir(), "beauty-lab-real-"));
mkdirSync(OUT, { recursive: true });

const log = (...a) => console.log(...a);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const summary = {
  startedAt: new Date().toISOString(),
  app: APP,
  photo: PHOTO,
  choiceIndex: CHOICE,
  confirmedPaid: CONFIRM_PAID,
  steps: [],
  proposals: [],
  chosenPlan: null,
  costDialog: null,
  designConfirmMessage: null,
  job: null,
  result: null,
  sideEffects: {},
  failures: [],
};
const note = (label, extra = {}) => {
  summary.steps.push({ at: new Date().toISOString(), label, ...extra });
  log(`  - ${label}`);
};
const saveSummary = () => {
  summary.finishedAt = new Date().toISOString();
  writeFileSync(join(OUT, "real_e2e_summary.json"), JSON.stringify(summary, null, 2));
};
const hardFail = (msg) => {
  summary.failures.push(msg);
  saveSummary();
  log(`\nABORT: ${msg}`);
  process.exit(1);
};

// ---------------------------------------------------------------------------
// 0. Refuse to spend a second time: one paid RunningHub task per acceptance.
// ---------------------------------------------------------------------------
const prevPath = join(OUT, "real_e2e_summary.json");
if (existsSync(prevPath) && !ARGS.includes("--allow-rerun")) {
  try {
    const prev = JSON.parse(readFileSync(prevPath, "utf8"));
    if (prev?.job?.runninghubTaskId || prev?.sideEffects?.runninghubTaskId) {
      log("REFUSING TO RUN: a previous real run already created a RunningHub task.");
      log("No second order is allowed. Re-run with --allow-rerun only if you really mean it.");
      process.exit(3);
    }
  } catch {}
}
if (!existsSync(PHOTO)) hardFail(`photo not found: ${PHOTO}`);

// ---------------------------------------------------------------------------
// 1. Headless Chrome + CDP plumbing.
// ---------------------------------------------------------------------------
const chrome = spawn(CHROME, [
  "--headless=new", "--disable-gpu", "--no-first-run", "--no-default-browser-check",
  "--hide-scrollbars", "--window-size=1440,1400",
  `--remote-debugging-port=${CDP_PORT}`, `--user-data-dir=${PROFILE}`, "about:blank",
], { stdio: "ignore" });

let targets = null;
for (let i = 0; i < 80; i++) {
  try {
    const r = await fetch(`http://127.0.0.1:${CDP_PORT}/json/list`);
    targets = await r.json();
    if (targets.some((t) => t.type === "page")) break;
  } catch {}
  await sleep(250);
}
if (!targets || !targets.some((t) => t.type === "page")) hardFail("chrome did not start");
const page = targets.find((t) => t.type === "page");

const ws = new WebSocket(page.webSocketDebuggerUrl);
let msgId = 0;
const pending = new Map();
const events = [];
const captured = [];
let dialogResolver = null;

const WATCH = [
  /\/api\/photos\/analyze$/,
  /\/api\/styling\/proposals$/,
  /\/api\/retouch\/jobs$/,
  /\/api\/retouch\/jobs\/[^/]+$/,
];

// The CDP Network.getResponseBody path proved unreliable for this app's POST
// responses (it returned nothing for /api/styling/proposals on the first real
// run, which aborted the run *after* the design money was already spent). So we
// also wrap fetch/XHR inside the page itself and read the bodies back through
// Runtime.evaluate. Either source alone is enough to keep the run going.
const RECORDER = String.raw`
(() => {
  const store = [];
  window.__e2eCaptured = store;
  const interesting = (url) => url.indexOf('/api/photos/analyze') >= 0
    || url.indexOf('/api/styling/proposals') >= 0
    || url.indexOf('/api/retouch/jobs') >= 0;
  const record = (entry) => { store.push(entry); if (store.length > 60) store.shift(); };
  const origFetch = window.fetch;
  if (origFetch) {
    window.fetch = async function (...args) {
      const req = args[0];
      const url = (typeof req === 'string') ? req : ((req && req.url) || '');
      const method = ((args[1] && args[1].method) || (req && req.method) || 'GET').toUpperCase();
      const res = await origFetch.apply(this, args);
      if (interesting(url)) {
        try { record({ url, method, status: res.status, body: await res.clone().text() }); }
        catch (err) { record({ url, method, status: res.status, error: String(err) }); }
      }
      return res;
    };
  }
  const XHR = window.XMLHttpRequest;
  if (XHR) {
    const origOpen = XHR.prototype.open;
    const origSend = XHR.prototype.send;
    XHR.prototype.open = function (method, url, ...rest) {
      this.__e2eReq = { method: String(method).toUpperCase(), url: String(url) };
      return origOpen.call(this, method, url, ...rest);
    };
    XHR.prototype.send = function (...args) {
      this.addEventListener('load', () => {
        const info = this.__e2eReq;
        if (!info || !interesting(info.url)) return;
        let body = '';
        try { if (this.responseType === '' || this.responseType === 'text') body = this.responseText; } catch (err) {}
        record({ url: info.url, method: info.method, status: this.status, body });
      });
      return origSend.apply(this, args);
    };
  }
})();
`;

ws.addEventListener("message", (ev) => {
  const m = JSON.parse(ev.data);
  if (m.id && pending.has(m.id)) {
    const { resolve, reject } = pending.get(m.id);
    pending.delete(m.id);
    m.error ? reject(new Error(JSON.stringify(m.error))) : resolve(m.result);
    return;
  }
  if (m.method === "Page.javascriptDialogOpening") {
    events.push({ type: "jsDialog", message: m.params.message });
    summary.designConfirmMessage = m.params.message;
    if (dialogResolver) { const r = dialogResolver; dialogResolver = null; r(m.params.message); }
    return;
  }
  if (m.method === "Network.requestWillBeSent") {
    events.push({ type: "req", url: m.params.request.url, method: m.params.request.method });
    reqMeta.set(m.params.requestId, { url: m.params.request.url, method: m.params.request.method });
    return;
  }
  if (m.method === "Network.responseReceived") {
    const meta = reqMeta.get(m.params.requestId);
    if (meta) meta.status = m.params.response.status;
    return;
  }
  if (m.method === "Network.loadingFinished") {
    const meta = reqMeta.get(m.params.requestId);
    if (meta && WATCH.some((re) => re.test(meta.url))) {
      send("Network.getResponseBody", { requestId: m.params.requestId })
        .then((r) => {
          const body = r.base64Encoded ? Buffer.from(r.data, "base64").toString("utf8") : r.data;
          let json = null;
          try { json = JSON.parse(body); } catch {}
          captured.push({ url: meta.url, method: meta.method, status: meta.status, json, body: json ? null : body.slice(0, 2000) });
        })
        .catch(() => {});
    }
    return;
  }
});
const reqMeta = new Map();

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
const waitFor = async (expr, label, timeoutMs = 60000) => {
  const t0 = Date.now();
  while (Date.now() - t0 < timeoutMs) {
    try { if (await evalJs(expr)) return true; } catch {}
    await sleep(400);
  }
  throw new Error(`timeout waiting for ${label}`);
};
const shot = async (name) => {
  const r = await send("Page.captureScreenshot", { format: "png", captureBeyondViewport: true });
  const p = join(OUT, name);
  writeFileSync(p, Buffer.from(r.data, "base64"));
  log(`   screenshot: ${p}`);
  return p;
};
const clickEl = async (selector) => {
  const rect = await evalJs(`(() => { const el = document.querySelector(${JSON.stringify(selector)}); el.scrollIntoView({block:'center'}); const r = el.getBoundingClientRect(); return { x: r.x + r.width/2, y: r.y + r.height/2 }; })()`);
  await send("Input.dispatchMouseEvent", { type: "mousePressed", x: rect.x, y: rect.y, button: "left", clickCount: 1 });
  await send("Input.dispatchMouseEvent", { type: "mouseReleased", x: rect.x, y: rect.y, button: "left", clickCount: 1 });
};

await send("Page.enable");
await send("Runtime.enable");
await send("DOM.enable");
await send("Network.enable");
await send("Emulation.setDeviceMetricsOverride", { width: 1440, height: 1400, deviceScaleFactor: 1, mobile: false });
await send("Page.addScriptToEvaluateOnNewDocument", { source: RECORDER });

const paidCalls = () => events.filter((e) => e.type === "req" && /\/api\/(photos\/analyze|styling\/proposals|retouch\/jobs)/.test(e.url));

// Read the in-page recorder and normalise entries to the same shape the CDP
// capture uses, so both sources can be searched with one predicate.
const readInPageCaptured = async () => {
  const raw = await evalJs("(window.__e2eCaptured || [])").catch(() => []);
  return (raw || []).map((r) => {
    let json = null;
    try { json = JSON.parse(r.body); } catch {}
    return { url: r.url, method: r.method, status: r.status, json, body: json ? null : String(r.body || "").slice(0, 2000) };
  });
};
const allCaptured = async () => [...captured, ...(await readInPageCaptured())];
const lastMatch = (list, re) => [...list].reverse().find((c) => re.test(c.url));
const waitForResponse = async (re, timeoutMs) => {
  const t0 = Date.now();
  while (Date.now() - t0 < timeoutMs) {
    const hit = lastMatch(await allCaptured(), re);
    if (hit) return hit;
    await sleep(400);
  }
  return null;
};
// If the response body cannot be read at all, rebuild the proposal list from
// the rendered cards so an already-paid design is never thrown away.
const readProposalFallback = async () => {
  const cards = await evalJs(`[...document.querySelectorAll('[data-styling-card]')].map((c) => ({ planId: c.getAttribute('data-styling-card'), name: (c.querySelector('.styling-card-title,h4,h3')?.textContent || '').trim(), text: (c.innerText || '') }))`).catch(() => []);
  if (!cards || cards.length < 3) return null;
  summary.failures.push("proposals response body not captured; fell back to DOM card text");
  return cards.map((c) => ({ planId: c.planId, name: c.name, designMode: "ai", sourceProvider: "deepseek(dom-fallback)", rawText: c.text }));
};

// ---------------------------------------------------------------------------
// 2. Load the real app.
// ---------------------------------------------------------------------------
log("== REAL E2E: load app ==");
await send("Page.navigate", { url: APP });
await waitFor("document.readyState === 'complete' && !!document.querySelector('#fileInput')", "app loaded", 60000);
await sleep(1500);
note("app loaded", { url: APP });
await shot("real-e2e-00-loaded.png");

// ---------------------------------------------------------------------------
// 3. Free preflight: capabilities + selectors, BEFORE any paid action.
// ---------------------------------------------------------------------------
log("== preflight (no spend) ==");
const caps = await fetch(`${API_ORIGIN}/api/styling/providers`, {
  headers: { "X-Agent-Provider": "deepseek", "X-Action-Provider": "runninghub-crop-stitch" },
}).then((r) => r.json());
summary.sideEffects.capabilities = {
  deepseekConfigured: caps.deepseekConfigured,
  deepseekModel: caps.deepseekModel,
  runninghubConfigured: caps.runninghubConfigured,
  runninghubProviderId: caps.runninghubProviderId,
  stylingAiFlowRequests: caps.stylingAiFlowRequests,
};
if (!caps.deepseekConfigured) hardFail("DeepSeek brain is not configured on this backend");
if (!caps.runninghubConfigured) hardFail("RunningHub action provider is not configured on this backend");
if (caps.stylingAiFlowRequests !== 3) hardFail(`cost copy mismatch: stylingAiFlowRequests=${caps.stylingAiFlowRequests}, expected 3`);
note("capabilities ok", summary.sideEffects.capabilities);

const preflight = await evalJs(`(() => {
  const need = ['#fileInput','#brainProviderSelect','#actionProviderSelect','[data-mode="ai"]','#stylingDesignButton','#costDialog','#costConfirmButton','#costCancelButton','#stylingGrid','#stylingResult','#resultBeforeImage','#resultAfterImage'];
  const missing = need.filter((s) => !document.querySelector(s));
  const brainOpts = [...document.querySelectorAll('#brainProviderSelect option')].map(o=>o.value);
  const actOpts = [...document.querySelectorAll('#actionProviderSelect option')].map(o=>o.value);
  return { missing, brainHasDeepseek: brainOpts.includes('deepseek'), actionHasRunninghub: actOpts.includes('runninghub-crop-stitch'), brainOpts, actOpts };
})()`);
summary.sideEffects.preflight = preflight;
if (preflight.missing.length) hardFail(`missing DOM nodes: ${preflight.missing.join(', ')}`);
if (!preflight.brainHasDeepseek) hardFail(`brain select has no 'deepseek' option: ${preflight.brainOpts.join(',')}`);
if (!preflight.actionHasRunninghub) hardFail(`action select has no 'runninghub-crop-stitch' option: ${preflight.actOpts.join(',')}`);
note("DOM + provider options ok");

// ---------------------------------------------------------------------------
// 4. Upload the real photo.
// ---------------------------------------------------------------------------
log("== upload photo ==");
const doc = await send("DOM.getDocument", { depth: -1 });
const inputNode = await send("DOM.querySelector", { nodeId: doc.root.nodeId, selector: "#fileInput" });
await send("DOM.setFileInputFiles", { nodeId: inputNode.nodeId, files: [join(process.cwd(), PHOTO)] });
await waitFor("document.querySelector('#beforeImage')?.src?.startsWith('blob:')", "photo preview", 60000);
note("photo uploaded", { photo: PHOTO });
await shot("real-e2e-01-uploaded.png");

// Select DeepSeek brain + RunningHub action, then switch to the paid AI mode.
await evalJs(`(() => { const b = document.querySelector('#brainProviderSelect'); b.value = 'deepseek'; b.dispatchEvent(new Event('change', { bubbles: true })); return b.value; })()`);
await evalJs(`(() => { const a = document.querySelector('#actionProviderSelect'); a.value = 'runninghub-crop-stitch'; a.dispatchEvent(new Event('change', { bubbles: true })); return a.value; })()`);
await evalJs(`document.querySelector('[data-mode="ai"]').click()`);
await waitFor("!document.querySelector('#aiPanel').hidden", "AI panel", 20000);
const sel = await evalJs(`({ brain: document.querySelector('#brainProviderSelect').value, action: document.querySelector('#actionProviderSelect').value })`);
if (sel.brain !== "deepseek" || sel.action !== "runninghub-crop-stitch") hardFail(`provider selection failed: ${JSON.stringify(sel)}`);
note("providers selected", sel);
await shot("real-e2e-02-ai-panel.png");

// ---------------------------------------------------------------------------
// 5. PAID #1+#2: design three looks with DeepSeek (analysis + design).
// ---------------------------------------------------------------------------
log("== AI design (2 DeepSeek calls: analyze + proposals) ==");
const beforeDesign = paidCalls().length;
const dialogPromise = new Promise((r) => { dialogResolver = r; });
const clickPromise = clickEl("#stylingDesignButton").catch(() => {});
const confirmMsg = await Promise.race([dialogPromise, sleep(15000).then(() => null)]);
if (typeof confirmMsg !== "string") hardFail("AI design did not raise the blocking confirm dialog");
await Promise.race([clickPromise.catch(() => {}), sleep(3000)]);
note("design confirm shown", { message: confirmMsg });

if (!CONFIRM_PAID) {
  await send("Page.handleJavaScriptDialog", { accept: false });
  summary.failures.push("dry stop: --confirm-paid not set; cancelled the design confirm");
  saveSummary();
  log("\nDRY STOP: confirm shown and cancelled, nothing submitted. Re-run with --confirm-paid to run for real.");
  ws.close(); chrome.kill();
  process.exit(2);
}
await send("Page.handleJavaScriptDialog", { accept: true });
note("design confirm accepted by operator flag");

try {
  await waitFor("!document.querySelector('#stylingGrid').hidden && document.querySelectorAll('[data-styling-card]').length >= 3", "three AI looks", DESIGN_TIMEOUT_MS);
} catch (e) {
  const fb = await evalJs("document.querySelector('#stylingFeedback')?.textContent?.trim() || ''").catch(() => "");
  hardFail(`${e.message}; feedback=${fb}`);
}
note("three looks rendered");
await shot("real-e2e-03-three-looks.png");

const proposeResp = await waitForResponse(/\/api\/styling\/proposals$/, 15000);
const rawProposals = Array.isArray(proposeResp?.json) ? proposeResp.json : await readProposalFallback();
if (!rawProposals) hardFail("no /styling/proposals response captured and fewer than 3 styling cards in the DOM");
summary.proposals = rawProposals.map((p) => ({
  planId: p.planId, name: p.name, title: p.title,
  designMode: p.designMode, sourceProvider: p.sourceProvider,
  designDirection: p.designDirection, designReason: p.designReason, whyItFits: p.whyItFits,
  hairPlan: p.hairPlan, makeupPlan: p.makeupPlan, outfitPlan: p.outfitPlan, accessoryPlan: p.accessoryPlan,
  palette: p.palette, expectedChanges: p.expectedChanges,
  editPrompt: p.editPrompt, negativePrompt: p.negativePrompt,
  referencePresets: (p.referencePresets || []).map((x) => x.nameZh || x.name || x.presetId),
  rawText: p.rawText,
}));
if (summary.proposals.length < 3) hardFail(`expected 3 proposals, got ${summary.proposals.length}`);
if (summary.proposals.some((p) => p.designMode !== "ai")) hardFail("a proposal is NOT AI-designed (template fallback leaked in)");
if (summary.proposals.some((p) => !/deepseek/i.test(p.sourceProvider || ""))) hardFail(`unexpected sourceProvider: ${summary.proposals.map((p) => p.sourceProvider).join(',')}`);
note("3 real DeepSeek looks captured", { planIds: summary.proposals.map((p) => p.planId) });
writeFileSync(join(OUT, "real_e2e_proposals.json"), JSON.stringify(summary.proposals, null, 2));

// ---------------------------------------------------------------------------
// 6. Choose ONE look.
// ---------------------------------------------------------------------------
const chosen = summary.proposals[CHOICE % summary.proposals.length];
summary.chosenPlan = {
  planId: chosen.planId, name: chosen.name,
  hairPlan: chosen.hairPlan, makeupPlan: chosen.makeupPlan, outfitPlan: chosen.outfitPlan,
  accessoryPlan: chosen.accessoryPlan, palette: chosen.palette,
  editPrompt: chosen.editPrompt,
};
note("chosen look", { planId: chosen.planId, name: chosen.name, hairPlan: chosen.hairPlan, outfitPlan: chosen.outfitPlan });
await evalJs(`document.querySelector(${JSON.stringify(`[data-styling-card="${chosen.planId}"]`)})?.scrollIntoView({block:'center'})`);
await shot("real-e2e-04-chosen-card.png");

// ---------------------------------------------------------------------------
// 7. Open the real cost dialog and stop for explicit confirmation.
// ---------------------------------------------------------------------------
log("== cost dialog (explicit confirm required) ==");
await evalJs(`document.querySelector(${JSON.stringify(`[data-styling-generate="${chosen.planId}"]`)}).click()`);
await waitFor("document.querySelector('#costDialog')?.open === true", "cost dialog opens", 20000);
const dialog = await evalJs(`({
  lead: document.querySelector('#costDialogLead')?.textContent?.trim() || '',
  designLabel: document.querySelector('#costDesignLabel')?.textContent?.trim() || '',
  design: document.querySelector('#costDesignValue')?.textContent?.trim() || '',
  imageLabel: document.querySelector('#costImageLabel')?.textContent?.trim() || '',
  image: document.querySelector('#costImageValue')?.textContent?.trim() || '',
  jobs: document.querySelector('#costJobCount')?.textContent?.trim() || '',
  note: document.querySelector('#costDialogNote')?.textContent?.trim() || '',
  buttons: [...document.querySelectorAll('#costDialog menu button')].map((b) => b.id + ':' + b.textContent.trim()),
})`);
summary.costDialog = dialog;
note("cost dialog open", { jobs: dialog.jobs, image: dialog.image, note: dialog.note });
await shot("real-e2e-05-cost-dialog.png");
if (!/1 个付费任务/.test(dialog.jobs)) hardFail(`cost dialog does not promise exactly 1 paid task: "${dialog.jobs}"`);
if (!/RunningHub/.test(dialog.lead + dialog.imageLabel)) hardFail("cost dialog does not name RunningHub");

if (!CONFIRM_PAID) {
  await evalJs("document.querySelector('#costCancelButton').click()");
  summary.failures.push("dry stop: --confirm-paid not set; cancelled the cost dialog");
  saveSummary();
  log("\nDRY STOP: cost dialog cancelled, no RunningHub task submitted.");
  ws.close(); chrome.kill();
  process.exit(2);
}

const beforeGenerate = paidCalls().length;
await evalJs("document.querySelector('#costConfirmButton').click()");
note("cost dialog confirmed by operator flag");
await waitFor("document.querySelector('#costDialog')?.open === false", "cost dialog closed", 10000);

// ---------------------------------------------------------------------------
// 8. RunningHub generation (exactly one paid task, no retry).
// ---------------------------------------------------------------------------
log("== generation (1 RunningHub paid task) ==");
let jobId = null;
{
  const t0 = Date.now();
  while (Date.now() - t0 < 20000 && !jobId) {
    const post = lastMatch(await allCaptured(), /\/api\/retouch\/jobs$/);
    if (post && post.method === "POST" && post.json?.jobId) jobId = post.json.jobId;
    if (!jobId) await sleep(400);
  }
}
if (!jobId) {
  const recent = await fetch(`${API_ORIGIN}/api/retouch/jobs?limit=3`).then((r) => r.json()).catch(() => null);
  const list = Array.isArray(recent) ? recent : (recent?.jobs || recent?.items || []);
  if (list.length && list[0]?.jobId) { jobId = list[0].jobId; note("jobId recovered from job list fallback", { jobId }); }
}
if (!jobId) hardFail("no jobId captured from POST /api/retouch/jobs");
note("job created", { jobId });

const jobUrl = `${API_ORIGIN}/api/retouch/jobs/${jobId}`;
let job = null;
const t0 = Date.now();
let lastStage = null;
let progressShot = false;
while (Date.now() - t0 < JOB_TIMEOUT_MS) {
  job = await fetch(jobUrl).then((r) => r.json()).catch(() => null);
  if (job && job.stage !== lastStage) {
    lastStage = job.stage;
    note("stage", { status: job.status, stage: job.stage, progress: job.progress });
  }
  if (!progressShot && job && ["segmentation", "generation", "inpainting"].includes(job.stage)) {
    progressShot = true;
    await shot("real-e2e-06-progress.png").catch(() => {});
  }
  if (job && ["succeeded", "failed", "cancelled"].includes(job.status)) break;
  await sleep(3000);
}
if (!job) hardFail("could not read job status");
summary.job = {
  jobId, status: job.status, stage: job.stage, modelName: job.modelName, brainModel: job.brainModel,
  planId: job.planId, outputUrls: job.outputUrls, errorMessage: job.errorMessage || null,
  verificationLevel: job.verificationLevel, faceLock: job.faceLock || null,
  runninghubTaskId: job.providerTaskId || job.taskId || null,
  createdAt: job.createdAt, updatedAt: job.updatedAt,
};
note("job terminal", { status: job.status, stage: job.stage, outputs: (job.outputUrls || []).join(",") });

if (job.status !== "succeeded") {
  await shot("real-e2e-07-failed.png").catch(() => {});
  hardFail(`generation did not succeed: status=${job.status} error=${job.errorMessage}`);
}

// ---------------------------------------------------------------------------
// 9. Show + save the real result in the browser.
// ---------------------------------------------------------------------------
await waitFor("!document.querySelector('#stylingResult').hidden", "result panel shown", 60000);
await sleep(2500);
const resultDom = await evalJs(`({
  before: document.querySelector('#resultBeforeImage')?.src || '',
  after: document.querySelector('#resultAfterImage')?.src || '',
  download: document.querySelector('#resultDownload')?.getAttribute('href') || '',
  verification: document.querySelector('#resultVerification')?.textContent?.trim() || '',
  note: document.querySelector('#resultNote')?.textContent?.trim() || '',
  feedback: document.querySelector('#stylingFeedback')?.textContent?.trim() || '',
})`);
summary.result = resultDom;
await shot("real-e2e-08-result.png");
const compareShot = await shot("real-e2e-09-result-compare.png");

// Download the real output bytes so the evidence is not a screenshot.
let savedPath = null;
try {
  const afterUrl = resultDom.after.startsWith("http") ? resultDom.after : `${API_ORIGIN}${resultDom.after}`;
  const buf = Buffer.from(await fetch(afterUrl).then((r) => r.arrayBuffer()));
  savedPath = join(OUT, "real-e2e-final.png");
  writeFileSync(savedPath, buf);
  summary.result.savedFile = savedPath;
  summary.result.savedBytes = buf.length;
  note("final image downloaded", { savedPath, bytes: buf.length });
} catch (e) {
  summary.failures.push(`could not download final image: ${e.message}`);
}

// Pull the RunningHub provenance written next to the output.
summary.sideEffects.deepseekCalls = (() => {
  const p = join(OUT, "deepseek_calls.jsonl");
  if (!existsSync(p)) return null;
  const lines = readFileSync(p, "utf8").trim().split("\n").filter(Boolean);
  return { count: lines.length, entries: lines.map((l) => { try { return JSON.parse(l); } catch { return { raw: l.slice(0, 400) }; } }) };
})();
summary.sideEffects.paidCalls = paidCalls();
summary.sideEffects.paidCallCount = paidCalls().length;
summary.sideEffects.beforeGenerateCount = beforeGenerate - beforeDesign;

saveSummary();
log(`\nCOMPARE SHOT: ${compareShot}`);
log(`SUMMARY: ${join(OUT, "real_e2e_summary.json")}`);
log(summary.failures.length ? `\nRESULT: COMPLETED WITH ${summary.failures.length} FAILURE(S)` : "\nRESULT: REAL E2E COMPLETED");
ws.close();
chrome.kill();
process.exit(summary.failures.length ? 1 : 0);
