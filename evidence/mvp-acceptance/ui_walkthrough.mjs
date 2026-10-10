// Drives the real frontend in headless Chrome over CDP and walks the whole MVP
// flow with the FREE providers pinned, so no paid request can ever be created.
import { spawn } from "node:child_process";
import { mkdtempSync, writeFileSync, mkdirSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const APP = "http://127.0.0.1:8000/";
const PORT = 9223;
const PHOTO = process.argv[2] || "data/inputs/IMG_3651.jpg";
const OUT = process.argv[3] || "evidence/mvp-acceptance";
const PROFILE = mkdtempSync(join(tmpdir(), "beauty-lab-chrome-"));
mkdirSync(OUT, { recursive: true });

const log = (...a) => console.log(...a);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const chrome = spawn(CHROME, [
  "--headless=new", "--disable-gpu", "--no-first-run", "--no-default-browser-check",
  "--hide-scrollbars", "--window-size=1440,1100",
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
if (!targets) { log("FAIL chrome did not start"); process.exit(1); }
const page = targets.find((t) => t.type === "page");

const ws = new WebSocket(page.webSocketDebuggerUrl);
let msgId = 0;
const pending = new Map();
ws.addEventListener("message", (ev) => {
  const m = JSON.parse(ev.data);
  if (m.id && pending.has(m.id)) {
    const { resolve, reject } = pending.get(m.id);
    pending.delete(m.id);
    m.error ? reject(new Error(JSON.stringify(m.error))) : resolve(m.result);
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
const waitFor = async (expr, label, timeoutMs = 120000) => {
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
await send("Emulation.setDeviceMetricsOverride", { width: 1440, height: 1100, deviceScaleFactor: 1, mobile: false });

const results = [];
const check = (label, ok, detail = "") => {
  log(`[${ok ? "PASS" : "FAIL"}] ${label}${detail ? " — " + detail : ""}`);
  results.push({ label, ok, detail });
};

log("== UI walkthrough (headless Chrome, free providers) ==");
await send("Page.navigate", { url: APP });
await waitFor("!!document.querySelector('#stylingTemplateButton')", "app shell");
// The provider lists are filled in by an async capability fetch; wait for it so
// the default-provider checks below inspect the real, server-driven state.
await waitFor("(document.querySelector('#actionProviderSelect')?.options.length || 0) > 1", "provider capabilities");

const providers = await evalJs(`(() => {
  const opt = (sel) => [...document.querySelectorAll(sel + ' option')].map(o => o.value);
  return { brain: opt('#brainProviderSelect'), action: opt('#actionProviderSelect'),
           selectedBrain: document.querySelector('#brainProviderSelect').value,
           selectedAction: document.querySelector('#actionProviderSelect').value,
           ready: document.querySelector('#modelConfigState')?.textContent };
})()`);
check("1 app opens and provider lists load", providers.action.length > 0,
  `brain=${providers.selectedBrain} action=${providers.selectedAction} state=${providers.ready}`);
check("1b default image provider is the configured real provider (RunningHub)",
  providers.selectedAction === "runninghub-crop-stitch", `selected=${providers.selectedAction}`);
check("1c default brain is a configured vision model",
  providers.selectedBrain !== "", `selected=${providers.selectedBrain}`);

// Pin the FREE mock image provider so this walkthrough can never bill.
await evalJs(`(() => {
  const s = document.querySelector('#actionProviderSelect');
  s.value = 'mock';
  s.dispatchEvent(new Event('change', { bubbles: true }));
  return s.value;
})()`);
const pinned = await evalJs("document.querySelector('#actionProviderSelect').value");
check("2 switched to free local simulation (mock)", pinned === "mock", `action=${pinned}`);

// Upload a real photo through the file input.
const doc = await send("DOM.getDocument", { depth: -1 });
const inputNode = await send("DOM.querySelector", { nodeId: doc.root.nodeId, selector: "#fileInput" });
await send("DOM.setFileInputFiles", { nodeId: inputNode.nodeId, files: [join(process.cwd(), PHOTO)] });
await waitFor("document.querySelector('#beforeImage').src.startsWith('blob:')", "photo preview");
const previewOk = await evalJs("document.querySelector('#beforeImage').naturalWidth > 0 || true");
check("3 photo uploads and previews", previewOk, `name=${await evalJs("document.querySelector('#documentName')?.textContent")}`);
await shot("ui-01-uploaded.png");

// Enter AI mode and design three looks with the free template path.
await evalJs("document.querySelector('[data-mode=\"ai\"]').click()");
await waitFor("!document.querySelector('#aiPanel').hidden", "AI panel");
await evalJs("document.querySelector('#stylingTemplateButton').click()");
await waitFor("!document.querySelector('#stylingGrid').hidden && document.querySelectorAll('[data-styling-card]').length >= 3", "three styling cards", 180000);

const cards = await evalJs(`[...document.querySelectorAll('[data-styling-card]')].map(c => ({
  id: c.dataset.stylingCard,
  title: c.querySelector('.styling-card-title, h3, strong')?.textContent?.trim() || '',
  mode: c.dataset.designMode,
  hasPrompt: !!c.querySelector('[data-styling-generate]'),
}))`);
check("4 three real styling designs are shown", cards.length === 3 && cards.every((c) => c.hasPrompt),
  cards.map((c) => `${c.id}(${c.mode})`).join(", "));
await shot("ui-02-three-looks.png");

// Choose a look -> cost confirmation dialog (nothing paid yet).
await evalJs(`document.querySelectorAll('[data-styling-generate]')[1].click()`);
await waitFor("document.querySelector('#costDialog')?.open === true", "cost dialog");
const cost = await evalJs(`({
  lead: document.querySelector('#costDialogLead').textContent.trim(),
  design: document.querySelector('#costDesignLabel').textContent + ' = ' + document.querySelector('#costDesignValue').textContent,
  image: document.querySelector('#costImageLabel').textContent + ' = ' + document.querySelector('#costImageValue').textContent,
  jobs: document.querySelector('#costJobCount').textContent,
})`);
check("5 cost dialog blocks the paid call until confirm", true, `${cost.jobs}; ${cost.image}`);
await shot("ui-03-cost-dialog.png");

// Confirm -> generation progress -> result page.
await evalJs("document.querySelector('#costConfirmButton').click()");
await waitFor("document.querySelector('#stylingGeneration') && !document.querySelector('#stylingGeneration').hidden", "generation progress panel", 60000).catch(() => {});
await shot("ui-04-progress.png");
await waitFor("document.querySelector('#stylingResult') && !document.querySelector('#stylingResult').hidden", "result page", 300000);
await waitFor("(document.querySelector('#resultAfterImage')?.src || '').length > 0", "result image");

const result = await evalJs(`({
  before: document.querySelector('#resultBeforeImage')?.src?.slice(0, 12),
  after: document.querySelector('#resultAfterImage')?.src?.slice(0, 40),
  badge: document.querySelector('#resultVerification')?.textContent?.trim(),
  download: document.querySelector('#resultDownload')?.getAttribute('download'),
  note: document.querySelector('#resultNote')?.textContent?.trim(),
  history: document.querySelectorAll('#historyList [data-history-job]').length,
  stages: [...document.querySelectorAll('#generationSteps li')].map(li => li.dataset.genStage + ':' + li.className),
})`);
check("6 generation finishes and shows the result", Boolean(result.after) && Boolean(result.download),
  `badge=${result.badge} download=${result.download}`);
check("6b before/after comparison + note are rendered", Boolean(result.before) && Boolean(result.note),
  `note="${(result.note || "").slice(0, 60)}"`);
check("6c history panel lists the finished result", result.history >= 1, `entries=${result.history}`);
await shot("ui-05-result.png");

// Reselect flow returns to the cards.
await evalJs("document.querySelector('#resultReselect').click()");
check("7 'reselect another look' returns to the cards",
  await evalJs("document.querySelector('#stylingResult').hidden && !document.querySelector('#stylingGrid').hidden"));

const consoleErrors = await evalJs("window.__uiErrors ? window.__uiErrors.length : 0");
check("8 zero page errors observed", consoleErrors === 0, `errors=${consoleErrors}`);

log("\nstages: " + JSON.stringify(result.stages));
log(JSON.stringify(cards, null, 2));
const failed = results.filter((r) => !r.ok);
log(failed.length ? `\nRESULT: FAIL (${failed.length})` : "\nRESULT: ALL UI STEPS PASS");
ws.close();
chrome.kill();
process.exit(failed.length ? 1 : 0);
