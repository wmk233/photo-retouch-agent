// FREE interface check for the "3-call / no-vision" copy fixes.
//
// It drives the real UI in headless Chrome over CDP and verifies:
//   1. the always-visible cost note states the FULL flow is 3 paid model calls
//      (analysis + design + generation prompt planning), not 2;
//   2. the copy says the DeepSeek-class brain designs from locally measured
//      FaceFacts and never looks at the uploaded photo;
//   3. the pre-design guard is a real blocking confirm (never auto-confirmed)
//      whose cancel path submits NOTHING;
//   4. the paid cost <dialog> is preserved and requires an explicit confirm.
//
// Every step runs with no billable request: the confirm is dismissed and the
// only generation exercised uses the free local simulation.
import { spawn } from "node:child_process";
import { mkdtempSync, writeFileSync, mkdirSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const APP = "http://127.0.0.1:8000/";
const PORT = 9225;
const PHOTO = process.argv[2] || "data/inputs/IMG_3651.jpg";
const OUT = process.argv[3] || "evidence/mvp-acceptance";
const PROFILE = mkdtempSync(join(tmpdir(), "beauty-lab-copy-"));
mkdirSync(OUT, { recursive: true });

const log = (...a) => console.log(...a);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const chrome = spawn(CHROME, [
  "--headless=new", "--disable-gpu", "--no-first-run", "--no-default-browser-check",
  "--hide-scrollbars", "--window-size=1440,1200",
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
const events = [];
let dialogResolver = null;
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
    if (dialogResolver) { const r = dialogResolver; dialogResolver = null; r(m.params.message); }
  }
  if (m.method === "Network.requestWillBeSent") {
    events.push({ type: "req", url: m.params.request.url, method: m.params.request.method });
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
const waitFor = async (expr, label, timeoutMs = 60000) => {
  const t0 = Date.now();
  while (Date.now() - t0 < timeoutMs) {
    try { if (await evalJs(expr)) return true; } catch {}
    await sleep(300);
  }
  throw new Error(`timeout waiting for ${label}`);
};
const shot = async (name) => {
  const r = await send("Page.captureScreenshot", { format: "png", captureBeyondViewport: true });
  const p = join(OUT, name);
  writeFileSync(p, Buffer.from(r.data, "base64"));
  log(`   screenshot: ${p}`);
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
await send("Emulation.setDeviceMetricsOverride", { width: 1440, height: 1200, deviceScaleFactor: 1, mobile: false });

const results = [];
const check = (label, ok, detail = "") => {
  log(`[${ok ? "PASS" : "FAIL"}] ${label}${detail ? " — " + detail : ""}`);
  results.push({ label, ok, detail });
};
const paidCalls = () =>
  events.filter((e) => e.type === "req" && /\/api\/(photos\/analyze|styling\/proposals|retouch\/jobs)/.test(e.url));

log("== FREE copy / guard check (headless Chrome, no billable request) ==");
await send("Page.navigate", { url: APP });
await waitFor("!!document.querySelector('#stylingTemplateButton')", "app shell");
await waitFor("(document.querySelector('#brainProviderSelect')?.options.length || 0) > 1", "provider capabilities");
await sleep(600);

// 1. Always-visible panel copy.
const intro = await evalJs("document.querySelector('#aiPanel .ai-intro p')?.textContent?.trim() || ''");
check("1 AI intro states DeepSeek never sees the original image",
  /没有图像输入能力/.test(intro) && /FaceFacts/.test(intro), intro.slice(0, 120));

const note = await evalJs("document.querySelector('#stylingCostNote')?.textContent?.trim() || ''");
check("2 cost note states the FULL flow is 3 paid calls (not 2)",
  /共 3 次付费模型调用/.test(note) && /人物事实分析/.test(note) && /出图提示词规划/.test(note), note.slice(0, 140));
check("2b cost note states the model designs from local FaceFacts without seeing the photo",
  /不会看到原图/.test(note) && /FaceFacts/.test(note) && /并未直接做视觉看图分析/.test(note));

const button = await evalJs("document.querySelector('#stylingDesignButton b')?.textContent?.trim() || ''");
check("3 AI design button no longer claims only 2 calls", /3 次/.test(button), button);
check("3b template button still advertises 0 calls",
  /0 次调用/.test(await evalJs("document.querySelector('#stylingTemplateButton b')?.textContent || ''")));

// 2. Upload + AI mode with the real DeepSeek brain selected.
const doc = await send("DOM.getDocument", { depth: -1 });
const inputNode = await send("DOM.querySelector", { nodeId: doc.root.nodeId, selector: "#fileInput" });
await send("DOM.setFileInputFiles", { nodeId: inputNode.nodeId, files: [join(process.cwd(), PHOTO)] });
await waitFor("document.querySelector('#beforeImage').src.startsWith('blob:')", "photo preview");
await evalJs(`(() => { const b = document.querySelector('#brainProviderSelect'); b.value = 'deepseek'; b.dispatchEvent(new Event('change', { bubbles: true })); return b.value; })()`);
await evalJs("document.querySelector('[data-mode=\"ai\"]').click()");
await waitFor("!document.querySelector('#aiPanel').hidden", "AI panel");
await shot("copy-01-panel.png");

const selectedBrain = await evalJs("document.querySelector('#brainProviderSelect').value");
check("4 DeepSeek brain selected for the guard check", selectedBrain === "deepseek", `brain=${selectedBrain}`);

// 3. The guard must be a real blocking confirm that never auto-confirms.
const before = paidCalls().length;
const dialogPromise = new Promise((r) => { dialogResolver = r; });
const clickPromise = clickEl("#stylingDesignButton").catch(() => {});
const message = await Promise.race([dialogPromise, sleep(8000).then(() => null)]);
check("5 clicking AI design blocks on a real confirm (never auto-confirmed)", typeof message === "string");
if (message) {
  check("5a confirm copy states the complete flow is at most 3 paid requests",
    /最多 3 次付费请求/.test(message) && /出图提示词规划/.test(message), message.split("\n")[0]);
  check("5b confirm copy states the model never sees the photo",
    /不会看到原图/.test(message) && /FaceFacts/.test(message));
  await send("Page.handleJavaScriptDialog", { accept: false });
}
await clickPromise;
await sleep(1200);

const feedback = await evalJs("document.querySelector('#stylingFeedback')?.textContent?.trim() || ''");
check("6 cancelling the confirm submits nothing and says so",
  /已取消 AI 设计/.test(feedback) && /没有提交任何付费请求/.test(feedback), feedback);
check("6b no analyze/design/generation request was sent while cancelling",
  paidCalls().length === before, `paidCalls=${paidCalls().length}`);

// 4. The paid cost <dialog> is preserved and needs an explicit confirm.
const dialogDom = await evalJs(`(() => {
  const d = document.querySelector('#costDialog');
  return {
    present: !!d,
    openInitially: !!d?.open,
    isDialog: d?.tagName === 'DIALOG',
    buttons: [...(d?.querySelectorAll('menu button') || [])].map(b => b.id + ':' + b.textContent.trim()),
    warning: d?.querySelector('.cost-dialog-warning')?.textContent?.trim() || '',
  };
})()`);
check("7 real cost dialog is present, closed and modal",
  dialogDom.present && dialogDom.isDialog && dialogDom.openInitially === false,
  `buttons=${dialogDom.buttons.join(' | ')}`);
check("7b dialog keeps an explicit cancel + confirm and the no-auto-retry warning",
  dialogDom.buttons.some((b) => b.startsWith("costCancelButton")) &&
  dialogDom.buttons.some((b) => b.startsWith("costConfirmButton")) &&
  /绝不自动重试/.test(dialogDom.warning));

// 5. Open the dialog through the free template flow (local sim) and cancel it.
await evalJs(`(() => { const s = document.querySelector('#actionProviderSelect'); s.value = 'mock'; s.dispatchEvent(new Event('change', { bubbles: true })); return s.value; })()`);
await evalJs("document.querySelector('#stylingTemplateButton').click()");
await waitFor("!document.querySelector('#stylingGrid').hidden && document.querySelectorAll('[data-styling-card]').length >= 3", "three template cards", 180000);
const beforeDialog = paidCalls().length;
await evalJs("document.querySelectorAll('[data-styling-generate]')[0].click()");
await waitFor("document.querySelector('#costDialog')?.open === true", "cost dialog opens on generate");
const dialogText = await evalJs(`({
  lead: document.querySelector('#costDialogLead').textContent.trim(),
  design: document.querySelector('#costDesignValue').textContent.trim(),
  image: document.querySelector('#costImageValue').textContent.trim(),
  jobs: document.querySelector('#costJobCount').textContent.trim(),
})`);
check("8 dialog opens only after the user clicks generate", true,
  `${dialogText.jobs} · 图像生成 ${dialogText.image}`);
await shot("copy-02-cost-dialog.png");

await evalJs("document.querySelector('#costCancelButton').click()");
await sleep(800);
check("9 cancelling the dialog closes it and submits nothing",
  (await evalJs("document.querySelector('#costDialog')?.open === false")) && paidCalls().length === beforeDialog,
  `paidCalls=${paidCalls().length}`);

const summary = { checkedAt: new Date().toISOString(), results, panelNote: note, confirmMessage: message, dialog: dialogText, paidCalls: paidCalls() };
writeFileSync(join(OUT, "free_copy_check.json"), JSON.stringify(summary, null, 2));
const failed = results.filter((r) => !r.ok);
log(failed.length ? `\nRESULT: FAIL (${failed.length})` : "\nRESULT: ALL FREE CHECKS PASS");
ws.close();
chrome.kill();
process.exit(failed.length ? 1 : 0);
