import { writeFileSync } from "node:fs";

const PORT = process.argv[2] || "9223";
const OUT = process.argv[3] || "/tmp/studio-shots";
const BASE = "http://127.0.0.1:8000/studio.html";

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const log = (...a) => console.log(...a);

let id = 0;
const pending = new Map();
const errors = [];

const targets = await (await fetch(`http://127.0.0.1:${PORT}/json/list`)).json();
const page = targets.find((t) => t.type === "page");
if (!page) throw new Error("no page target");

const ws = new WebSocket(page.webSocketDebuggerUrl);
await new Promise((res, rej) => {
  ws.addEventListener("open", res, { once: true });
  ws.addEventListener("error", rej, { once: true });
});

ws.addEventListener("message", (ev) => {
  const msg = JSON.parse(ev.data);
  if (msg.id && pending.has(msg.id)) {
    const { resolve, reject } = pending.get(msg.id);
    pending.delete(msg.id);
    msg.error ? reject(new Error(JSON.stringify(msg.error))) : resolve(msg.result);
    return;
  }
  if (msg.method === "Log.entryAdded" && msg.params.entry.level === "error") {
    errors.push(`${msg.params.entry.source}: ${msg.params.entry.text}`);
  }
});

function send(method, params = {}) {
  return new Promise((resolve, reject) => {
    const myId = ++id;
    pending.set(myId, { resolve, reject });
    ws.send(JSON.stringify({ id: myId, method, params }));
  });
}

async function evaluate(expression) {
  const res = await send("Runtime.evaluate", {
    expression,
    awaitPromise: true,
    returnByValue: true,
  });
  if (res.exceptionDetails) {
    throw new Error("page error: " + JSON.stringify(res.exceptionDetails.exception?.description || res.exceptionDetails));
  }
  return res.result.value;
}

async function shot(name) {
  const { data } = await send("Page.captureScreenshot", { format: "png" });
  writeFileSync(`${OUT}/${name}.png`, Buffer.from(data, "base64"));
}

async function waitFor(expr, label, timeout = 20000) {
  const start = Date.now();
  for (;;) {
    if (await evaluate(expr)) return true;
    if (Date.now() - start > timeout) throw new Error(`timeout waiting for ${label}`);
    await sleep(250);
  }
}

const results = [];
const check = (name, ok, detail = "") => {
  results.push({ name, ok, detail });
  log(`${ok ? "PASS" : "FAIL"}  ${name}${detail ? " :: " + detail : ""}`);
};

await send("Page.enable");
await send("Runtime.enable");
await send("Log.enable");

await send("Page.navigate", { url: BASE });
await sleep(2500);

// --- 1. 状态灯由真实接口驱动 ---------------------------------------------
const providerText = await evaluate("document.getElementById('providerText').textContent");
check("状态灯显示已连接", providerText.includes("已连接"), providerText);

// --- 2. 文生图模式：尺寸出现、输入图被禁用 -------------------------------
await evaluate(`document.querySelector('.mode-button[data-mode="text2image"]').click()`);
await sleep(300);
const modeUi = await evaluate(`JSON.stringify({
  sizeVisible: !document.getElementById('sizeSelect').hidden,
  sizeCount: document.getElementById('sizeSelect').options.length,
  dropzoneDisabled: document.getElementById('dropzone').classList.contains('is-disabled'),
  hint: document.getElementById('sourceHint').textContent,
  model: document.getElementById('modelSelect').value
})`);
const ui = JSON.parse(modeUi);
check("文生图显示画布尺寸", ui.sizeVisible && ui.sizeCount > 0, `${ui.sizeCount} 个尺寸`);
check("文生图禁用输入图", ui.dropzoneDisabled, ui.hint);
check("文生图默认 Qwen-Image", ui.model === "Qwen/Qwen-Image", ui.model);

// --- 3. 写提示词 -> 费用预览 + 按钮可用 ---------------------------------
await evaluate(`(() => {
  const el = document.getElementById('promptInput');
  el.value = '北欧风格客厅，原木地板，落地窗自然光，米白沙发';
  el.dispatchEvent(new Event('input', { bubbles: true }));
})()`);
await sleep(400);
const preview = await evaluate("document.getElementById('costPreview').textContent");
const btnEnabled = await evaluate("!document.getElementById('generateButton').disabled");
check("费用预览说明付费次数", preview.includes("付费请求"), preview.replace(/\s+/g, " ").trim());
check("按钮变为可用", btnEnabled === true);
await shot("shot-02-text-mode");

// --- 4. 点生成 -> 只发 quote，弹出费用确认 ------------------------------
await evaluate("document.getElementById('generateButton').click()");
await waitFor("!document.getElementById('confirmBackdrop').hidden", "confirm dialog");
const facts = await evaluate(`Array.from(document.getElementById('confirmFacts').children).map(n => n.textContent)`);
check("弹出真实费用确认框", true, facts.join(" | "));
check("确认框写明付费次数", facts.some((f) => f.includes("1 次")), facts.join(" | "));
await shot("shot-03-confirm-modal");

// --- 5. 点取消 -> 弹窗关闭、绝不出请求 ----------------------------------
await evaluate("document.getElementById('confirmCancel').click()");
await sleep(600);
const afterCancel = await evaluate(`JSON.stringify({
  hidden: document.getElementById('confirmBackdrop').hidden,
  emptyVisible: !document.getElementById('resultStage').hidden,
  factsHidden: document.getElementById('resultFacts').hidden
})`);
const ac = JSON.parse(afterCancel);
check("取消后弹窗关闭", ac.hidden === true);
check("取消后没有产出结果", ac.emptyVisible === true && ac.factsHidden === true);

// --- 6. 改图模式：真实上传 ------------------------------------------------
await evaluate(`document.querySelector('.mode-button[data-mode="edit"]').click()`);
await sleep(300);
await evaluate(`(async () => {
  const blob = await (await fetch('/data/inputs/IMG_3651.jpg')).blob();
  const file = new File([blob], 'IMG_3651.jpg', { type: 'image/jpeg' });
  const dt = new DataTransfer();
  dt.items.add(file);
  const input = document.getElementById('fileInput');
  input.files = dt.files;
  input.dispatchEvent(new Event('change', { bubbles: true }));
})()`);
await waitFor("(document.getElementById('sourceMetaText').textContent.includes('×') && document.getElementById('sourceMetaText').textContent.includes('KB'))", "upload done");
const meta = await evaluate("document.getElementById('sourceMetaText').textContent");
const previewShown = await evaluate("!document.getElementById('sourcePreview').hidden");
check("上传后显示缩略图与信息", previewShown === true && meta.includes("904×1190"), meta);
await shot("shot-04-uploaded");

// --- 7. 改图模式报价：必须带上输入图 -------------------------------------
await evaluate(`(() => {
  const el = document.getElementById('promptInput');
  el.value = '把背景换成清晨的海边，光线柔和自然，人物面部和发型保持不变。';
  el.dispatchEvent(new Event('input', { bubbles: true }));
})()`);
await sleep(400);
await evaluate("document.getElementById('generateButton').click()");
await waitFor("!document.getElementById('confirmBackdrop').hidden", "confirm dialog 2");
const facts2 = await evaluate(`Array.from(document.getElementById('confirmFacts').children).map(n => n.textContent)`);
check("改图确认框带输入图名", facts2.some((f) => f.includes("img_")), facts2.join(" | "));
await shot("shot-05-confirm-edit");
await evaluate("document.getElementById('confirmCancel').click()");
await sleep(400);

log("\n--- 页面 JS 报错 ---");
log(errors.length ? errors.join("\n") : "无");
log("\n--- 汇总 ---");
log(`${results.filter((r) => r.ok).length}/${results.length} 通过`);
ws.close();
