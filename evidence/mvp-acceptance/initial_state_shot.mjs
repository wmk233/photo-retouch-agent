// Screenshot the app's first-load state: no photo, no fake detection card,
// and the visible upload affordance.
import { spawn } from "node:child_process";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const APP = "http://127.0.0.1:8000/";
const PORT = 9247;
const OUT = "evidence/mvp-acceptance/initial-state.png";
const PROFILE = mkdtempSync(join(tmpdir(), "init-chrome-"));
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const chrome = spawn(CHROME, [
  "--headless=new", "--disable-gpu", "--no-first-run", "--window-size=1440,1100",
  `--remote-debugging-port=${PORT}`, `--user-data-dir=${PROFILE}`, "about:blank",
], { stdio: "ignore" });

let targets = null;
for (let i = 0; i < 60; i++) {
  try { targets = await (await fetch(`http://127.0.0.1:${PORT}/json/list`)).json(); if (targets.some((t) => t.type === "page")) break; } catch {}
  await sleep(250);
}
const ws = new WebSocket(targets.find((t) => t.type === "page").webSocketDebuggerUrl);
let id = 0;
const pend = new Map();
ws.addEventListener("message", (ev) => {
  const m = JSON.parse(ev.data);
  if (m.id && pend.has(m.id)) { const { resolve } = pend.get(m.id); pend.delete(m.id); resolve(m.result); }
});
await new Promise((r) => ws.addEventListener("open", r));
const send = (method, params = {}) => { const i = ++id; return new Promise((res) => { pend.set(i, { resolve: res }); ws.send(JSON.stringify({ id: i, method, params })); }); };
const ev = async (expression) => (await send("Runtime.evaluate", { expression, returnByValue: true })).result.value;

await send("Page.enable");
await send("Runtime.enable");
await send("Page.navigate", { url: APP });
await sleep(4000);

console.log("emptyHint:", await ev(`document.querySelector("#workbenchEmpty")?.textContent.trim()`));
console.log("hintVisible:", await ev(`(() => { const el = document.querySelector("#workbenchEmpty"); return el ? getComputedStyle(el).display !== "none" : false; })()`));
console.log("detectionCard:", await ev(`document.querySelector("#detectionLabel")?.textContent.trim()`));
console.log("faceStats:", await ev(`JSON.stringify([...document.querySelectorAll("#detectionFaceCount,#detectionVisibility,#detectionLight")].map((e) => e.textContent.trim()))`));
console.log("topMeta:", await ev(`document.querySelector("#documentMeta")?.textContent.trim()`));
console.log("aiTabsVisible:", await ev(`(() => [...document.querySelectorAll("button, .mode-switch button")].filter((b) => /实时美化|AI 精修/.test(b.textContent)).map((b) => b.textContent.trim() + ":" + (b.offsetHeight > 0)).join(" | "))()`));

const r = await send("Page.captureScreenshot", { format: "png" });
writeFileSync(OUT, Buffer.from(r.data, "base64"));
console.log("[screenshot]", OUT);
ws.close();
chrome.kill();
