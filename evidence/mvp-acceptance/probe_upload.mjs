import { spawn } from "node:child_process";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const APP = "http://127.0.0.1:8000/";
const PORT = 9231;
const REPO = process.cwd();
const PROFILE = mkdtempSync(join(tmpdir(), "probe-chrome-"));
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const chrome = spawn(CHROME, ["--headless=new","--disable-gpu","--no-first-run","--no-default-browser-check",
  `--remote-debugging-port=${PORT}`,`--user-data-dir=${PROFILE}`,"about:blank"], { stdio: "ignore" });
let targets = null;
for (let i = 0; i < 80; i++) {
  try { const r = await fetch(`http://127.0.0.1:${PORT}/json/list`); targets = await r.json(); if (targets.some(t=>t.type==="page")) break; } catch {}
  await sleep(250);
}
const page = targets.find(t=>t.type==="page");
const ws = new WebSocket(page.webSocketDebuggerUrl);
let id = 0; const pending = new Map(); const logs = [];
ws.addEventListener("message", (ev) => { const m = JSON.parse(ev.data);
  if (m.id && pending.has(m.id)) { const p = pending.get(m.id); pending.delete(m.id); m.error ? p.reject(new Error(JSON.stringify(m.error))) : p.resolve(m.result); }
  else if (m.method === "Runtime.consoleAPICalled") logs.push(m.params.args.map(a=>a.value ?? a.description).join(" "));
});
await new Promise(r => ws.addEventListener("open", r));
const send = (method, params={}) => { const i = ++id; return new Promise((res, rej) => { pending.set(i, {resolve:res, reject:rej}); ws.send(JSON.stringify({id:i, method, params})); }); };
const evalJs = async (e) => (await send("Runtime.evaluate", { expression: e, returnByValue: true, awaitPromise: true })).result.value;

await send("Page.enable"); await send("Runtime.enable"); await send("DOM.enable");
await send("Page.setInterceptFileChooserDialog", { enabled: true });
await send("Page.navigate", { url: APP });
await sleep(3000);

const o1 = await evalJs(`document.querySelector('#canvasStatus').textContent.trim()`);
console.log("before upload:", JSON.stringify(o1));

const doc = await send("DOM.getDocument", { depth: -1 });
const n = await send("DOM.querySelector", { nodeId: doc.root.nodeId, selector: "#fileInput" });
await send("DOM.setFileInputFiles", { nodeId: n.nodeId, files: [join(REPO, "data/inputs/IMG_3651.jpg")] });

for (const t of [0, 200, 500, 1000, 2000, 3000]) {
  await sleep(t === 0 ? 0 : t);
  const snap = await evalJs(`JSON.stringify({t:${t}, cs: document.querySelector('#canvasStatus').textContent.trim(), html: document.querySelector('#canvasStatus').innerHTML.replace(/\\s+/g,' ').trim().slice(0,120), dm: document.querySelector('#documentMeta').textContent.trim(), pm: document.querySelector('#portraitMeta').textContent.trim(), loaded: document.querySelector('#imageWorkbench').dataset.loaded, natural: document.querySelector('#beforeImage').naturalWidth})`);
  console.log("after +", snap);
}
console.log("console logs:", JSON.stringify(logs.filter(l=>/photo|upload|error/i.test(l))));
process.exit(0);
