// Zero-cost self test for the in-page response recorder used by
// real_e2e_acceptance.mjs. It injects the exact RECORDER source from that file
// into a real app page, then performs a fetch() and an XHR against a watched
// path that cannot bill (GET /api/photos/analyze -> 405). If both bodies come
// back, the capture path that saved the first real run from aborting works.
import { spawn } from "node:child_process";
import { mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const APP = process.env.E2E_APP || "http://127.0.0.1:8001/";
const PORT = Number(process.env.E2E_CDP_PORT || 9227);
const src = readFileSync("evidence/mvp-acceptance/real_e2e_acceptance.mjs", "utf8");
const RECORDER = src.match(/const RECORDER = String\.raw`([\s\S]*?)\n`;/)[1];
const profile = mkdtempSync(join(tmpdir(), "recorder-selftest-"));
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const chrome = spawn(CHROME, ["--headless=new", "--disable-gpu", "--no-first-run",
  `--remote-debugging-port=${PORT}`, `--user-data-dir=${profile}`, "about:blank"], { stdio: "ignore" });

let page = null;
for (let i = 0; i < 80 && !page; i++) {
  try {
    const t = await fetch(`http://127.0.0.1:${PORT}/json/list`).then((r) => r.json());
    page = t.find((x) => x.type === "page");
  } catch {}
  if (!page) await sleep(250);
}
if (!page) { console.log("FAIL: no chrome page"); process.exit(1); }

const ws = new WebSocket(page.webSocketDebuggerUrl);
let id = 0; const pending = new Map();
ws.addEventListener("message", (ev) => {
  const m = JSON.parse(ev.data);
  if (m.id && pending.has(m.id)) { const p = pending.get(m.id); pending.delete(m.id); m.error ? p.reject(new Error(JSON.stringify(m.error))) : p.resolve(m.result); }
});
await new Promise((r) => ws.addEventListener("open", r));
const send = (method, params = {}) => new Promise((resolve, reject) => { const i = ++id; pending.set(i, { resolve, reject }); ws.send(JSON.stringify({ id: i, method, params })); });
const evalJs = async (expression) => (await send("Runtime.evaluate", { expression, returnByValue: true, awaitPromise: true })).result.value;

await send("Page.enable"); await send("Runtime.enable"); await send("Network.enable");
await send("Page.addScriptToEvaluateOnNewDocument", { source: RECORDER });
await send("Page.navigate", { url: APP });
for (let i = 0; i < 60; i++) { if (await evalJs("document.readyState === 'complete'")) break; await sleep(250); }
await sleep(500);
console.log("recorder installed:", await evalJs("Array.isArray(window.__e2eCaptured)"));

// watched path, POST via fetch (405, no billing), and XHR
const fetchStatus = await evalJs("fetch('/api/photos/analyze', { method: 'GET' }).then(r => r.status).catch(e => 'ERR ' + e.message)");
const xhrStatus = await evalJs(`new Promise((res) => { const x = new XMLHttpRequest(); x.open('GET', '/api/styling/proposals'); x.onload = () => res(x.status); x.onerror = () => res('ERR'); x.send(); })`);

const captured = await evalJs("window.__e2eCaptured");
const parsed = (captured || []).map((c) => { let ok = false; try { JSON.parse(c.body); ok = true; } catch {} return { url: c.url, method: c.method, status: c.status, jsonOk: ok, bytes: (c.body || "").length }; });
console.log("fetch status:", fetchStatus, "| xhr status:", xhrStatus);
console.log("captured:", JSON.stringify(parsed, null, 2));
const pass = Array.isArray(captured) && captured.length >= 2 && parsed.every((p) => p.jsonOk);
writeFileSync("evidence/mvp-acceptance/recorder_selftest.json", JSON.stringify({ pass, fetchStatus, xhrStatus, captured: parsed }, null, 2));
console.log(pass ? "SELFTEST PASS" : "SELFTEST FAIL");
ws.close(); chrome.kill();
process.exit(pass ? 0 : 1);
