// Count every HTTP request a single page load makes and their statuses,
// to prove/disprove the rate-limit hypothesis behind sporadic 500s.
import { spawn } from "node:child_process";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const APP = "http://127.0.0.1:8000/";
const PORT = 9241;
const PROFILE = mkdtempSync(join(tmpdir(), "probe-chrome-"));
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

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
const requests = new Map();
const responses = [];
const consoleErrors = [];
ws.addEventListener("message", (ev) => {
  const m = JSON.parse(ev.data);
  if (m.method === "Network.requestWillBeSent") {
    requests.set(m.params.requestId, { method: m.params.request.method, url: m.params.request.url });
  }
  if (m.method === "Network.responseReceived") {
    const info = requests.get(m.params.requestId) || { method: "?", url: m.params.response.url };
    responses.push({ ...info, status: m.params.response.status });
  }
  if (m.method === "Runtime.exceptionThrown") {
    consoleErrors.push(m.params.exceptionDetails.exception?.description || m.params.exceptionDetails.text);
  }
  if (m.method === "Log.entryAdded" && m.params.entry.level === "error") {
    consoleErrors.push(m.params.entry.text);
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

await send("Page.enable");
await send("Network.enable");
await send("Runtime.enable");
await send("Log.enable");
await send("Page.navigate", { url: APP });
await sleep(6000);

const api = responses.filter((r) => r.url.includes("/api/"));
const statics = responses.filter((r) => !r.url.includes("/api/"));
const bad = responses.filter((r) => r.status >= 400);
console.log("total requests:", responses.length);
console.log("static requests:", statics.length);
console.log("api requests:", api.length, api.map((r) => `${r.method} ${r.status} ${r.url.replace(APP.replace(/\/$/, ""), "")}`).join(" | "));
console.log("non-2xx:", bad.map((r) => `${r.method} ${r.status} ${r.url}`).join(" | ") || "(none)");
console.log("console errors:", consoleErrors.length ? consoleErrors.join(" || ") : "(none)");
ws.close();
chrome.kill();
