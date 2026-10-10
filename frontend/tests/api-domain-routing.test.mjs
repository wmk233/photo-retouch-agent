import assert from "node:assert/strict";
import test from "node:test";
import { readFile } from "node:fs/promises";

// api.js is a browser ES module with no imports, so it can be loaded straight
// from source as a data: URL once the browser globals it touches are stubbed.
const source = await readFile(new URL("../js/api.js", import.meta.url), "utf8");
const apiModuleUrl = `data:text/javascript;base64,${Buffer.from(source, "utf8").toString("base64")}`;

globalThis.window = {
  PHOTO_AGENT_API_BASE: "http://agent.test/api",
  location: { port: "8000", origin: "http://agent.test" },
};

function stubFetch(reply) {
  const calls = [];
  globalThis.fetch = async (url, options = {}) => {
    calls.push({ url, options });
    return { ok: true, status: 200, json: async () => reply };
  };
  return calls;
}

test("analyzePhoto asks the server to infer the photo domain", async () => {
  const calls = stubFetch({ imageId: "img_1", domainType: "portrait" });
  const api = await import(apiModuleUrl);

  await api.analyzePhoto("img_1", { name: "local" }, { name: "mock" });

  assert.equal(calls.length, 1);
  assert.match(calls[0].url, /\/photos\/analyze$/);
  assert.equal(calls[0].options.method, "POST");
  const body = JSON.parse(calls[0].options.body);
  assert.equal(body.imageId, "img_1");
  assert.equal(body.domainType, "auto");
  assert.notEqual(body.domainType, "general");
  assert.notEqual(body.domainType, "portrait");
});

test("analyzePhoto still forwards the selected brain and action models", async () => {
  const calls = stubFetch({ imageId: "img_2", domainType: "portrait" });
  const api = await import(apiModuleUrl);

  await api.analyzePhoto(
    "img_2",
    { name: "qwen", apiKey: "brain-key" },
    { name: "qwen", apiKey: "action-key" },
  );

  assert.equal(calls[0].options.headers["X-Agent-Provider"], "qwen");
  assert.equal(calls[0].options.headers["X-Agent-API-Key"], "brain-key");
  assert.equal(calls[0].options.headers["X-Action-Provider"], "qwen");
  assert.equal(calls[0].options.headers["X-Action-API-Key"], "action-key");
});
