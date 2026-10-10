import assert from "node:assert/strict";
import test from "node:test";
import { readFile } from "node:fs/promises";

// app.js is DOM-coupled, so instead of importing it we assert on the source:
// the shared-key reuse table must let the action layer reuse the brain key when
// both sides are the same platform (e.g. SiliconFlow vision + image editing).
const source = await readFile(new URL("../js/app.js", import.meta.url), "utf8");

test("action layer reuses the brain key for same-platform providers", () => {
  const match = source.match(/const sharedKeyActions = \{(?<body>[\s\S]*?)\};/);
  assert.ok(match, "sharedKeyActions table should exist in app.js");

  const table = {};
  for (const line of match.groups.body.split("\n")) {
    const entry = line.match(/(\w+):\s*\[([^\]]*)\]/);
    if (!entry) continue;
    table[entry[1]] = entry[2]
      .split(",")
      .map((name) => name.trim().replace(/["']/g, ""))
      .filter(Boolean);
  }

  assert.deepEqual(table.siliconflow, ["siliconflow"]);
  assert.ok(table.qwen.includes("qwen"));
  assert.ok(table.qwen.includes("wan"));
  assert.deepEqual(table.doubao, ["seedream"]);
});

test("portrait routing is not hardcoded to a single domain", () => {
  assert.doesNotMatch(source, /domainType\s*=\s*["']portrait["']/);
});
