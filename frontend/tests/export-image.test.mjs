import assert from "node:assert/strict";
import test from "node:test";

import {
  buildExportFilename,
  canvasToBlob,
  triggerBlobDownload,
} from "../js/export-image.mjs";

test("builds safe local and AI export names", () => {
  assert.equal(
    buildExportFilename("my portrait.png", "local", "image/png"),
    "my-portrait-retouched.png",
  );
  assert.equal(
    buildExportFilename("头像:最终版.webp", "ai", "image/jpeg"),
    "头像-最终版-ai-retouched.jpg",
  );
});

test("converts a canvas to a blob", async () => {
  const expected = { type: "image/png", size: 128 };
  const canvas = {
    toBlob(callback, mimeType, quality) {
      assert.equal(mimeType, "image/png");
      assert.equal(quality, 0.94);
      callback(expected);
    },
  };

  assert.equal(await canvasToBlob(canvas), expected);
});

test("rejects unsupported canvas export", async () => {
  await assert.rejects(canvasToBlob(null), /无法导出画布/);
});

test("triggers a temporary browser download", () => {
  const calls = [];
  const anchor = {
    click() {
      calls.push("click");
    },
    remove() {
      calls.push("remove");
    },
  };
  const documentRef = {
    body: {
      append(value) {
        assert.equal(value, anchor);
        calls.push("append");
      },
    },
    createElement(tag) {
      assert.equal(tag, "a");
      return anchor;
    },
  };
  const urlApi = {
    createObjectURL() {
      calls.push("create");
      return "blob:test";
    },
    revokeObjectURL(url) {
      assert.equal(url, "blob:test");
      calls.push("revoke");
    },
  };

  const url = triggerBlobDownload({ size: 1 }, "result.png", {
    documentRef,
    urlApi,
    schedule(callback) {
      callback();
    },
  });

  assert.equal(url, "blob:test");
  assert.deepEqual(calls, ["create", "append", "click", "remove", "revoke"]);
  assert.equal(anchor.download, "result.png");
});
