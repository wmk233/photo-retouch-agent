import assert from "node:assert/strict";
import test from "node:test";

import {
  buildRetouchInstruction,
  executeAiRetouch,
  selectRetouchPlan,
  validateModelSelection,
} from "../js/ai-retouch.mjs";

test("builds AI instruction from preset, tags, and user text", () => {
  const instruction = buildRetouchInstruction(
    { smooth: 30, eyeBags: 20, slimFace: 0 },
    "natural",
    "整体再自然一点",
  );

  assert.match(instruction, /自然原生肌/);
  assert.match(instruction, /光滑肌肤 30%/);
  assert.match(instruction, /去除眼袋 20%/);
  assert.doesNotMatch(instruction, /瘦脸/);
  assert.match(instruction, /整体再自然一点/);
  assert.match(instruction, /保持本人身份/);
});

test("selects a plan that matches the AI preset", () => {
  const plans = [
    { planId: "natural" },
    { planId: "clean" },
    { planId: "mood" },
  ];

  assert.equal(selectRetouchPlan(plans, "clear").planId, "clean");
  assert.equal(selectRetouchPlan(plans, "warm").planId, "mood");
  assert.equal(selectRetouchPlan(plans, "sculpt").planId, "natural");
});

test("validates provider API key requirements", () => {
  const capabilities = {
    brainProviders: [
      { id: "qwen", label: "Qwen Vision", requiresApiKey: true, configured: false },
    ],
    actionProviders: [
      { id: "mock", label: "本地模拟", requiresApiKey: false, configured: true },
    ],
  };

  assert.throws(
    () =>
      validateModelSelection(
        capabilities,
        { name: "qwen", apiKey: "" },
        { name: "mock", apiKey: "" },
      ),
    /Qwen Vision 需要 API Key/,
  );
  assert.doesNotThrow(() =>
    validateModelSelection(
      capabilities,
      { name: "qwen", apiKey: "session-key" },
      { name: "mock", apiKey: "" },
    ),
  );
});

test("executes upload, analysis, planning, and image editing in order", async () => {
  const calls = [];
  const api = {
    async uploadPhoto(file) {
      calls.push(["upload", file.name]);
      return { imageId: "img_test" };
    },
    async analyzePhoto(imageId, brain, action) {
      calls.push(["analyze", imageId, brain.name, action.name]);
      return { imageId, domainType: "portrait" };
    },
    async createPlans() {
      calls.push(["plans"]);
      return [
        {
          planId: "natural",
          editPrompt: "自然美化",
          expectedChanges: ["提亮"],
        },
      ];
    },
    async createJob(imageId, plan, instruction, brain, action) {
      calls.push([
        "job",
        imageId,
        plan.editPrompt,
        instruction,
        brain.apiKey,
        action.apiKey,
      ]);
      return {
        status: "succeeded",
        outputUrls: ["/data/outputs/result.png"],
      };
    },
  };

  const result = await executeAiRetouch({
    file: { name: "portrait.png" },
    values: { smooth: 20 },
    preset: "natural",
    userInstruction: "保留发丝",
    brain: { name: "qwen", apiKey: "brain-key" },
    action: { name: "qwen", apiKey: "action-key" },
    api,
  });

  assert.deepEqual(
    calls.map((call) => call[0]),
    ["upload", "analyze", "plans", "job"],
  );
  assert.match(calls[3][2], /光滑肌肤 20%/);
  assert.equal(calls[3][4], "brain-key");
  assert.equal(calls[3][5], "action-key");
  assert.equal(result.job.status, "succeeded");
});

test("surfaces a failed AI job safely", async () => {
  const api = {
    async uploadPhoto() {
      return { imageId: "img_test" };
    },
    async analyzePhoto() {
      return {};
    },
    async createPlans() {
      return [{ planId: "natural", editPrompt: "", expectedChanges: [] }];
    },
    async createJob() {
      return { status: "failed", outputUrls: [], errorMessage: "provider failed" };
    },
  };

  await assert.rejects(
    executeAiRetouch({
      file: { name: "portrait.png" },
      values: {},
      preset: "natural",
      userInstruction: "",
      brain: { name: "local" },
      action: { name: "mock" },
      api,
    }),
    /provider failed/,
  );
});
