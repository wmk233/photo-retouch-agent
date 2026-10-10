import assert from "node:assert/strict";
import test from "node:test";

import {
  buildStylingInstruction,
  designStyling,
  estimateStylingCost,
  executeStyling,
  normalizeStylingProposals,
  refineStyling,
} from "../js/ai-retouch.mjs";

function makeLook(id, overrides = {}) {
  return {
    planId: id,
    name: `造型 ${id}`,
    designAxis: id,
    hairPlan: `${id} 发型：耳下短发 / 层次 / 发色`,
    makeupPlan: `${id} 妆容：眉 / 眼 / 唇 / 底妆`,
    outfitPlan: `${id} 穿搭：款式 / 材质 / 颜色 / 搭配`,
    accessoryPlan: `${id} 配饰`,
    palette: `${id} 配色`,
    identityAnchors: ["保留圆润脸型轮廓", "保留原本肤色基调"],
    whyItFits: "贴合当前人物脸型与发量",
    expectedChanges: ["发型 → 短发"],
    editPrompt: `Restyle into ${id}. change the hair from current to ${id} hair; change the makeup from current to ${id} makeup; replace the current top with ${id} outfit.`,
    estimatedPaidRequests: 1,
    ...overrides,
  };
}

const THREE_LOOKS = [
  makeLook("urban_short"),
  makeLook("artistic_wave"),
  makeLook("bold_street"),
];

test("normalizeStylingProposals requires three distinct prompts", () => {
  assert.equal(normalizeStylingProposals(THREE_LOOKS).length, 3);

  assert.throws(() => normalizeStylingProposals(THREE_LOOKS.slice(0, 2)), /三套/);

  assert.throws(
    () =>
      normalizeStylingProposals([
        makeLook("a", { editPrompt: "same prompt" }),
        makeLook("b", { editPrompt: "same prompt" }),
        makeLook("c"),
      ]),
    /各不相同/,
  );

  assert.throws(
    () => normalizeStylingProposals([
      makeLook("a"),
      makeLook("b", { hairPlan: "" }),
      makeLook("c"),
    ]),
    /发型/,
  );
});

test("buildStylingInstruction merges the look and anchors without duplicating user text", () => {
  // The supplementary request travels as the job ``userInstruction`` and is
  // appended exactly once by the image provider; it must not be folded in here.
  const instruction = buildStylingInstruction(THREE_LOOKS[0], "头发再短一点");
  assert.match(instruction, /urban_short/);
  assert.match(instruction, /保留本人辨识特征/);
  assert.match(instruction, /保留圆润脸型轮廓/);
  assert.doesNotMatch(instruction, /用户补充要求/);
  assert.doesNotMatch(instruction, /头发再短一点/);
  assert.match(instruction, /不保证百分之百/);
});

test("estimateStylingCost reports requests and the free design phase", () => {
  const one = estimateStylingCost(THREE_LOOKS, { selectedCount: 1 });
  assert.equal(one.totalRequests, 1);
  assert.match(one.note, /免费生成/);
  assert.match(one.note, /1 次/);

  const two = estimateStylingCost(THREE_LOOKS, { selectedCount: 2 });
  assert.equal(two.totalRequests, 2);
});

test("estimateStylingCost separates free template preview from the AI design flow", () => {
  const template = estimateStylingCost(THREE_LOOKS, { selectedCount: 1, mode: "template" });
  assert.equal(template.designMode, "template");
  assert.equal(template.analysisRequests, 0);
  assert.equal(template.designRequests, 0);
  assert.equal(template.generationRequests, 0);
  assert.equal(template.paidDesignRequests, 0);
  assert.equal(template.paidFlowRequests, 0);
  assert.match(template.note, /免费生成/);

  const ai = estimateStylingCost(THREE_LOOKS, { selectedCount: 1, mode: "ai" });
  assert.equal(ai.designMode, "ai");
  assert.equal(ai.analysisRequests, 1);
  assert.equal(ai.designRequests, 1);
  // Generating the chosen look also bills one prompt-planning pass, so the
  // complete flow is three paid calls, not the two that only counted design.
  assert.equal(ai.generationRequests, 1);
  assert.equal(ai.paidDesignRequests, 2);
  assert.equal(ai.paidFlowRequests, 3);
  assert.equal(ai.analysisReused, false);
  assert.match(ai.note, /3 次付费模型调用/);
  assert.match(ai.note, /1 次人物事实分析/);
  // A text-only brain must be labelled as never seeing the photo.
  assert.equal(ai.visionDerived, true);
  assert.match(ai.visionNote, /FaceFacts/);
  assert.match(ai.visionNote, /并未直接/);
  // Generation cost is tracked separately: one look, one image edit.
  assert.equal(ai.totalRequests, 1);
});

test("estimateStylingCost counts a reused analysis as free", () => {
  const ai = estimateStylingCost(THREE_LOOKS, {
    selectedCount: 1,
    mode: "ai",
    analysisReused: true,
  });
  assert.equal(ai.analysisReused, true);
  assert.equal(ai.analysisRequests, 1);
  assert.equal(ai.designRequests, 1);
  // The analysis was served from cache, so only design + planning are billed.
  assert.equal(ai.paidDesignRequests, 1);
  assert.equal(ai.paidFlowRequests, 2);
  assert.match(ai.note, /复用/);
  assert.match(ai.note, /2 次付费模型调用/);
});

test("designStyling forwards the design mode and user text, and never mislabels templates as AI", async () => {
  const calls = [];
  const api = {
    async uploadPhoto() {
      return { imageId: "img_1" };
    },
    async analyzePhoto() {
      return { imageId: "img_1", domainType: "portrait" };
    },
    async createStylingProposals(analysis, options) {
      calls.push(options);
      return THREE_LOOKS.map((look) => ({ ...look, designMode: options.mode }));
    },
  };

  const aiResult = await designStyling({
    file: { name: "portrait.png" },
    brain: { name: "siliconflow", apiKey: "sk-x" },
    action: { name: "siliconflow" },
    mode: "ai",
    userInstruction: "想要更干练",
    api,
  });
  assert.equal(aiResult.mode, "ai");
  assert.equal(aiResult.analysisReused, false);
  assert.equal(calls[0].mode, "ai");
  assert.equal(calls[0].userInstruction, "想要更干练");
  assert.equal(calls[0].brain.name, "siliconflow");
  assert.ok(aiResult.proposals.every((look) => look.designMode === "ai"));

  const templateResult = await designStyling({
    file: { name: "portrait.png" },
    brain: {},
    action: {},
    api,
  });
  assert.equal(templateResult.mode, "template");
  assert.equal(calls[1].mode, "template");
  assert.ok(templateResult.proposals.every((look) => look.designMode !== "ai"));
});

test("designStyling surfaces a cached analysis so the caller can report one paid call", async () => {
  const api = {
    async uploadPhoto() {
      return { imageId: "img_reuse" };
    },
    async analyzePhoto() {
      // The backend serves a cached analysis: no provider call was billed.
      return { imageId: "img_reuse", domainType: "portrait", analysisReused: true };
    },
    async createStylingProposals() {
      return THREE_LOOKS.map((look) => ({ ...look, designMode: "ai" }));
    },
  };

  const result = await designStyling({
    file: { name: "portrait.png" },
    brain: { name: "siliconflow", apiKey: "sk-x" },
    action: { name: "siliconflow" },
    mode: "ai",
    api,
  });

  assert.equal(result.analysisReused, true);
  const cost = estimateStylingCost(result.proposals, {
    mode: "ai",
    analysisReused: result.analysisReused,
  });
  assert.equal(cost.paidDesignRequests, 1);
  assert.equal(cost.paidFlowRequests, 2);
});

test("designs three looks, then generates only the selected look from the original", async () => {
  const calls = [];
  let submittedPrompt = "";
  let submittedInstruction = "";
  const api = {
    async uploadPhoto(file) {
      calls.push(["upload", file.name]);
      return { imageId: "img_original" };
    },
    async analyzePhoto(imageId, brain, action) {
      calls.push(["analyze", imageId, brain.name, action.name]);
      return { imageId, domainType: "portrait" };
    },
    async createStylingProposals(analysis) {
      calls.push(["proposals", analysis.imageId]);
      return THREE_LOOKS;
    },
    async createJob(sourceImageId, plan, instruction, brain, action) {
      calls.push(["job", sourceImageId, plan.planId, brain.apiKey]);
      submittedPrompt = plan.editPrompt;
      submittedInstruction = instruction;
      return {
        jobId: "job_1",
        status: "succeeded",
        outputUrls: ["/data/outputs/styled.jpg"],
      };
    },
    async refineJob(jobId, instruction) {
      calls.push(["refine", jobId, instruction]);
      return {
        jobId: "job_2",
        status: "succeeded",
        outputUrls: ["/data/outputs/styled-2.jpg"],
      };
    },
  };

  const brain = { name: "local", apiKey: "" };
  const action = { name: "mock", apiKey: "" };

  const { photo, proposals } = await designStyling({
    file: { name: "portrait.png" },
    brain,
    action,
    api,
  });
  assert.equal(proposals.length, 3);

  const result = await executeStyling({
    photo,
    proposals,
    selectedId: "bold_street",
    userInstruction: "",
    brain,
    action,
    api,
  });

  // upload → analyze → proposals → exactly one job, always from the original.
  assert.deepEqual(calls.map((call) => call[0]), [
    "upload",
    "analyze",
    "proposals",
    "job",
  ]);
  assert.equal(calls[3][1], "img_original");
  assert.equal(calls[3][2], "bold_street");
  assert.equal(result.proposal.planId, "bold_street");
  assert.equal(result.job.outputUrls[0], "/data/outputs/styled.jpg");

  // The prompt handed to the image editor only describes the chosen look.
  assert.match(submittedPrompt, /bold_street/);
  assert.doesNotMatch(submittedPrompt, /urban_short/);
  assert.doesNotMatch(submittedPrompt, /artistic_wave/);
  assert.match(submittedPrompt, /change the hair from current to bold_street hair/);
  assert.match(submittedPrompt, /保留本人辨识特征/);
  assert.equal(submittedInstruction, "");

  const refined = await refineStyling({
    jobId: result.job.jobId,
    userInstruction: "衣服换成皮衣",
    brain,
    action,
    api,
  });
  assert.equal(calls[4][0], "refine");
  assert.equal(calls[4][1], "job_1");
  assert.equal(calls[4][2], "衣服换成皮衣");
  assert.equal(refined.job.outputUrls[0], "/data/outputs/styled-2.jpg");

  // The user's own words ride along once, and only for the look being edited.
  const withExtra = await executeStyling({
    photo,
    proposals,
    selectedId: "artistic_wave",
    userInstruction: "头发再短一点",
    brain,
    action,
    api,
  });
  assert.equal(withExtra.proposal.planId, "artistic_wave");
  // The look prompt never repeats the user's own words...
  assert.doesNotMatch(submittedPrompt, /用户补充要求/);
  assert.doesNotMatch(submittedPrompt, /头发再短一点/);
  assert.match(submittedPrompt, /artistic_wave/);
  assert.doesNotMatch(submittedPrompt, /bold_street/);
  // ...they ride along exactly once as the job's userInstruction.
  assert.equal(submittedInstruction, "头发再短一点");
});

test("executeStyling rejects an unknown look and a missing photo", async () => {
  const api = {
    async createJob() {
      throw new Error("should not be called");
    },
  };

  await assert.rejects(
    executeStyling({
      photo: { imageId: "img_original" },
      proposals: THREE_LOOKS,
      selectedId: "nope",
      api,
    }),
    /请选择一套/,
  );

  await assert.rejects(
    executeStyling({ photo: null, proposals: THREE_LOOKS, selectedId: "urban_short", api }),
    /缺少原图/,
  );
});

test("refineStyling requires an instruction", async () => {
  await assert.rejects(
    refineStyling({ jobId: "job_1", userInstruction: "   ", api: {} }),
    /修改要求/,
  );
  await assert.rejects(refineStyling({ api: {} }), /先生成一套造型/);
});

test("a styling job that fails the strict face lock is reported as an identity failure", async () => {
  const api = {
    async createJob() {
      return { jobId: "job_locked", status: "queued" };
    },
    async getJob() {
      return {
        jobId: "job_locked",
        status: "failed",
        outputUrls: [],
        errorMessage: "保脸失败（alignment_failed）：…",
        faceLock: {
          status: "alignment_failed",
          applied: false,
          ok: false,
          message: "Face rotation differs by 12.0deg (limit 8.0deg).",
          diagnostics: { mode: "strict" },
        },
      };
    },
  };

  await assert.rejects(
    executeStyling({
      photo: { imageId: "img_original" },
      proposals: THREE_LOOKS,
      selectedId: "urban_short",
      brain: { name: "local" },
      action: { name: "mock" },
      api,
    }),
    (error) => {
      // The failure carries the face-lock report so the UI never presents a
      // failed identity lock as a successful styling.
      assert.equal(error.faceLock.ok, false);
      assert.equal(error.faceLock.status, "alignment_failed");
      assert.equal(error.jobStatus, "failed");
      return true;
    },
  );
});

test("a successful styling job keeps face-lock mode available to the UI", async () => {
  const api = {
    async createJob() {
      return {
        jobId: "job_ok",
        status: "succeeded",
        outputUrls: ["/data/outputs/locked.png"],
        faceLock: { status: "applied", applied: true, ok: true, diagnostics: { mode: "strict" } },
      };
    },
  };

  const result = await executeStyling({
    photo: { imageId: "img_original" },
    proposals: THREE_LOOKS,
    selectedId: "urban_short",
    brain: { name: "local" },
    action: { name: "mock" },
    api,
  });

  assert.equal(result.job.faceLock.ok, true);
  assert.equal(result.job.outputUrls[0], "/data/outputs/locked.png");
});
