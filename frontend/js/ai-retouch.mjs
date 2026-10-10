const adjustmentLabels = {
  smooth: "光滑肌肤",
  skinTone: "肤色均匀",
  whiten: "自然美白",
  sculpt: "立体光影",
  acne: "面部祛痘",
  wrinkle: "淡化细纹",
  slimFace: "瘦脸",
  smallFace: "小脸",
  jawline: "下颌线",
  chin: "下巴",
  enlargeEyes: "放大眼睛",
  eyeBags: "去除眼袋",
  darkCircles: "淡化黑眼圈",
  iris: "眼眸增强",
  brightEyes: "亮眼",
  noseBridge: "立体鼻梁",
  slimNose: "瘦鼻",
  teeth: "牙齿美白",
  smile: "自然微笑",
  slimBelly: "瘦肚子",
  slimWaist: "细腰",
  slimArms: "瘦手臂",
  slimLegs: "瘦腿",
  longLegs: "长腿",
  foundation: "底妆",
  blush: "腮红",
  contour: "修容",
  lipstick: "口红",
  brightness: "亮度",
  contrast: "对比度",
  warmth: "色温",
  saturation: "饱和度",
  clarity: "清晰度",
  background: "背景净化",
};

const presetInstructions = {
  natural: "采用自然原生肌风格，保留毛孔和真实皮肤纹理，避免明显修图感。",
  clear: "采用清透职业照风格，画面明亮干净，肤色自然，整体保持克制。",
  sculpt: "采用立体轮廓风格，增强自然光影和面部层次，避免改变身份特征。",
  warm: "采用暖调氛围风格，肤色健康柔和，背景色调协调统一。",
};

const presetPlans = {
  natural: ["natural", "clean", "mood"],
  clear: ["clean", "natural", "mood"],
  sculpt: ["natural", "clean", "mood"],
  warm: ["mood", "natural", "clean"],
};

export function buildRetouchInstruction(
  values = {},
  preset = "natural",
  userInstruction = "",
) {
  const adjustments = Object.entries(values)
    .filter(([id, value]) => adjustmentLabels[id] && Number(value) > 0)
    .sort((a, b) => Number(b[1]) - Number(a[1]))
    .slice(0, 12)
    .map(([id, value]) => `${adjustmentLabels[id]} ${Math.round(Number(value))}%`);

  return [
    presetInstructions[preset] || presetInstructions.natural,
    adjustments.length ? `按以下强度精修：${adjustments.join("、")}。` : "",
    userInstruction.trim(),
    "保持本人身份、五官比例、身体结构和真实皮肤质感，避免过度美化与背景变形。",
  ]
    .filter(Boolean)
    .join("\n");
}

export function selectRetouchPlan(plans, preset = "natural") {
  if (!Array.isArray(plans) || plans.length === 0) {
    throw new Error("AI 未返回可用的美化方案。");
  }

  const preferredIds = presetPlans[preset] || presetPlans.natural;
  return (
    preferredIds
      .map((planId) => plans.find((plan) => plan.planId === planId))
      .find(Boolean) || plans[0]
  );
}

export function validateModelSelection(capabilities, brain, action) {
  const brainMeta = capabilities.brainProviders?.find(
    (provider) => provider.id === brain.name,
  );
  const actionMeta = capabilities.actionProviders?.find(
    (provider) => provider.id === action.name,
  );

  if (!brainMeta) throw new Error("请选择可用的视觉分析模型。");
  if (!actionMeta) throw new Error("请选择可用的图像编辑模型。");
  if (brainMeta.requiresApiKey && !brainMeta.configured && !brain.apiKey) {
    throw new Error(`${brainMeta.label} 需要 API Key。`);
  }
  if (actionMeta.requiresApiKey && !actionMeta.configured && !action.apiKey) {
    throw new Error(`${actionMeta.label} 需要 API Key。`);
  }
  return { brainMeta, actionMeta };
}

export async function executeAiRetouch({
  file,
  values,
  preset,
  userInstruction,
  brain,
  action,
  api,
}) {
  if (!file) throw new Error("请先选择一张照片。");

  const photo = await api.uploadPhoto(file);
  const analysis = await api.analyzePhoto(photo.imageId, brain, action);
  const plans = await api.createPlans(analysis);
  const selectedPlan = selectRetouchPlan(plans, preset);
  const instruction = buildRetouchInstruction(values, preset, userInstruction);
  const plan = {
    ...selectedPlan,
    editPrompt: `${selectedPlan.editPrompt}\n${instruction}`.trim(),
    expectedChanges: [
      ...(selectedPlan.expectedChanges || []),
      "按标签参数完成细粒度人像美化",
    ],
  };
  const job = await api.createJob(
    photo.imageId,
    plan,
    instruction,
    brain,
    action,
  );

  // ``POST /retouch/jobs`` answers 202 with a *queued* job -- the render runs
  // on the backend dispatcher. Treating that reply as terminal reported
  // "AI 美化任务未生成结果" the instant the request returned, even though the
  // job was still running (and even when it later succeeded). Poll the job to
  // its real outcome instead, exactly like the styling flow does.
  const finished = await resolveJobUntilDone(api, job, {
    failureMessage: "AI 美化任务未生成结果。",
    timeoutMessage: "AI 美化生成超时，请稍后重试。",
  });

  return { photo, analysis, plans, plan, instruction, job: finished };
}

// A masked RunningHub render can legitimately spend several minutes in the
// provider queue, so the poll window is generous; the UI reports the job's own
// ``stage`` while it waits, so a long wait is visible rather than silent.
const STYLING_POLL_TIMEOUT_MS = 900000;
// How often the job's ``stage`` is echoed back to the caller.
const STYLING_POLL_INTERVAL_MS = 900;

export function normalizeStylingProposals(proposals) {
  if (!Array.isArray(proposals) || proposals.length < 3) {
    throw new Error("AI 造型设计需要三套完整方案。");
  }

  const looks = proposals.slice(0, 3);
  const ids = looks.map((look) => look.planId || look.designAxis);
  if (ids.some((id) => !id) || new Set(ids).size !== ids.length) {
    throw new Error("造型方案标识重复，无法区分不同设计。");
  }

  const prompts = looks.map((look) => (look.editPrompt || "").trim());
  if (prompts.some((prompt) => !prompt)) {
    throw new Error("每套造型都必须提供独立的图像编辑指令。");
  }
  if (new Set(prompts).size !== prompts.length) {
    throw new Error("三套造型的编辑指令必须各不相同，不能复用同一段 Prompt。");
  }

  for (const look of looks) {
    if (!look.hairPlan || !look.makeupPlan || !look.outfitPlan) {
      throw new Error(`造型「${look.name || look.planId}」缺少发型／妆容／穿搭说明。`);
    }
  }

  return looks;
}

export function buildStylingInstruction(proposal) {
  // The user's supplementary request is deliberately *not* folded in here: it
  // travels as the job's ``userInstruction`` and is appended exactly once by
  // the image provider as the final hop. Adding it here as well would send the
  // same sentence to the model twice.
  const parts = [
    `按「${proposal.name || proposal.title}」造型方案改造本人照片。`,
    (proposal.editPrompt || "").trim(),
  ];
  if (proposal.identityAnchors?.length) {
    parts.push(`保留本人辨识特征（尽力而为，不保证百分之百）：${proposal.identityAnchors.join("、")}。`);
  }
  return parts.filter(Boolean).join("\n");
}

export function estimateStylingCost(
  proposals,
  {
    selectedCount = 1,
    mode = "template",
    analysisReused = false,
    brainIsPaid = true,
  } = {},
) {
  const looks = Array.isArray(proposals) ? proposals : [];
  const isAi = mode === "ai";
  const perLook = Math.max(1, Number(looks[0]?.estimatedPaidRequests) || 1);
  const count = Math.max(1, Math.min(Number(selectedCount) || 1, looks.length || 1));

  // The complete AI flow is *three* billable model calls: one
  // ``POST /photos/analyze`` pass, one ``POST /styling/proposals`` pass, and one
  // ``brain.optimize`` prompt-planning pass inside ``POST /retouch/jobs`` when
  // the chosen look is generated. Reporting only the design pass under-counted
  // the real cost. When the backend reuses an existing analysis for the same
  // photo and model, the analysis pass is free, so the honest estimate is two.
  const analysisRequests = isAi ? 1 : 0;
  const designRequests = isAi ? 1 : 0;
  const generationRequests = isAi && brainIsPaid ? 1 : 0;
  const reused = isAi && analysisReused === true;
  const paidAnalysisRequests = reused ? 0 : analysisRequests;
  const paidDesignRequests = isAi
    ? paidAnalysisRequests + designRequests
    : 0;
  const paidFlowRequests = paidDesignRequests + generationRequests;

  const designNote = isAi
    ? reused
      ? `AI 个性化设计：已复用该照片的有效分析缓存（分析 0 次调用），本流程共 ${paidFlowRequests} 次付费模型调用 = 1 次造型设计 + 1 次出图提示词规划。`
      : `AI 个性化设计：完整流程含 1 次人物事实分析 + 1 次造型设计 + 1 次出图提示词规划，共 ${paidFlowRequests} 次付费模型调用，点击后一次性生成三套方案。`
    : "免费模板预览：0 次模型调用，免费生成，不产生任何费用。";

  return {
    designMode: isAi ? "ai" : "template",
    analysisRequests,
    designRequests,
    generationRequests,
    analysisReused: reused,
    paidDesignRequests,
    paidFlowRequests,
    visionDerived: isAi,
    visionNote: isAi
      ? "当前模型不支持图像输入：DeepSeek 只依据本地程序测量的 FaceFacts 设计，并未直接对上传的原图做视觉看图分析。"
      : "免费模板预览由本地规则组合，不调用任何模型。",
    selectedCount: count,
    requestsPerLook: perLook,
    totalRequests: perLook * count,
    note: `${designNote}出图只对选中的 ${count} 套各发起 ${perLook} 次图像编辑请求，合计约 ${perLook * count} 次。`,
  };
}

export async function designStyling({
  file,
  brain,
  action,
  mode = "template",
  userInstruction = "",
  api,
}) {
  if (!file) throw new Error("请先选择一张照片。");

  const photo = await api.uploadPhoto(file);
  const analysis = await api.analyzePhoto(photo.imageId, brain, action);
  const proposals = normalizeStylingProposals(
    await api.createStylingProposals(analysis, { mode, brain, userInstruction }),
  );
  // The backend flags a cached analysis so the caller can report the real
  // number of paid calls (one analysis + one design, or just the design).
  return { photo, analysis, proposals, mode, analysisReused: analysis?.analysisReused === true };
}

function stylingJobFailure(job, fallback) {
  const error = new Error(job?.errorMessage || fallback);
  // Carry the strict face-lock report (when present) so the UI can tell an
  // identity-preservation failure apart from any other job failure instead of
  // silently presenting it as a successful styling.
  error.faceLock = job?.faceLock || null;
  error.jobStatus = job?.status || "failed";
  // Keep the whole job so the progress panel can mark the exact failed stage and
  // the failure reason can be shown without a second lookup.
  error.job = job || null;
  return error;
}

// Both the styling and the skin-retouch flows create jobs through the same 202
// endpoint, so they share one poll-to-terminal helper. ``failureMessage`` is
// used when the backend reports a failure (and as the fallback when no reason
// is attached); ``timeoutMessage`` is used when the poll window is exhausted.
async function resolveJobUntilDone(
  api,
  created,
  { onStage, failureMessage, timeoutMessage },
) {
  const report = (job) => {
    if (typeof onStage === "function") {
      try {
        onStage(job);
      } catch (error) {
        // Progress reporting must never break the render.
      }
    }
  };
  report(created);
  if (created?.status === "succeeded" && created.outputUrls?.length) {
    return created;
  }
  if (created?.status === "failed") {
    throw stylingJobFailure(created, failureMessage);
  }
  if (!created?.jobId || typeof api.getJob !== "function") {
    throw new Error(created?.errorMessage || failureMessage);
  }

  const deadline = Date.now() + STYLING_POLL_TIMEOUT_MS;
  let delay = STYLING_POLL_INTERVAL_MS;
  let job = created;
  while (Date.now() < deadline) {
    await new Promise((resolve) => setTimeout(resolve, delay));
    job = await api.getJob(created.jobId);
    report(job);
    if (job?.status === "failed") {
      throw stylingJobFailure(job, failureMessage);
    }
    if (job?.status === "succeeded" && job.outputUrls?.length) {
      return job;
    }
  }
  throw new Error(timeoutMessage);
}

async function resolveStylingJob(api, created, onStage) {
  return resolveJobUntilDone(api, created, {
    onStage,
    failureMessage: "AI 造型任务生成失败。",
    timeoutMessage: "AI 造型生成超时，请稍后重试。",
  });
}

export async function executeStyling({
  photo,
  proposals,
  selectedId,
  userInstruction = "",
  brain,
  action,
  api,
  onStage,
}) {
  const looks = normalizeStylingProposals(proposals);
  const proposal = looks.find(
    (look) => (look.planId || look.designAxis) === selectedId,
  );
  if (!proposal) throw new Error("请选择一套要生成的造型。");
  if (!photo?.imageId) throw new Error("缺少原图，请重新上传照片后再生成。");

  const instruction = buildStylingInstruction(proposal);
  const plan = {
    ...proposal,
    editPrompt: instruction,
  };
  const created = await api.createJob(
    photo.imageId,
    plan,
    (userInstruction || "").trim(),
    brain,
    action,
  );
  const job = await resolveStylingJob(api, created, onStage);
  return { proposal, instruction, job };
}

export async function refineStyling({
  jobId,
  userInstruction,
  brain,
  action,
  api,
  onStage,
}) {
  if (!jobId) throw new Error("请先生成一套造型，再进行二次修改。");
  const instruction = (userInstruction || "").trim();
  if (!instruction) throw new Error("请填写你的修改要求。");

  const created = await api.refineJob(jobId, instruction, brain, action);
  const job = await resolveStylingJob(api, created, onStage);
  return { job };
}
