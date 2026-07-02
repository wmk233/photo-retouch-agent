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

  if (job.status !== "succeeded" || !job.outputUrls?.length) {
    throw new Error(job.errorMessage || "AI 美化任务未生成结果。");
  }

  return { photo, analysis, plans, plan, instruction, job };
}
