const MAX_FILE_SIZE = 10 * 1024 * 1024;
const ACCEPTED_TYPES = new Set(["image/jpeg", "image/png", "image/webp"]);

const state = {
  imageId: null,
  sourceDataUrl: null,
  sourceName: null,
  sourceSize: 0,
  sourceType: null,
  sourceImage: null,
  analysis: null,
  plans: [],
  selectedPlanId: null,
  resultDataUrl: null,
  activeBaseDataUrl: null,
  history: [],
};

const $ = (selector) => document.querySelector(selector);

const els = {
  fileInput: $("#fileInput"),
  uploadButton: $("#uploadButton"),
  sampleButton: $("#sampleButton"),
  sourceImage: $("#sourceImage"),
  sourceEmpty: $("#sourceEmpty"),
  photoMeta: $("#photoMeta"),
  globalStatus: $("#globalStatus"),
  analyzeButton: $("#analyzeButton"),
  analysisEmpty: $("#analysisEmpty"),
  analysisContent: $("#analysisContent"),
  sceneType: $("#sceneType"),
  subjectSummary: $("#subjectSummary"),
  riskSummary: $("#riskSummary"),
  analysisGroups: $("#analysisGroups"),
  planList: $("#planList"),
  planCount: $("#planCount"),
  resultEmpty: $("#resultEmpty"),
  comparison: $("#comparison"),
  compareBefore: $("#compareBefore"),
  compareAfter: $("#compareAfter"),
  compareSlider: $("#compareSlider"),
  afterLayer: $("#afterLayer"),
  splitLine: $("#splitLine"),
  resultStage: $("#resultStage"),
  jobOverlay: $("#jobOverlay"),
  jobText: $("#jobText"),
  executionNote: $("#executionNote"),
  regenerateButton: $("#regenerateButton"),
  downloadButton: $("#downloadButton"),
  refineForm: $("#refineForm"),
  instructionInput: $("#instructionInput"),
  refineButton: $("#refineButton"),
  historyStrip: $("#historyStrip"),
};

const planTemplates = [
  {
    planId: "natural",
    title: "自然美化",
    description: "轻微提亮，统一肤色和整体质感，保留真实本人感。",
    intensity: "natural",
    negativePrompt: "避免过度磨皮、避免改变五官、避免塑料质感",
    expectedChanges: ["提亮", "肤色统一", "轻质感"],
  },
  {
    planId: "avatar",
    title: "精致头像",
    description: "清理视觉干扰，强化面部清晰度，输出更适合头像的构图。",
    intensity: "medium",
    negativePrompt: "避免强换脸、避免证件照伪造、避免背景脏纹",
    expectedChanges: ["头像裁切", "背景弱化", "脸部清晰"],
  },
  {
    planId: "mood",
    title: "氛围风格",
    description: "增强色彩情绪和层次，保留自然人像基础。",
    intensity: "medium",
    negativePrompt: "避免色偏过重、避免人像 AI 化、避免皮肤失真",
    expectedChanges: ["胶片感", "色彩层次", "轻颗粒"],
  },
];

els.uploadButton.addEventListener("click", () => els.fileInput.click());
els.fileInput.addEventListener("change", handleFileSelect);
els.sampleButton.addEventListener("click", loadSampleImage);
els.analyzeButton.addEventListener("click", runAnalysis);
els.regenerateButton.addEventListener("click", () => {
  const plan = getSelectedPlan();
  if (plan) startRetouchJob({ plan, instruction: "", baseDataUrl: state.sourceDataUrl });
});
els.downloadButton.addEventListener("click", downloadResult);
els.refineForm.addEventListener("submit", handleRefine);
els.compareSlider.addEventListener("input", updateComparison);
window.addEventListener("resize", syncStageWidth);

function setStatus(label, tone = "idle") {
  const dotClass = tone === "running" ? "running" : tone === "done" ? "done" : tone === "error" ? "error" : "idle";
  els.globalStatus.innerHTML = `<span class="status-dot ${dotClass}"></span><span>${label}</span>`;
}

async function handleFileSelect(event) {
  const file = event.target.files?.[0];
  if (!file) return;

  if (!ACCEPTED_TYPES.has(file.type)) {
    showError("仅支持 JPG、PNG、WebP");
    return;
  }

  if (file.size > MAX_FILE_SIZE) {
    showError("图片超过 10MB");
    return;
  }

  const dataUrl = await readFileAsDataUrl(file);
  await setSourceImage({
    dataUrl,
    name: file.name,
    size: file.size,
    type: file.type,
  });
}

async function loadSampleImage() {
  const dataUrl = createSamplePortrait();
  await setSourceImage({
    dataUrl,
    name: "sample-portrait.png",
    size: Math.round((dataUrl.length * 3) / 4),
    type: "image/png",
  });
}

async function setSourceImage({ dataUrl, name, size, type }) {
  const image = await loadImage(dataUrl);
  const imageId = `img_${Date.now().toString(36)}`;

  state.imageId = imageId;
  state.sourceDataUrl = dataUrl;
  state.sourceName = name;
  state.sourceSize = size;
  state.sourceType = type;
  state.sourceImage = image;
  state.analysis = null;
  state.plans = [];
  state.selectedPlanId = null;
  state.resultDataUrl = null;
  state.activeBaseDataUrl = dataUrl;
  state.history = [`上传 ${name}`];

  els.sourceImage.src = dataUrl;
  els.sourceImage.hidden = false;
  els.sourceEmpty.hidden = true;
  els.compareBefore.src = dataUrl;

  renderMeta(image);
  resetAnalysis();
  resetResult();
  renderHistory();
  setStatus("已上传", "done");
  els.analyzeButton.disabled = false;
  els.analyzeButton.focus();
}

function renderMeta(image) {
  const values = [
    state.imageId,
    `${image.naturalWidth} × ${image.naturalHeight}`,
    formatBytes(state.sourceSize),
    normalizeType(state.sourceType),
  ];

  [...els.photoMeta.querySelectorAll("dd")].forEach((node, index) => {
    node.textContent = values[index] || "--";
  });
}

function resetAnalysis() {
  els.analysisEmpty.hidden = false;
  els.analysisEmpty.textContent = "待分析";
  els.analysisContent.hidden = true;
  els.planList.innerHTML = "";
  els.planCount.textContent = "0";
}

function resetResult() {
  els.resultEmpty.hidden = false;
  els.comparison.hidden = true;
  els.compareAfter.removeAttribute("src");
  els.executionNote.textContent = "--";
  els.regenerateButton.disabled = true;
  els.downloadButton.disabled = true;
  els.instructionInput.disabled = true;
  els.refineButton.disabled = true;
}

async function runAnalysis() {
  if (!state.sourceImage) return;

  setBusy(true, "分析照片");
  setStatus("分析中", "running");
  els.analysisEmpty.hidden = false;
  els.analysisEmpty.textContent = "分析中";

  await wait(650);

  const imageStats = sampleImageStats(state.sourceImage);
  state.analysis = createPhotoAnalysis(imageStats);
  state.plans = buildRetouchPlans(state.analysis);
  state.selectedPlanId = state.plans[0].planId;
  state.history.push("生成 3 个方案");

  renderAnalysis();
  renderPlans();
  renderHistory();

  setBusy(false);
  setStatus("方案就绪", "done");
}

function createPhotoAnalysis(stats) {
  const { brightness, contrast, warmth, width, height } = stats;
  const isPortrait = height >= width * 1.08;
  const isSquare = Math.abs(width - height) < Math.max(width, height) * 0.08;
  const isDark = brightness < 98;
  const isBright = brightness > 185;
  const isWarm = warmth > 14;
  const isCool = warmth < -10;
  const lowContrast = contrast < 42;
  const lowResolution = width < 900 || height < 900;

  const sceneType = isPortrait ? "人像 / 自拍" : isSquare ? "头像 / 社媒图" : "旅行 / 街拍";
  const lightingIssues = [
    isDark && "整体偏暗",
    isBright && "高光偏亮",
    isCool && "色温偏冷",
    isWarm && "色温偏暖",
    lowContrast && "层次略平",
  ].filter(Boolean);

  if (!lightingIssues.length) lightingIssues.push("光线基础良好");

  const backgroundIssues = [
    !isPortrait && "背景信息较多",
    lowContrast && "主体和背景分离度一般",
    isPortrait && "边缘干扰可进一步弱化",
  ].filter(Boolean);

  const portraitSuggestions = [
    isDark ? "提升面部亮度" : "保持自然肤色",
    "轻度统一肤色",
    "保留皮肤纹理",
    lowContrast ? "增强眼神光和面部层次" : "轻微增强五官清晰度",
  ];

  const compositionSuggestions = [
    isPortrait ? "适合保留竖版构图" : "可裁成头像或社媒封面",
    "主体位置保持稳定",
    "边缘杂物适度弱化",
  ];

  const recommendedStyles = [
    "自然",
    isPortrait ? "精致头像" : "旅行氛围",
    isWarm ? "电影感" : "清透社媒",
  ];

  return {
    imageId: state.imageId,
    sceneType,
    subjects: {
      count: 1,
      position: isPortrait || isSquare ? "中心区域" : "中间偏左/偏右需模型确认",
      faceVisibility: isPortrait ? "较高" : "中等",
    },
    lightingIssues,
    backgroundIssues: backgroundIssues.length ? backgroundIssues : ["背景干扰较少"],
    portraitSuggestions,
    compositionSuggestions,
    recommendedStyles,
    riskFlags: [
      lowResolution && "原图分辨率偏低",
      isBright && "过曝区域需保守处理",
      isDark && "暗部提亮可能带来噪点",
    ].filter(Boolean),
    stats,
  };
}

function buildRetouchPlans(analysis) {
  return planTemplates.map((template) => {
    const editPrompt = createEditPrompt(template, analysis);
    return {
      ...template,
      editPrompt,
      expectedChanges: [...template.expectedChanges],
    };
  });
}

function createEditPrompt(template, analysis) {
  const base = [
    `场景：${analysis.sceneType}`,
    `光线处理：${analysis.lightingIssues.join("、")}`,
    `背景处理：${analysis.backgroundIssues.join("、")}`,
    `人像要求：${analysis.portraitSuggestions.join("、")}`,
  ];

  if (template.planId === "natural") {
    base.push("目标：自然美化，轻微提亮，肤色均匀，不改变五官。");
  }

  if (template.planId === "avatar") {
    base.push("目标：头像构图，面部清晰，背景干净，主体真实。");
  }

  if (template.planId === "mood") {
    base.push("目标：胶片/社媒氛围，增强色彩情绪，保留本人感。");
  }

  return base.join("\n");
}

function renderAnalysis() {
  const analysis = state.analysis;
  els.analysisEmpty.hidden = true;
  els.analysisContent.hidden = false;
  els.sceneType.textContent = analysis.sceneType;
  els.subjectSummary.textContent = `${analysis.subjects.count} 人 · ${analysis.subjects.faceVisibility}`;
  els.riskSummary.textContent = analysis.riskFlags.length ? `${analysis.riskFlags.length} 项` : "低";

  const groups = [
    ["光线问题", analysis.lightingIssues, "amber"],
    ["背景问题", analysis.backgroundIssues, "blue"],
    ["人像优化", analysis.portraitSuggestions, ""],
    ["构图建议", analysis.compositionSuggestions, "blue"],
    ["风格建议", analysis.recommendedStyles, ""],
    ["风险标记", analysis.riskFlags.length ? analysis.riskFlags : ["暂无明显风险"], analysis.riskFlags.length ? "rose" : ""],
  ];

  els.analysisGroups.innerHTML = groups
    .map(
      ([title, items, tone]) => `
        <section class="analysis-group">
          <h3>${escapeHtml(title)}</h3>
          <div class="tag-row">
            ${items.map((item) => `<span class="tag ${tone}">${escapeHtml(item)}</span>`).join("")}
          </div>
        </section>
      `,
    )
    .join("");
}

function renderPlans() {
  els.planCount.textContent = String(state.plans.length);
  els.planList.innerHTML = state.plans
    .map((plan) => {
      const selected = plan.planId === state.selectedPlanId;
      return `
        <article class="plan-row ${selected ? "selected" : ""}" data-plan-id="${escapeHtml(plan.planId)}">
          <div>
            <div class="plan-title">
              <span>${escapeHtml(plan.title)}</span>
              <span class="plan-badge">${escapeHtml(plan.intensity)}</span>
            </div>
            <p>${escapeHtml(plan.description)}</p>
            <div class="tag-row" aria-label="预期变化">
              ${plan.expectedChanges.map((item) => `<span class="tag">${escapeHtml(item)}</span>`).join("")}
            </div>
          </div>
          <button class="small-action" type="button">${selected ? "生成" : "选择"}</button>
        </article>
      `;
    })
    .join("");

  els.planList.querySelectorAll(".plan-row").forEach((row) => {
    row.addEventListener("click", () => {
      const id = row.getAttribute("data-plan-id");
      if (state.selectedPlanId === id) {
        const plan = getSelectedPlan();
        startRetouchJob({ plan, instruction: "", baseDataUrl: state.sourceDataUrl });
        return;
      }
      state.selectedPlanId = id;
      renderPlans();
    });
  });
}

async function startRetouchJob({ plan, instruction, baseDataUrl }) {
  if (!plan || !baseDataUrl) return;

  setBusy(true, instruction ? "继续修改" : "生成结果");
  setStatus("生成中", "running");
  els.jobText.textContent = instruction ? "修改中" : "生成中";
  els.jobOverlay.hidden = false;

  await wait(900);

  const recipe = createRecipe(plan, instruction);
  const output = await renderRetouchImage(baseDataUrl, recipe);

  state.resultDataUrl = output;
  state.activeBaseDataUrl = output;
  state.history.push(instruction ? `修改：${instruction}` : `生成：${plan.title}`);

  els.compareBefore.src = state.sourceDataUrl;
  els.compareAfter.src = output;
  els.resultEmpty.hidden = true;
  els.comparison.hidden = false;
  els.executionNote.textContent = createExecutionNote(plan, instruction, recipe);
  els.regenerateButton.disabled = false;
  els.downloadButton.disabled = false;
  els.instructionInput.disabled = false;
  els.refineButton.disabled = false;
  els.instructionInput.value = "";

  syncStageWidth();
  updateComparison();
  renderHistory();
  setBusy(false);
  els.jobOverlay.hidden = true;
  setStatus("结果完成", "done");
}

function createRecipe(plan, instruction = "") {
  const text = instruction.trim();
  const wantsBrighter = /亮|提亮|白|清透/.test(text);
  const wantsNatural = /自然|轻|别太|不要太|少一点/.test(text);
  const wantsAvatar = /头像|裁|方图|简历|LinkedIn/i.test(text);
  const wantsMood = /氛围|胶片|电影|小红书|风格/.test(text);
  const wantsBlur = /背景.*虚|虚化|干净|路人|杂物|清理/.test(text);
  const reduceBlur = /别.*虚|不要.*虚|背景.*清楚/.test(text);

  const recipe = {
    id: plan.planId,
    crop: plan.planId === "avatar",
    brightness: 1,
    contrast: 1,
    saturation: 1,
    sepia: 0,
    warmth: 0,
    vignette: 0.08,
    grain: 0,
    backdropBlur: 0,
  };

  if (plan.planId === "natural") {
    Object.assign(recipe, {
      brightness: 1.07,
      contrast: 1.04,
      saturation: 1.06,
      warmth: 0.05,
      vignette: 0.04,
    });
  }

  if (plan.planId === "avatar") {
    Object.assign(recipe, {
      crop: true,
      brightness: 1.1,
      contrast: 1.08,
      saturation: 1.03,
      warmth: 0.03,
      vignette: 0.03,
      backdropBlur: 14,
    });
  }

  if (plan.planId === "mood") {
    Object.assign(recipe, {
      brightness: 1.01,
      contrast: 1.12,
      saturation: 1.18,
      sepia: 0.12,
      warmth: 0.1,
      vignette: 0.16,
      grain: 0.12,
    });
  }

  if (wantsBrighter) recipe.brightness += wantsNatural ? 0.04 : 0.1;
  if (wantsNatural) {
    recipe.contrast = Math.min(recipe.contrast, 1.06);
    recipe.saturation = Math.min(recipe.saturation, 1.08);
    recipe.sepia = Math.min(recipe.sepia, 0.05);
    recipe.vignette = Math.min(recipe.vignette, 0.05);
    recipe.grain = Math.min(recipe.grain, 0.04);
  }
  if (wantsAvatar) recipe.crop = true;
  if (wantsMood) {
    recipe.contrast += 0.04;
    recipe.saturation += 0.06;
    recipe.grain = Math.max(recipe.grain, 0.08);
  }
  if (wantsBlur) recipe.backdropBlur = Math.max(recipe.backdropBlur, 12);
  if (reduceBlur) recipe.backdropBlur = Math.min(recipe.backdropBlur, 4);

  recipe.brightness = clamp(recipe.brightness, 0.82, 1.28);
  recipe.contrast = clamp(recipe.contrast, 0.85, 1.28);
  recipe.saturation = clamp(recipe.saturation, 0.85, 1.35);
  return recipe;
}

async function renderRetouchImage(dataUrl, recipe) {
  const image = await loadImage(dataUrl);
  const sourceWidth = image.naturalWidth;
  const sourceHeight = image.naturalHeight;
  const maxSide = recipe.crop ? 1200 : 1500;
  const outputWidth = recipe.crop ? maxSide : Math.round(sourceWidth * Math.min(1, maxSide / Math.max(sourceWidth, sourceHeight)));
  const outputHeight = recipe.crop ? maxSide : Math.round(sourceHeight * Math.min(1, maxSide / Math.max(sourceWidth, sourceHeight)));

  const canvas = document.createElement("canvas");
  canvas.width = outputWidth;
  canvas.height = outputHeight;
  const ctx = canvas.getContext("2d");

  if (recipe.crop || recipe.backdropBlur > 0) {
    ctx.save();
    ctx.filter = `blur(${recipe.backdropBlur || 10}px) brightness(${Math.max(0.95, recipe.brightness - 0.03)}) saturate(${Math.max(0.9, recipe.saturation - 0.08)})`;
    drawCover(ctx, image, 0, 0, outputWidth, outputHeight);
    ctx.restore();
  }

  const filter = [
    `brightness(${recipe.brightness})`,
    `contrast(${recipe.contrast})`,
    `saturate(${recipe.saturation})`,
    recipe.sepia ? `sepia(${recipe.sepia})` : "",
  ]
    .filter(Boolean)
    .join(" ");

  ctx.save();
  ctx.filter = filter;
  if (recipe.crop) {
    drawCover(ctx, image, outputWidth * 0.08, outputHeight * 0.04, outputWidth * 0.84, outputHeight * 0.9);
  } else {
    ctx.drawImage(image, 0, 0, outputWidth, outputHeight);
  }
  ctx.restore();

  applyWarmth(ctx, outputWidth, outputHeight, recipe.warmth);
  applyFaceLight(ctx, outputWidth, outputHeight, recipe.crop ? 0.24 : 0.16);
  applyVignette(ctx, outputWidth, outputHeight, recipe.vignette);
  if (recipe.grain) applyGrain(ctx, outputWidth, outputHeight, recipe.grain);

  return canvas.toDataURL("image/jpeg", 0.92);
}

function drawCover(ctx, image, dx, dy, dw, dh) {
  const imageRatio = image.naturalWidth / image.naturalHeight;
  const targetRatio = dw / dh;
  let sx = 0;
  let sy = 0;
  let sw = image.naturalWidth;
  let sh = image.naturalHeight;

  if (imageRatio > targetRatio) {
    sw = image.naturalHeight * targetRatio;
    sx = (image.naturalWidth - sw) / 2;
  } else {
    sh = image.naturalWidth / targetRatio;
    sy = (image.naturalHeight - sh) / 2;
  }

  ctx.drawImage(image, sx, sy, sw, sh, dx, dy, dw, dh);
}

function applyWarmth(ctx, width, height, amount) {
  if (!amount) return;
  ctx.save();
  ctx.globalCompositeOperation = "soft-light";
  ctx.globalAlpha = clamp(amount, 0, 0.18);
  ctx.fillStyle = "#ffd4a5";
  ctx.fillRect(0, 0, width, height);
  ctx.restore();
}

function applyFaceLight(ctx, width, height, amount) {
  ctx.save();
  const gradient = ctx.createRadialGradient(width * 0.5, height * 0.38, width * 0.08, width * 0.5, height * 0.38, width * 0.45);
  gradient.addColorStop(0, `rgba(255,255,255,${amount})`);
  gradient.addColorStop(0.55, `rgba(255,255,255,${amount * 0.18})`);
  gradient.addColorStop(1, "rgba(255,255,255,0)");
  ctx.globalCompositeOperation = "screen";
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, width, height);
  ctx.restore();
}

function applyVignette(ctx, width, height, amount) {
  if (!amount) return;
  ctx.save();
  const gradient = ctx.createRadialGradient(width * 0.5, height * 0.44, width * 0.2, width * 0.5, height * 0.5, width * 0.72);
  gradient.addColorStop(0, "rgba(0,0,0,0)");
  gradient.addColorStop(1, `rgba(0,0,0,${amount})`);
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, width, height);
  ctx.restore();
}

function applyGrain(ctx, width, height, amount) {
  const grainCanvas = document.createElement("canvas");
  const scale = 0.35;
  grainCanvas.width = Math.max(1, Math.floor(width * scale));
  grainCanvas.height = Math.max(1, Math.floor(height * scale));
  const grainCtx = grainCanvas.getContext("2d");
  const imageData = grainCtx.createImageData(grainCanvas.width, grainCanvas.height);

  for (let index = 0; index < imageData.data.length; index += 4) {
    const shade = 118 + Math.random() * 42;
    imageData.data[index] = shade;
    imageData.data[index + 1] = shade;
    imageData.data[index + 2] = shade;
    imageData.data[index + 3] = 255 * amount;
  }

  grainCtx.putImageData(imageData, 0, 0);
  ctx.save();
  ctx.globalCompositeOperation = "overlay";
  ctx.globalAlpha = 0.28;
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(grainCanvas, 0, 0, width, height);
  ctx.restore();
}

function sampleImageStats(image) {
  const canvas = document.createElement("canvas");
  const size = 72;
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(image, 0, 0, size, size);
  const data = ctx.getImageData(0, 0, size, size).data;
  let total = 0;
  let totalSquared = 0;
  let warmth = 0;

  for (let i = 0; i < data.length; i += 4) {
    const r = data[i];
    const g = data[i + 1];
    const b = data[i + 2];
    const luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    total += luma;
    totalSquared += luma * luma;
    warmth += r - b;
  }

  const count = data.length / 4;
  const mean = total / count;
  const variance = totalSquared / count - mean * mean;
  return {
    width: image.naturalWidth,
    height: image.naturalHeight,
    brightness: mean,
    contrast: Math.sqrt(Math.max(0, variance)),
    warmth: warmth / count,
  };
}

function handleRefine(event) {
  event.preventDefault();
  const instruction = els.instructionInput.value.trim();
  if (!instruction || !state.resultDataUrl) return;

  const plan = getSelectedPlan() || state.plans[0];
  startRetouchJob({
    plan,
    instruction,
    baseDataUrl: state.activeBaseDataUrl || state.resultDataUrl,
  });
}

function createExecutionNote(plan, instruction, recipe) {
  const changes = [
    recipe.crop && "头像构图",
    recipe.brightness > 1.05 && "提亮",
    recipe.contrast > 1.06 && "增强层次",
    recipe.saturation > 1.1 && "强化色彩",
    recipe.backdropBlur > 0 && "背景弱化",
    recipe.grain > 0 && "轻颗粒",
  ].filter(Boolean);

  const source = instruction ? `二次修改：${instruction}` : `方案：${plan.title}`;
  return `${source}。执行项：${changes.length ? changes.join("、") : "自然优化"}。负向约束：${plan.negativePrompt}。`;
}

function getSelectedPlan() {
  return state.plans.find((plan) => plan.planId === state.selectedPlanId);
}

function renderHistory() {
  if (!state.history.length) {
    els.historyStrip.innerHTML = `<span class="history-item muted">等待任务</span>`;
    return;
  }

  els.historyStrip.innerHTML = state.history
    .slice(-5)
    .map((item) => `<span class="history-item">${escapeHtml(item)}</span>`)
    .join("");
  els.historyStrip.scrollLeft = els.historyStrip.scrollWidth;
}

function updateComparison() {
  const value = Number(els.compareSlider.value);
  els.afterLayer.style.width = `${value}%`;
  els.splitLine.style.left = `${value}%`;
}

function syncStageWidth() {
  const width = els.resultStage.clientWidth || 1;
  els.afterLayer.style.setProperty("--stage-width", `${width}px`);
}

function downloadResult() {
  if (!state.resultDataUrl) return;

  const link = document.createElement("a");
  link.href = state.resultDataUrl;
  link.download = `photo-retouch-${Date.now()}.jpg`;
  document.body.appendChild(link);
  link.click();
  link.remove();
}

function setBusy(isBusy, label = "") {
  els.analyzeButton.disabled = isBusy || !state.sourceDataUrl;
  els.uploadButton.disabled = isBusy;
  els.sampleButton.disabled = isBusy;
  els.regenerateButton.disabled = isBusy || !state.resultDataUrl;
  els.refineButton.disabled = isBusy || !state.resultDataUrl;
  els.instructionInput.disabled = isBusy || !state.resultDataUrl;
  els.downloadButton.disabled = isBusy || !state.resultDataUrl;
  if (isBusy && label) els.jobText.textContent = label;
}

function showError(message) {
  setStatus(message, "error");
  els.analysisEmpty.hidden = false;
  els.analysisEmpty.textContent = message;
}

function readFileAsDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(file);
  });
}

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("图片加载失败"));
    image.src = src;
  });
}

function createSamplePortrait() {
  const canvas = document.createElement("canvas");
  canvas.width = 1200;
  canvas.height = 1500;
  const ctx = canvas.getContext("2d");

  const bg = ctx.createLinearGradient(0, 0, canvas.width, canvas.height);
  bg.addColorStop(0, "#d8e4df");
  bg.addColorStop(0.5, "#f4dfca");
  bg.addColorStop(1, "#8fb0ba");
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  ctx.fillStyle = "rgba(255,255,255,0.58)";
  ctx.fillRect(86, 120, 310, 1180);
  ctx.fillStyle = "rgba(20,42,50,0.14)";
  ctx.fillRect(440, 150, 590, 920);
  ctx.fillStyle = "rgba(178,86,55,0.32)";
  ctx.fillRect(900, 880, 170, 240);

  ctx.save();
  ctx.translate(600, 760);
  ctx.fillStyle = "#2c2927";
  roundedRect(ctx, -230, -380, 460, 570, 210);
  ctx.fill();

  ctx.fillStyle = "#e3b08d";
  roundedRect(ctx, -158, -250, 316, 410, 156);
  ctx.fill();

  ctx.fillStyle = "#1f1e1d";
  roundedRect(ctx, -198, -316, 396, 230, 170);
  ctx.fill();

  ctx.fillStyle = "#d69d7d";
  roundedRect(ctx, -64, 116, 128, 142, 58);
  ctx.fill();

  ctx.fillStyle = "#2c6b68";
  roundedRect(ctx, -280, 222, 560, 410, 62);
  ctx.fill();

  ctx.fillStyle = "rgba(255,255,255,0.82)";
  roundedRect(ctx, -114, 248, 228, 180, 42);
  ctx.fill();

  ctx.fillStyle = "#3a2a25";
  ctx.beginPath();
  ctx.ellipse(-72, -64, 24, 14, 0, 0, Math.PI * 2);
  ctx.ellipse(76, -64, 24, 14, 0, 0, Math.PI * 2);
  ctx.fill();

  ctx.strokeStyle = "#8b4c43";
  ctx.lineWidth = 8;
  ctx.lineCap = "round";
  ctx.beginPath();
  ctx.moveTo(-58, 58);
  ctx.quadraticCurveTo(0, 96, 68, 54);
  ctx.stroke();

  ctx.fillStyle = "rgba(255,230,211,0.45)";
  ctx.beginPath();
  ctx.ellipse(-110, 12, 38, 24, 0, 0, Math.PI * 2);
  ctx.ellipse(112, 12, 38, 24, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();

  ctx.fillStyle = "rgba(255,255,255,0.24)";
  ctx.beginPath();
  ctx.arc(185, 260, 42, 0, Math.PI * 2);
  ctx.arc(1010, 330, 36, 0, Math.PI * 2);
  ctx.arc(1035, 1130, 26, 0, Math.PI * 2);
  ctx.fill();

  return canvas.toDataURL("image/png");
}

function roundedRect(ctx, x, y, width, height, radius) {
  const r = Math.min(radius, width / 2, height / 2);
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + width - r, y);
  ctx.quadraticCurveTo(x + width, y, x + width, y + r);
  ctx.lineTo(x + width, y + height - r);
  ctx.quadraticCurveTo(x + width, y + height, x + width - r, y + height);
  ctx.lineTo(x + r, y + height);
  ctx.quadraticCurveTo(x, y + height, x, y + height - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
}

function formatBytes(bytes) {
  if (!bytes) return "--";
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function normalizeType(type) {
  if (type === "image/jpeg") return "JPG";
  if (type === "image/png") return "PNG";
  if (type === "image/webp") return "WebP";
  return type || "--";
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function wait(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}
