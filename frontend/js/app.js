import { state, on, categories, allTools, toolMap } from "./state.js";

import {
  analyzePhoto,
  assetUrl,
  createJob,
  createPlans,
  createStylingProposals,
  getJob,
  getProviderCapabilities,
  getStylingProviders,
  listJobs,
  refineJob,
  uploadPhoto,
} from "./api.js";

import {
  designStyling,
  estimateStylingCost,
  executeAiRetouch,
  executeStyling,
  normalizeStylingProposals,
  refineStyling,
  validateModelSelection,
} from "./ai-retouch.mjs";

import {
  buildExportFilename,
  canvasToBlob,
  triggerBlobDownload,
} from "./export-image.mjs";

import { renderLocalRetouch } from "./local-retouch.mjs";


const elements = {
  categoryTitle: document.querySelector("#categoryTitle"),
  quickTags: document.querySelector("#quickTags"),
  adjustmentList: document.querySelector("#adjustmentList"),
  toolSearch: document.querySelector("#toolSearch"),
  localPanel: document.querySelector("#localPanel"),
  aiPanel: document.querySelector("#aiPanel"),
  modeNote: document.querySelector("#modeNote"),
  previewCanvas: document.querySelector("#previewCanvas"),
  beforeImage: document.querySelector("#beforeImage"),
  afterLayer: document.querySelector("#afterLayer"),
  compareLine: document.querySelector("#compareLine"),
  compareRange: document.querySelector("#compareRange"),
  compareButton: document.querySelector("#compareButton"),
  changeList: document.querySelector("#changeList"),
  changeCount: document.querySelector("#changeCount"),
  historyState: document.querySelector("#historyState"),
  footerStatus: document.querySelector("#footerStatus"),
  canvasStatus: document.querySelector("#canvasStatus"),
  fileInput: document.querySelector("#fileInput"),
  uploadButton: document.querySelector("#uploadButton"),
  exportButton: document.querySelector("#exportButton"),
  resetButton: document.querySelector("#resetButton"),
  resetCategoryButton: document.querySelector("#resetCategoryButton"),
  undoButton: document.querySelector("#undoButton"),
  aiPrompt: document.querySelector("#aiPrompt"),
  promptCount: document.querySelector("#promptCount"),
  generateButton: document.querySelector("#generateButton"),
  floatingTip: document.querySelector("#floatingTip"),
  aiFeedback: document.querySelector("#aiFeedback"),
  modelConfigState: document.querySelector("#modelConfigState"),
  brainProviderSelect: document.querySelector("#brainProviderSelect"),
  actionProviderSelect: document.querySelector("#actionProviderSelect"),
  brainApiKeyField: document.querySelector("#brainApiKeyField"),
  actionApiKeyField: document.querySelector("#actionApiKeyField"),
  brainApiKeyInput: document.querySelector("#brainApiKeyInput"),
  actionApiKeyInput: document.querySelector("#actionApiKeyInput"),
  brainWorkspaceField: document.querySelector("#brainWorkspaceField"),
  actionWorkspaceField: document.querySelector("#actionWorkspaceField"),
  brainWorkspaceInput: document.querySelector("#brainWorkspaceInput"),
  actionWorkspaceInput: document.querySelector("#actionWorkspaceInput"),
  documentName: document.querySelector("#documentName"),
  documentMeta: document.querySelector("#documentMeta"),
  imageWorkbench: document.querySelector("#imageWorkbench"),
  workbenchEmpty: document.querySelector("#workbenchEmpty"),
  portraitName: document.querySelector("#portraitName"),
  portraitMeta: document.querySelector("#portraitMeta"),
  detectionLabel: document.querySelector("#detectionLabel"),
  detectionFaceCount: document.querySelector("#detectionFaceCount"),
  detectionVisibility: document.querySelector("#detectionVisibility"),
  detectionLight: document.querySelector("#detectionLight"),
  stylingSource: document.querySelector("#stylingSource"),
  stylingDesignButton: document.querySelector("#stylingDesignButton"),
  stylingTemplateButton: document.querySelector("#stylingTemplateButton"),
  stylingCostNote: document.querySelector("#stylingCostNote"),
  stylingGrid: document.querySelector("#stylingGrid"),
  stylingRefine: document.querySelector("#stylingRefine"),
  stylingInstruction: document.querySelector("#stylingInstruction"),
  stylingRefineButton: document.querySelector("#stylingRefineButton"),
  stylingFeedback: document.querySelector("#stylingFeedback"),
  stylingGeneration: document.querySelector("#stylingGeneration"),
  generationSteps: document.querySelector("#generationSteps"),
  generationStageNote: document.querySelector("#generationStageNote"),
  stylingResult: document.querySelector("#stylingResult"),
  resultBeforeImage: document.querySelector("#resultBeforeImage"),
  resultAfterImage: document.querySelector("#resultAfterImage"),
  resultAfterCaption: document.querySelector("#resultAfterCaption"),
  resultVerification: document.querySelector("#resultVerification"),
  resultNote: document.querySelector("#resultNote"),
  resultDownload: document.querySelector("#resultDownload"),
  resultReselect: document.querySelector("#resultReselect"),
  historyList: document.querySelector("#historyList"),
  historyRefresh: document.querySelector("#historyRefresh"),
  costDialog: document.querySelector("#costDialog"),
  costDialogLead: document.querySelector("#costDialogLead"),
  costDesignLabel: document.querySelector("#costDesignLabel"),
  costDesignValue: document.querySelector("#costDesignValue"),
  costImageLabel: document.querySelector("#costImageLabel"),
  costImageValue: document.querySelector("#costImageValue"),
  costJobCount: document.querySelector("#costJobCount"),
  costDialogNote: document.querySelector("#costDialogNote"),
  costCancelButton: document.querySelector("#costCancelButton"),
  costConfirmButton: document.querySelector("#costConfirmButton"),
};


const aiApi = {
  uploadPhoto,
  analyzePhoto,
  createPlans,
  createJob,
  createStylingProposals,
  getJob,
  refineJob,
};


function providerIsReady(provider) {
  return Boolean(provider) && (provider.configured || !provider.requiresApiKey);
}


// Choose the provider the app starts on. The ordered ``candidates`` list encodes
// the product's preference (a real, server-configured model before the free
// local fallback); anything not ready to run is skipped so a first-time visitor
// never lands on a provider the server has no key for.
function pickDefaultProviderId(providers, candidates) {
  const ordered = Array.isArray(candidates) ? candidates : [candidates].filter(Boolean);
  const byId = (id) => providers.find((provider) => provider.id === id);
  for (const id of ordered) {
    const hit = byId(id);
    if (hit && providerIsReady(hit)) return hit.id;
  }
  const ready = providers.find(providerIsReady);
  if (ready) return ready.id;
  const anyCandidate = ordered.map(byId).find(Boolean);
  return (anyCandidate || providers[0] || {}).id || "";
}


function populateProviderSelect(select, providers, candidates) {
  const previous = select.value;
  // The first population happens on page load, when ``previous`` is still the
  // static <option> baked into the HTML — not a user choice. Only preserve the
  // value once the list has genuinely been populated before (a later re-sync),
  // so a first-time visitor lands on a provider the server can actually run.
  const alreadyPopulated = select.dataset.populated === "true";
  select.replaceChildren(
    ...providers.map((provider) => {
      const option = document.createElement("option");
      option.value = provider.id;
      option.textContent = `${provider.label} · ${provider.model}`;
      return option;
    }),
  );
  const availableIds = new Set(providers.map((p) => p.id));
  select.dataset.populated = "true";
  select.value = alreadyPopulated && availableIds.has(previous)
    ? previous
    : pickDefaultProviderId(providers, candidates) || providers[0]?.id || "";
}


function selectedProvider(layer) {
  const isBrain = layer === "brain";
  const providers = isBrain ? state.capabilities.brainProviders : state.capabilities.actionProviders;
  const select = isBrain ? elements.brainProviderSelect : elements.actionProviderSelect;
  return providers.find((p) => p.id === select.value);
}


function modelRequest(layer) {
  const isBrain = layer === "brain";
  const providerName = isBrain ? elements.brainProviderSelect.value : elements.actionProviderSelect.value;
  const directApiKey = (isBrain ? elements.brainApiKeyInput.value : elements.actionApiKeyInput.value).trim();
  const brainProvider = elements.brainProviderSelect.value;
  const sharedKeyActions = {
    qwen: ["qwen", "wan"],
    doubao: ["seedream"],
    siliconflow: ["siliconflow"],
  };
  const canReuseBrainKey =
    !isBrain && (sharedKeyActions[brainProvider] || []).includes(providerName);
  return {
    name: providerName,
    apiKey: directApiKey || (canReuseBrainKey ? elements.brainApiKeyInput.value.trim() : ""),
    workspaceId: (isBrain ? elements.brainWorkspaceInput.value : elements.actionWorkspaceInput.value).trim(),
  };
}


function syncModelConfig() {
  const brain = selectedProvider("brain");
  const action = selectedProvider("action");
  elements.brainApiKeyField.hidden = !brain?.requiresApiKey;
  elements.actionApiKeyField.hidden = !action?.requiresApiKey;
  elements.brainWorkspaceField.hidden = !brain?.workspaceSupported;
  elements.actionWorkspaceField.hidden = !action?.workspaceSupported;
  elements.brainApiKeyInput.placeholder = brain?.configured ? "服务端已配置，可留空" : "请输入本次会话 Key";
  elements.actionApiKeyInput.placeholder = action?.configured ? "服务端已配置，可留空" : "可留空复用同平台视觉 Key";

  const br = modelRequest("brain");
  const ar = modelRequest("action");
  const ready = brain && action &&
    (!brain.requiresApiKey || brain.configured || br.apiKey) &&
    (!action.requiresApiKey || action.configured || ar.apiKey);
  elements.modelConfigState.textContent = ready ? "已就绪" : "需要配置";
  elements.modelConfigState.dataset.state = ready ? "ready" : "error";
}


async function loadProviderCapabilities() {
  try {
    state.capabilities = await getProviderCapabilities();
    const brains = state.capabilities.brainProviders || [];
    const actions = state.capabilities.actionProviders || [];
    // Prefer a real, configured vision model to author the AI designs, but
    // default the *image* provider to the free local simulation: a first-time
    // visitor must be able to finish upload → design → generate → download
    // without a paid request, and RunningHub CropAndStitch only understands
    // styling plans (a skin-retouch plan is refused at preflight). A real image
    // provider is one explicit selection — and one cost confirmation — away.
    const brainOrder = [
      ...brains.filter((p) => p.id !== "local").map((p) => p.id),
      ...brains.filter((p) => p.id === "local").map((p) => p.id),
    ];
    const freeActions = new Set(["mock", "local"]);
    const actionOrder = [
      ...actions.filter((p) => freeActions.has(p.id)).map((p) => p.id),
      ...actions.filter((p) => !freeActions.has(p.id)).map((p) => p.id),
    ];
    populateProviderSelect(elements.brainProviderSelect, brains, brainOrder);
    populateProviderSelect(elements.actionProviderSelect, actions, actionOrder);
    syncModelConfig();
  } catch (error) {
    elements.modelConfigState.textContent = "服务不可用";
    elements.modelConfigState.dataset.state = "error";
    elements.aiFeedback.textContent = error.message;
    elements.aiFeedback.className = "ai-feedback error";
  }
}


async function currentSourceFile() {
  if (state.sourceFile) return state.sourceFile;
  const src = elements.beforeImage.currentSrc || elements.beforeImage.src;
  if (!src) throw new Error("请先选择一张照片。");
  const response = await fetch(src);
  if (!response.ok) throw new Error("无法读取当前照片，请重新选择图片。");
  const blob = await response.blob();
  const extension = blob.type === "image/jpeg" ? "jpg" : "png";
  state.sourceFile = new File([blob], `portrait.${extension}`, {
    type: blob.type || "image/png",
  });
  return state.sourceFile;
}


function drawImageToPreview(image) {
  const width = image.naturalWidth || image.width;
  const height = image.naturalHeight || image.height;
  elements.previewCanvas.width = width;
  elements.previewCanvas.height = height;
  const ctx = elements.previewCanvas.getContext("2d", { alpha: false });
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(image, 0, 0, width, height);
}


function showAiResult(url) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.addEventListener("load", () => {
      state.aiResultImage = img;
      state.aiResultUrl = url;
      state.displayingAiResult = true;
      drawImageToPreview(img);
      resolve();
    });
    img.addEventListener("error", () => reject(new Error("AI 结果图片加载失败。")));
    img.src = url;
  });
}


async function exportCurrentImage(event) {
  event.preventDefault();
  const isAi = state.mode === "ai" && state.displayingAiResult && state.aiResultUrl;
  elements.footerStatus.textContent = "正在准备导出图片";
  try {
    let blob;
    if (isAi) {
      const response = await fetch(state.aiResultUrl);
      if (!response.ok) throw new Error("无法读取 AI 结果图片。");
      blob = await response.blob();
    } else {
      blob = await canvasToBlob(elements.previewCanvas, "image/png");
    }
    const fileName = buildExportFilename(state.sourceName || "portrait", isAi ? "ai" : "local", blob.type);
    triggerBlobDownload(blob, fileName);
    elements.footerStatus.textContent = `已导出 ${fileName}`;
  } catch (error) {
    elements.footerStatus.textContent = error.message;
  }
}


function renderCategory() {
  const cat = categories[state.category];
  elements.categoryTitle.textContent = cat.title;
  const tools = [...cat.tools];

  elements.quickTags.innerHTML = tools
    .map(
      (tool) => `
        <button class="tool-tag ${state.values[tool.id] > 0 ? "active" : ""}"
                data-tool-tag="${tool.id}" type="button">${tool.label}</button>`,
    ).join("");

  elements.adjustmentList.innerHTML = tools
    .map((tool) => {
      const v = state.values[tool.id];
      return `
        <label class="adjustment">
          <span class="adjustment-head">
            <span class="adjustment-label"><i>${tool.short}</i><span>${tool.label}</span></span>
            <b class="adjustment-value" data-value-for="${tool.id}">${v}</b>
          </span>
          <span class="slider-row">
            <span>−</span>
            <input type="range" min="0" max="100" value="${v}" data-tool-range="${tool.id}"
                   style="--range-progress: ${v}%" />
            <span>＋</span>
          </span>
        </label>`;
    }).join("");

  elements.quickTags.querySelectorAll("[data-tool-tag]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const id = btn.dataset.toolTag;
      const prev = state.values[id];
      state.history = [...state.history, { id, value: prev }];
      state.values = { ...state.values, [id]: prev > 0 ? 0 : 20 };
      renderCategory();
      updatePreview();
    });
  });

  elements.adjustmentList.querySelectorAll("[data-tool-range]").forEach((input) => {
    input.addEventListener("pointerdown", () => {
      input.dataset.startValue = String(state.values[input.dataset.toolRange]);
    });
    input.addEventListener("input", () => {
      const id = input.dataset.toolRange;
      state.values = { ...state.values, [id]: Number(input.value) };
      input.style.setProperty("--range-progress", `${input.value}%`);
      const valEl = elements.adjustmentList.querySelector(`[data-value-for="${id}"]`);
      if (valEl) valEl.textContent = input.value;
      const tag = elements.quickTags.querySelector(`[data-tool-tag="${id}"]`);
      if (tag) tag.classList.toggle("active", Number(input.value) > 0);
      updatePreview();
    });
    input.addEventListener("change", () => {
      const start = Number(input.dataset.startValue ?? 0);
      if (start !== Number(input.value)) {
        state.history = [...state.history, { id: input.dataset.toolRange, value: start }];
      }
    });
  });
}


let renderFrame = 0;

function scheduleLocalRender() {
  if (!elements.beforeImage.complete || !elements.beforeImage.naturalWidth) return;
  if (state.mode === "ai" && state.displayingAiResult) return;
  window.cancelAnimationFrame(renderFrame);
  renderFrame = window.requestAnimationFrame(() => {
    renderLocalRetouch(elements.previewCanvas, elements.beforeImage, state.values);
  });
}


function updatePreview() {
  scheduleLocalRender();
  renderChanges();
}


function renderChanges() {
  const changes = Object.entries(state.values)
    .filter(([, v]) => v > 0)
    .sort((a, b) => b[1] - a[1]);
  elements.changeCount.textContent = `${changes.length} 项`;
  elements.changeList.innerHTML = changes.length
    ? changes.slice(0, 5).map(([id, v]) => {
        const t = toolMap[id];
        return `<div class="change-item"><span>${t.short}</span><div><strong>${t.label}</strong><small>${t.hint}</small></div><b>+${v}</b></div>`;
      }).join("")
    : `<div class="change-item"><span>原</span><div><strong>尚未调整</strong><small>选择左侧标签开始美化</small></div><b>0</b></div>`;
  elements.historyState.textContent = changes.length
    ? changes.slice(0, 4).map(([id, v]) => `${toolMap[id].label} ${v}`).join(" · ")
    : "尚未添加美化参数";
}


function setMode(mode) {
  state.mode = mode;
  document.querySelectorAll("[data-mode]").forEach((b) => b.classList.toggle("active", b.dataset.mode === mode));
  const isLocal = mode === "local";
  elements.localPanel.hidden = !isLocal;
  elements.aiPanel.hidden = isLocal;
  elements.modeNote.innerHTML = isLocal
    ? '<span class="local-dot"></span>本地实时预览 · 图片无需上传模型'
    : '<span class="local-dot"></span>AI 造型设计 · 生成前会确认费用';
  elements.canvasStatus.classList.toggle("ai", !isLocal);
  elements.canvasStatus.innerHTML = isLocal
    ? "<span></span>实时预览已开启"
    : state.displayingAiResult
      ? "<span></span>AI 美化结果"
      : state.backendPhoto
        ? "<span></span>AI 造型设计 · 照片已就绪，可设计三套造型"
        : "<span></span>AI 造型设计 · 请先上传照片";
  elements.footerStatus.textContent = isLocal ? "实时美化已就绪" : "AI 造型设计已就绪";
  if (isLocal) {
    state.displayingAiResult = false;
    scheduleLocalRender();
  } else if (state.aiResultImage) {
    state.displayingAiResult = true;
    drawImageToPreview(state.aiResultImage);
  }
}


function resetTools(tools) {
  const snapshot = {};
  tools.forEach((t) => { snapshot[t.id] = state.values[t.id]; });
  const newValues = { ...state.values };
  tools.forEach((t) => { newValues[t.id] = 0; });
  state.values = newValues;
  state.history = [...state.history, { snapshot }];
  renderCategory();
  updatePreview();
}


function applyPreset(preset) {
  const presets = {
    natural: { smooth: 24, skinTone: 16, whiten: 6, sculpt: 8, eyeBags: 10 },
    clear: { smooth: 18, skinTone: 20, whiten: 18, brightEyes: 14, clarity: 12 },
    sculpt: { smooth: 12, sculpt: 28, jawline: 14, noseBridge: 12, contrast: 10 },
    warm: { smooth: 20, skinTone: 24, blush: 14, warmth: 18, lipColor: 9 },
  };
  state.values = { ...state.values, ...presets[preset] };
  updatePreview();
}


function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}


function currentStylingProposals() {
  return Array.isArray(state.stylingProposals) ? state.stylingProposals : [];
}


function selectedStylingProposal() {
  return currentStylingProposals().find((look) => look.planId === state.selectedStylingId) || null;
}


function selectStylingProposal(planId) {
  state.selectedStylingId = planId;
  elements.stylingGrid.querySelectorAll("[data-styling-card]").forEach((card) => {
    card.classList.toggle("active", card.dataset.stylingCard === planId);
  });
  updateStylingCostNote();
}


function analysisReused() {
  return state.stylingAnalysis?.analysisReused === true;
}


function updateStylingCostNote() {
  const looks = currentStylingProposals();
  if (!looks.length) return;
  const mode = stylingDesignMode();
  const cost = estimateStylingCost(looks, {
    selectedCount: 1,
    mode,
    analysisReused: analysisReused(),
    brainIsPaid: Boolean(selectedProvider("brain")?.requiresApiKey),
  });
  const selected = selectedStylingProposal();
  const vision = mode === "ai" ? ` ${cost.visionNote}` : "";
  elements.stylingCostNote.textContent = selected
    ? `已选「${selected.name || selected.planId}」：${cost.note}${vision} 未点击生成前不会产生任何请求。`
    : `${cost.note}${vision}`;
}


function stylingDesignMode() {
  return currentStylingProposals()[0]?.designMode === "ai" ? "ai" : "template";
}


// The paid "AI 设计三套造型" flow spends real model calls, so it is the only
// path that asks for an explicit confirmation before submitting anything. The
// free template flow never reaches this guard.
function confirmPaidStylingDesign() {
  const brain = selectedProvider("brain");
  if (brain && !brain.requiresApiKey) return true;
  const reused = analysisReused();
  const calls = reused ? 2 : 3;
  const label = brain?.label || brain?.id || "付费视觉模型";
  const approved = window.confirm(
    `「AI 设计三套造型」会调用付费模型「${label}」，完整流程最多 ${calls} 次付费请求` +
      (reused
        ? "（已有可复用的人物分析缓存，本次为 1 次造型设计 + 1 次出图提示词规划）"
        : "（1 次人物事实分析 + 1 次造型设计 + 1 次出图提示词规划）") +
      "。\n无图像输入能力的模型（如 DeepSeek）不会看到原图，只依据本地程序测量的 FaceFacts 设计。" +
      "\n\n只有你现在点击「确定」才会提交，本轮不会自动重试。是否继续？",
  );
  if (approved) return true;
  elements.stylingFeedback.textContent =
    "已取消 AI 设计，没有提交任何付费请求。可改用「免费模板预览」（0 次模型调用，不产生费用）。";
  elements.stylingFeedback.className = "ai-feedback";
  return false;
}


function designSourceLabel(proposal) {
  const isAi = proposal.designMode === "ai";
  const provider = proposal.sourceProvider ? ` · ${proposal.sourceProvider}` : "";
  return isAi
    ? { text: `AI 个性化设计${provider}`, className: "styling-source-badge ai" }
    : { text: "免费模板预览", className: "styling-source-badge template" };
}


function referenceSourceLabel(proposal) {
  const presets = Array.isArray(proposal.referencePresets) ? proposal.referencePresets : [];
  if (!presets.length) return "";
  const source = proposal.stylingVocabularySource || presets[0].sourceFile || "stylekit";
  return `风格库来源：${source}`;
}


function renderStylingCards(proposals) {
  const looks = Array.isArray(proposals) && proposals.length ? proposals : currentStylingProposals();
  elements.stylingGrid.innerHTML = looks
    .map((look) => {
      const active = look.planId === state.selectedStylingId;
      const presets = Array.isArray(look.referencePresets) ? look.referencePresets : [];
      const source = designSourceLabel(look);
      const presetChips = presets
        .slice(0, 6)
        .map((preset) => `<span class="styling-chip">${escapeHtml(preset.nameZh || preset.name || preset.presetId)}</span>`)
        .join("");
      return `
        <article class="styling-card ${active ? "active" : ""}" data-styling-card="${escapeHtml(look.planId)}" data-design-mode="${escapeHtml(look.designMode || "template")}">
          <header class="styling-card-head">
            <strong>${escapeHtml(look.name || look.title)}</strong>
            <em>${escapeHtml(referenceSourceLabel(look) || "stylekit")}</em>
          </header>
          <span class="${source.className}">${escapeHtml(source.text)}</span>
          <span class="styling-card-direction">${escapeHtml(look.designDirection || look.designAxis || "")}</span>
          <dl>
            <dt>发型</dt><dd>${escapeHtml(look.hairPlan)}</dd>
            <dt>妆容</dt><dd>${escapeHtml(look.makeupPlan)}</dd>
            <dt>穿搭</dt><dd>${escapeHtml(look.outfitPlan)}</dd>
            <dt>配饰</dt><dd>${escapeHtml(look.accessoryPlan || "无")}</dd>
            <dt>配色</dt><dd>${escapeHtml(look.palette || "")}</dd>
          </dl>
          <p class="styling-why">为什么适合当前人物：${escapeHtml(look.whyItFits || look.designReason || "")}</p>
          ${presetChips ? `<div class="styling-chips">${presetChips}</div>` : ""}
          <button type="button" data-styling-generate="${escapeHtml(look.planId)}">生成这套造型</button>
        </article>`;
    })
    .join("");

  elements.stylingGrid.querySelectorAll("[data-styling-card]").forEach((card) => {
    card.addEventListener("click", () => selectStylingProposal(card.dataset.stylingCard));
  });
  elements.stylingGrid.querySelectorAll("[data-styling-generate]").forEach((btn) => {
    btn.addEventListener("click", (event) => {
      event.stopPropagation();
      selectStylingProposal(btn.dataset.stylingGenerate);
      openCostDialog(btn.dataset.stylingGenerate);
    });
  });
}


function setStylingBusy(busy) {
  state.stylingBusy = busy;
  elements.stylingDesignButton.disabled = busy;
  if (elements.stylingTemplateButton) elements.stylingTemplateButton.disabled = busy;
  if (elements.stylingRefineButton) elements.stylingRefineButton.disabled = busy;
}


async function designStylingFlow(mode = "template") {
  if (state.stylingBusy) return;
  const isAi = mode === "ai";
  const brainIsPaid = Boolean(selectedProvider("brain")?.requiresApiKey);
  if (isAi && !confirmPaidStylingDesign()) return;
  const userInstruction = (elements.stylingInstruction?.value || "").trim();
  setStylingBusy(true);
  elements.stylingFeedback.textContent = isAi
    ? brainIsPaid
      ? "正在调用 AI 设计三套造型（完整流程最多 3 次付费模型调用：1 次人物事实分析 + 1 次造型设计 + 1 次出图提示词规划；已有可用分析缓存时只计 2 次。该模型不看原图，只依据本地 FaceFacts 设计）。"
      : "正在用本地规则模型设计三套造型（0 次付费模型调用，不产生任何费用）。"
    : "正在使用免费模板组合生成三套造型预览（0 次模型调用，不产生任何费用）。";
  elements.stylingFeedback.className = "ai-feedback";

  try {
    // The free "模板预览" flow must stay genuinely zero-cost, so its photo
    // analysis runs on the local derived brain (no model calls) and the image
    // provider is only constructed, never used. The paid "AI 设计" flow uses the
    // vision brain and image provider the user has selected.
    const freeBrain = (state.capabilities.brainProviders || []).find(
      (provider) => provider.id === "local" && !provider.requiresApiKey,
    );
    const freeAction = (state.capabilities.actionProviders || []).find(
      (provider) => provider.id === "mock" && !provider.requiresApiKey,
    );
    const brain =
      !isAi && freeBrain
        ? { name: freeBrain.id, apiKey: "", workspaceId: "" }
        : modelRequest("brain");
    const action =
      !isAi && freeAction
        ? { name: freeAction.id, apiKey: "", workspaceId: "" }
        : modelRequest("action");
    validateModelSelection(state.capabilities, brain, action);
    const file = await currentSourceFile();
    const result = await designStyling({ file, brain, action, mode, userInstruction, api: aiApi });
    state.stylingPhoto = result.photo;
    state.stylingAnalysis = result.analysis;
    state.stylingProposals = result.proposals;
    state.selectedStylingId = result.proposals[0].planId;
    state.generatedStylingJobId = null;
    renderDetectionFromAnalysis(result.analysis);
    elements.stylingGrid.hidden = false;
    elements.stylingRefine.hidden = true;
    renderStylingCards(result.proposals);
    selectStylingProposal(state.selectedStylingId);
    elements.stylingFeedback.textContent = isAi
      ? brainIsPaid
        ? `AI 个性化设计完成 ${result.proposals.length} 套造型（${
            result.analysisReused
              ? "复用已有分析缓存，本流程共 2 次付费模型调用：1 次造型设计 + 1 次出图提示词规划"
              : "完整流程 3 次付费模型调用：1 次人物事实分析 + 1 次造型设计 + 1 次出图提示词规划"
          }；该模型不看原图，只依据本地 FaceFacts 设计），选择一套再生成图片。`
        : `AI 个性化设计完成 ${result.proposals.length} 套造型（本地规则模型，0 次付费模型调用），选择一套再生成图片。`
      : `已生成 ${result.proposals.length} 套免费模板预览（0 次模型调用），选择一套再生成图片。`;
    elements.stylingFeedback.className = "ai-feedback success";
  } catch (error) {
    // A failed AI design must surface as an error — never silently fall back to
    // the local templates while pretending the AI produced the result.
    elements.stylingFeedback.textContent = isAi
      ? `AI 个性化设计失败：${error.message}（未生成或使用任何模板方案，可改用「免费模板预览」。）`
      : error.message;
    elements.stylingFeedback.className = "ai-feedback error";
  } finally {
    setStylingBusy(false);
  }
}


function describeFaceLock(faceLock) {
  if (!faceLock || typeof faceLock.ok !== "boolean") return "";
  const mode = faceLock.diagnostics?.mode;
  const suffix = mode ? `（${mode}）` : "";
  return faceLock.ok
    ? ` · 严格保脸校验通过${suffix}`
    : ` · 保脸校验未通过（${faceLock.status}）`;
}

function describeStylingFailure(error) {
  const faceLock = error?.faceLock;
  if (faceLock && faceLock.ok === false) {
    // The styling pixels may look nice, but the original face could not be
    // restored with confidence -- report the identity failure honestly.
    return `保脸失败（${faceLock.status}）：${faceLock.message || "未能把原始五官无痕还原。"} 生成图未通过严格人脸锁定校验，请人工检查保护区域或调整掩膜后重试。`;
  }
  return error?.message || "AI 造型生成失败。";
}

// The backend names each render stage so the UI can show real progress instead
// of an opaque spinner. Ordered list of the six visible steps the person sees.
const GENERATION_STEP_ORDER = [
  "queued",
  "segmentation",
  "mask_build",
  "provider",
  "verification",
  "done",
];

const STAGE_LABELS = {
  queued: "准备照片",
  segmentation: "自动分割",
  mask_build: "生成蒙版",
  provider_preflight: "RunningHub 处理",
  provider_upload: "RunningHub 处理",
  provider_submitted: "RunningHub 处理",
  provider_running: "RunningHub 处理",
  provider_download: "RunningHub 处理",
  verification: "结果校验",
  done: "完成",
};

// Maps every fine-grained backend stage onto one of the six visible steps.
const STAGE_TO_STEP = {
  queued: "queued",
  segmentation: "segmentation",
  mask_build: "mask_build",
  provider_preflight: "provider",
  provider_upload: "provider",
  provider_submitted: "provider",
  provider_running: "provider",
  provider_download: "provider",
  verification: "verification",
  done: "done",
};

function resolveJobStage(job) {
  if (job?.stage) return job.stage;
  if (job?.status === "succeeded") return "done";
  return "queued";
}

function generationStageText(job) {
  const stage = job?.stage || resolveJobStage(job);
  if (job?.status === "failed") {
    return `${STAGE_LABELS[stage] || "生成"} · 失败`;
  }
  switch (stage) {
    case "provider_preflight":
      return "RunningHub 处理 · 校验工作流与蒙版";
    case "provider_upload":
      return "RunningHub 处理 · 自动上传原图与蒙版";
    case "provider_submitted":
      return job?.providerTaskId
        ? `RunningHub 处理 · 已提交任务 ${job.providerTaskId}`
        : "RunningHub 处理 · 已提交任务";
    case "provider_running":
      return "RunningHub 处理 · 等待生成结果";
    case "provider_download":
      return "RunningHub 处理 · 下载并保存结果";
    case "done":
      return job?.status === "succeeded" ? "完成" : "已结束";
    default:
      return STAGE_LABELS[stage] || "准备照片";
  }
}

function resetGenerationProgress() {
  state.lastGenerationStage = "queued";
  if (!elements.generationSteps) return;
  elements.generationSteps.querySelectorAll("[data-gen-stage]").forEach((li) => {
    li.classList.remove("active", "done", "failed");
  });
}

function renderGenerationProgress(job) {
  if (!elements.generationSteps) return;
  elements.stylingGeneration.hidden = false;
  const failed = job?.status === "failed";
  let stage = resolveJobStage(job);
  if (failed && stage === "done") {
    stage = state.lastGenerationStage || "verification";
  }
  if (!failed) state.lastGenerationStage = stage;
  const currentStep = STAGE_TO_STEP[stage] || "queued";
  const currentIndex = GENERATION_STEP_ORDER.indexOf(currentStep);
  const complete = !failed && currentStep === "done";
  elements.generationSteps.querySelectorAll("[data-gen-stage]").forEach((li) => {
    const index = GENERATION_STEP_ORDER.indexOf(li.dataset.genStage);
    li.classList.remove("active", "done", "failed");
    if (index < currentIndex) {
      li.classList.add("done");
    } else if (index === currentIndex) {
      if (failed) li.classList.add("failed");
      else if (complete) li.classList.add("done");
      else li.classList.add("active");
    }
  });
  if (elements.generationStageNote) {
    elements.generationStageNote.textContent = generationStageText(job);
  }
}

function verificationLabel(job) {
  const level = job?.verificationLevel;
  if (level === "visual_pending") {
    return {
      text: "技术检查通过 · 未完成视觉验收",
      className: "verification-badge pending",
    };
  }
  if (level === "visual_failed") {
    return { text: "视觉校验失败", className: "verification-badge failed" };
  }
  if (job?.status === "succeeded") {
    return { text: "技术检查通过", className: "verification-badge passed" };
  }
  return { text: "", className: "verification-badge" };
}

function verificationNote(job, proposalName) {
  const level = job?.verificationLevel;
  const lockBadge = describeFaceLock(job?.faceLock);
  const model = job?.modelName ? ` · ${job.modelName}` : "";
  if (level === "visual_pending" || level === "technical_pass" || !level) {
    return `「${proposalName}」已通过机器技术校验（尺寸、身份保护、背景变化）${model}${lockBadge}。该结果未完成人工视觉验收，局部细节可能存在瑕疵，可查看和下载，也可回到上方重新选择其他造型或提出二次修改。`;
  }
  return `「${proposalName}」生成完成${model}${lockBadge}。`;
}

let activeHistoryJobId = null;

function hideStylingResult() {
  if (!elements.stylingResult) return;
  elements.stylingResult.hidden = true;
  activeHistoryJobId = null;
}

async function showStylingResult(job, proposal) {
  const url = Array.isArray(job?.outputUrls) ? job.outputUrls[0] : "";
  if (!url) return;
  const name = proposal?.name || proposal?.title || job?.planId || "造型结果";
  const absolute = assetUrl(url);
  const beforeSrc = elements.beforeImage?.currentSrc || elements.beforeImage?.src || "";

  state.mode = "ai";
  await showAiResult(absolute);
  elements.exportButton.href = absolute;
  elements.canvasStatus.innerHTML = `<span></span>造型结果 · ${escapeHtml(name)}`;
  elements.footerStatus.textContent = "造型结果已生成";

  const badge = verificationLabel(job);
  if (elements.resultVerification) {
    elements.resultVerification.textContent = badge.text;
    elements.resultVerification.className = badge.className;
  }
  if (elements.resultBeforeImage) elements.resultBeforeImage.src = beforeSrc;
  if (elements.resultAfterImage) elements.resultAfterImage.src = absolute;
  if (elements.resultAfterCaption) elements.resultAfterCaption.textContent = name;
  if (elements.resultNote) elements.resultNote.textContent = verificationNote(job, name);
  if (elements.resultDownload) {
    elements.resultDownload.href = absolute;
    elements.resultDownload.setAttribute("download", `${name}.png`);
  }
  elements.stylingResult.hidden = false;
  activeHistoryJobId = job.jobId || null;
  state.generatedStylingJobId = job.jobId || null;
  elements.stylingRefine.hidden = false;
}

function selectedActionProviderName() {
  return modelRequest("action").name || "";
}

function actionIsLocalSimulation() {
  const name = selectedActionProviderName();
  return name === "mock" || name === "local" || name === "";
}

function openCostDialog(planId) {
  const proposal = currentStylingProposals().find((look) => look.planId === planId);
  if (!proposal) return;
  if (!state.stylingPhoto) {
    elements.stylingFeedback.textContent = "请先让 AI 设计三套造型，再生成图片。";
    elements.stylingFeedback.className = "ai-feedback error";
    return;
  }

  const cost = estimateStylingCost(currentStylingProposals(), {
    selectedCount: 1,
    mode: stylingDesignMode(),
    analysisReused: analysisReused(),
    brainIsPaid: Boolean(selectedProvider("brain")?.requiresApiKey),
  });
  const designRequests = cost.paidDesignRequests;
  const localSim = actionIsLocalSimulation();
  const isAi = stylingDesignMode() === "ai";
  const brainIsPaid = Boolean(selectedProvider("brain")?.requiresApiKey);
  const actionLabel = selectedProvider("action")?.label || "图像服务商";
  const designLabel = isAi
    ? selectedProvider("brain")?.label || "视觉模型"
    : "本地模板";

  const name = proposal.name || proposal.planId;
  if (elements.costDialogLead) {
    elements.costDialogLead.textContent = localSim
      ? `已选择「${name}」。当前图像服务商为本地模拟，不会产生任何付费请求。`
      : `已选择「${name}」。点击确认后会立即向 ${actionLabel} 提交 1 个付费任务。`;
  }
  if (elements.costDesignLabel) {
    elements.costDesignLabel.textContent = `${designLabel} 造型设计（已完成）`;
  }
  if (elements.costDesignValue) {
    if (!isAi) {
      elements.costDesignValue.textContent = "0 次（免费模板模式）";
    } else if (!brainIsPaid) {
      elements.costDesignValue.textContent = `0 次付费（${
        selectedProvider("brain")?.label || "本地规则"
      }，不计费）`;
    } else if (designRequests > 0) {
      elements.costDesignValue.textContent = analysisReused()
        ? `${designRequests} 次（复用分析缓存，仅 1 次造型设计）`
        : `${designRequests} 次（1 次人物事实分析 + 1 次造型设计）`;
    } else {
      elements.costDesignValue.textContent = "0 次";
    }
  }
  if (elements.costImageLabel) {
    elements.costImageLabel.textContent = localSim
      ? "图像生成（本地模拟，不计费）"
      : `${actionLabel} 图像生成（本次提交）`;
  }
  if (elements.costImageValue) {
    elements.costImageValue.textContent = localSim
      ? `0 次付费（本地模拟 ${cost.requestsPerLook} 次）`
      : isAi
        ? `${cost.requestsPerLook} 次付费请求（另含 1 次出图提示词规划）`
        : `${cost.requestsPerLook} 次付费请求`;
  }
  if (elements.costJobCount) {
    elements.costJobCount.textContent = localSim
      ? "1 个本地任务（不计费）"
      : "1 个付费任务";
  }
  if (elements.costDialogNote) {
    const flowNote =
      isAi && brainIsPaid
        ? `完整流程最多 ${cost.paidFlowRequests} 次付费模型调用（${
            analysisReused()
              ? "复用分析缓存：1 次造型设计 + 1 次出图提示词规划"
              : "1 次人物事实分析 + 1 次造型设计 + 1 次出图提示词规划"
          }）。${cost.visionNote}`
        : "";
    elements.costDialogNote.textContent = localSim
      ? `本地模拟只用于验证流程，不调用 ${actionLabel}，也不产生费用。${flowNote}`
      : `服务商：${actionLabel} · 出图只对选中的这一套发起 ${cost.requestsPerLook} 次图像编辑请求，费用以 ${actionLabel} 实际计费为准。${flowNote}`;
  }

  const dialog = elements.costDialog;
  state.pendingStylingPlanId = planId;
  state.pendingRetouchRun = null;
  if (dialog && typeof dialog.showModal === "function") {
    if (!dialog.open) dialog.showModal();
  } else if (dialog) {
    dialog.setAttribute("open", "");
  }
}


// The legacy "AI 美化" button also ends in a paid image request, so it goes
// through the same confirmation dialog: nothing is submitted until the person
// sees the provider and the request count and presses confirm.
function openRetouchCostDialog(brain, action) {
  const actionMeta = (state.capabilities.actionProviders || []).find((p) => p.id === action.name);
  const brainMeta = (state.capabilities.brainProviders || []).find((p) => p.id === brain.name);
  const actionLabel = actionMeta?.label || "图像服务商";
  const localSim = !action.name || action.name === "mock" || action.name === "local";
  const brainLocal = !brain.name || brain.name === "local";

  if (elements.costDialogLead) {
    elements.costDialogLead.textContent = localSim
      ? "当前图像服务商为本地模拟，本次「AI 美化」不会产生任何付费请求。"
      : `点击确认后会立即向 ${actionLabel} 提交 1 个付费任务。`;
  }
  if (elements.costDesignLabel) {
    elements.costDesignLabel.textContent = brainLocal
      ? "人像分析（本地，不计费）"
      : `${brainMeta?.label || "视觉模型"} 人像分析`;
  }
  if (elements.costDesignValue) {
    elements.costDesignValue.textContent = brainLocal ? "0 次付费" : "1 次请求";
  }
  if (elements.costImageLabel) {
    elements.costImageLabel.textContent = localSim
      ? "图像生成（本地模拟，不计费）"
      : `${actionLabel} 图像生成（本次提交）`;
  }
  if (elements.costImageValue) {
    elements.costImageValue.textContent = localSim
      ? "0 次付费（本地模拟 1 次）"
      : "1 次付费请求";
  }
  if (elements.costJobCount) {
    elements.costJobCount.textContent = localSim ? "1 个本地任务（不计费）" : "1 个付费任务";
  }
  if (elements.costDialogNote) {
    elements.costDialogNote.textContent = localSim
      ? `本地模拟只用于验证流程，不调用 ${actionLabel}，也不产生费用。`
      : `服务商：${actionLabel} · 只提交 1 个任务，不自动重试，费用以 ${actionLabel} 实际计费为准。`;
  }

  state.pendingStylingPlanId = null;
  state.pendingRetouchRun = () => runAiRetouchGeneration(brain, action);
  const dialog = elements.costDialog;
  if (dialog && typeof dialog.showModal === "function") {
    if (!dialog.open) dialog.showModal();
  } else if (dialog) {
    dialog.setAttribute("open", "");
  }
}


async function runAiRetouchGeneration(brain, action) {
  if (state.aiBusy) return;
  const label = elements.generateButton.querySelector("b");
  state.aiBusy = true;
  elements.generateButton.classList.add("loading");
  label.textContent = "AI 正在分析并精修…";
  elements.footerStatus.textContent = "AI 精修生成中";
  elements.aiFeedback.textContent = "正在上传并分析照片，请保持页面打开。";
  elements.aiFeedback.className = "ai-feedback";

  try {
    validateModelSelection(state.capabilities, brain, action);
    const file = await currentSourceFile();
    const result = await executeAiRetouch({
      file, values: state.values, preset: state.selectedPreset,
      userInstruction: elements.aiPrompt.value, brain, action, api: aiApi,
    });
    const resultUrl = assetUrl(result.job.outputUrls[0]);
    await showAiResult(resultUrl);
    elements.exportButton.href = resultUrl;
    elements.canvasStatus.innerHTML = `<span></span>AI 结果 · ${result.job.modelName}`;
    elements.footerStatus.textContent = "AI 美化结果已生成";
    elements.aiFeedback.textContent = `${result.job.brainModel} + ${result.job.modelName} · 生成成功`;
    elements.aiFeedback.className = "ai-feedback success";
    label.textContent = "重新生成 AI 美化效果";
    loadJobHistory();
  } catch (error) {
    elements.footerStatus.textContent = "AI 美化失败";
    elements.aiFeedback.textContent = error.message;
    elements.aiFeedback.className = "ai-feedback error";
    label.textContent = "重新尝试 AI 美化";
  } finally {
    state.aiBusy = false;
    elements.generateButton.classList.remove("loading");
  }
}

function closeCostDialog() {
  const dialog = elements.costDialog;
  if (dialog && typeof dialog.close === "function" && dialog.open) dialog.close();
  else if (dialog) dialog.removeAttribute("open");
  // Cancel must never leave a queued paid run behind.
  state.pendingRetouchRun = null;
  state.pendingStylingPlanId = null;
}

async function runStylingGeneration(planId) {
  if (state.stylingBusy) return;
  const proposal = currentStylingProposals().find((look) => look.planId === planId);
  if (!proposal) return;
  if (!state.stylingPhoto) {
    elements.stylingFeedback.textContent = "请先让 AI 设计三套造型。";
    elements.stylingFeedback.className = "ai-feedback error";
    return;
  }

  const userInstruction = (elements.stylingInstruction?.value || "").trim();
  hideStylingResult();
  resetGenerationProgress();
  renderGenerationProgress({ status: "queued", stage: "queued" });
  setStylingBusy(true);
  elements.stylingFeedback.textContent = `正在生成「${proposal.name || planId}」，请保持页面打开，不要重复提交。`;
  elements.stylingFeedback.className = "ai-feedback";
  elements.footerStatus.textContent = "AI 造型生成中";

  try {
    const brain = modelRequest("brain");
    const action = modelRequest("action");
    const result = await executeStyling({
      photo: state.stylingPhoto,
      proposals: currentStylingProposals(),
      selectedId: planId,
      userInstruction,
      brain,
      action,
      api: aiApi,
      onStage: (job) => renderGenerationProgress(job),
    });
    renderGenerationProgress({ ...result.job, status: "succeeded", stage: "done" });
    await showStylingResult(result.job, result.proposal);
    elements.stylingFeedback.textContent = `「${result.proposal.name || planId}」生成完成，可查看前后对比、下载，或提出二次修改。`;
    elements.stylingFeedback.className = "ai-feedback success";
    loadJobHistory();
  } catch (error) {
    elements.footerStatus.textContent = "AI 造型失败";
    const failedJob = error?.job || null;
    if (failedJob) renderGenerationProgress(failedJob);
    else {
      renderGenerationProgress({ status: "failed", stage: state.lastGenerationStage || "verification" });
    }
    elements.stylingFeedback.textContent = describeStylingFailure(error);
    elements.stylingFeedback.className = "ai-feedback error";
  } finally {
    setStylingBusy(false);
  }
}

function historyJobProposal(job) {
  const plan = job?.plan || null;
  return {
    name: plan?.name || plan?.title || job?.planId || "历史造型",
    planId: job?.planId || plan?.planId || "",
  };
}

function renderJobHistory(jobs) {
  if (!elements.historyList) return;
  const entries = (Array.isArray(jobs) ? jobs : []).filter(
    (job) => Array.isArray(job?.outputUrls) && job.outputUrls.length,
  );
  if (!entries.length) {
    elements.historyList.innerHTML =
      '<p class="history-empty">暂无已生成的造型结果，生成成功后会出现在这里。</p>';
    return;
  }
  elements.historyList.innerHTML = entries
    .map((job) => {
      const proposal = historyJobProposal(job);
      const badge = verificationLabel(job);
      const when = job.createdAt ? escapeHtml(String(job.createdAt).replace("T", " ").slice(0, 19)) : "";
      return `
        <button class="history-item" type="button" data-history-job="${escapeHtml(job.jobId)}">
          <span class="history-title">${escapeHtml(proposal.name)}</span>
          <span class="${badge.className}">${escapeHtml(badge.text)}</span>
          <span class="history-meta">${when} · ${escapeHtml(job.jobId)}</span>
        </button>`;
    })
    .join("");
  elements.historyList.querySelectorAll("[data-history-job]").forEach((btn) => {
    btn.addEventListener("click", () => openHistoryJob(btn.dataset.historyJob));
  });
}

async function openHistoryJob(jobId) {
  try {
    const job = await getJob(jobId);
    if (job?.status !== "succeeded" || !job.outputUrls?.length) {
      elements.stylingFeedback.textContent = `该历史任务当前状态为「${job?.status || "未知"}」${
        job?.errorMessage ? `：${job.errorMessage}` : ""
      }`;
      elements.stylingFeedback.className = "ai-feedback error";
      return;
    }
    await showStylingResult(job, historyJobProposal(job));
    elements.stylingFeedback.textContent = "已载入历史造型结果，可查看对比或下载。";
    elements.stylingFeedback.className = "ai-feedback success";
    elements.stylingResult.scrollIntoView({ behavior: "smooth", block: "nearest" });
  } catch (error) {
    elements.stylingFeedback.textContent = `读取历史结果失败：${error.message}`;
    elements.stylingFeedback.className = "ai-feedback error";
  }
}

async function loadJobHistory() {
  if (!elements.historyList) return;
  try {
    const jobs = await listJobs(20, 0);
    renderJobHistory(jobs);
  } catch (error) {
    elements.historyList.innerHTML = `<p class="history-empty">历史结果读取失败：${escapeHtml(
      error.message,
    )}</p>`;
  }
}


async function refineStylingFlow() {
  if (state.stylingBusy) return;
  const userInstruction = (elements.stylingInstruction?.value || "").trim();
  if (!state.generatedStylingJobId) {
    elements.stylingFeedback.textContent = "请先生成一套造型，再进行二次修改。";
    elements.stylingFeedback.className = "ai-feedback error";
    return;
  }
  if (!userInstruction) {
    elements.stylingFeedback.textContent = "请填写你的修改要求，例如“头发再短一点”。";
    elements.stylingFeedback.className = "ai-feedback error";
    return;
  }

  setStylingBusy(true);
  elements.stylingFeedback.textContent = "即将基于上一版结果发起 1 次付费图像编辑请求。";
  elements.stylingFeedback.className = "ai-feedback";

  try {
    const brain = modelRequest("brain");
    const action = modelRequest("action");
    const result = await refineStyling({
      jobId: state.generatedStylingJobId,
      userInstruction,
      brain,
      action,
      api: aiApi,
    });
    const resultUrl = assetUrl(result.job.outputUrls[0]);
    state.mode = "ai";
    await showAiResult(resultUrl);
    elements.exportButton.href = resultUrl;
    elements.canvasStatus.innerHTML = "<span></span>造型二次修改结果";
    elements.footerStatus.textContent = "二次修改完成";
    state.generatedStylingJobId = result.job.jobId;
    elements.stylingInstruction.value = "";
    elements.stylingFeedback.textContent = "二次修改已生成，可继续提出下一条要求。";
    elements.stylingFeedback.className = "ai-feedback success";
  } catch (error) {
    elements.stylingFeedback.textContent = error.message;
    elements.stylingFeedback.className = "ai-feedback error";
  } finally {
    setStylingBusy(false);
  }
}


async function loadStylingVocabulary() {
  if (!elements.stylingSource) return;
  try {
    const info = await getStylingProviders();
    const vocab = info.stylingVocabulary || {};
    const files = Array.isArray(vocab.loadedFiles) ? vocab.loadedFiles.length : 0;
    elements.stylingSource.textContent = vocab.available
      ? `风格库已加载 · ${vocab.sourceLabel || "stylekit"} · ${files} 个文件`
      : "风格库不可用，将使用内置兜底词汇";
    const cost = estimateStylingCost(
      [{ estimatedPaidRequests: info.estimatedRequestsPerGeneration || 1 }],
      { selectedCount: 1, mode: "ai" },
    );
    if (elements.stylingCostNote) {
      const flowRequests = Number(info.stylingAiFlowRequests ?? cost.paidFlowRequests) || 3;
      const designRequests =
        Number(info.stylingAiDesignRequests ?? cost.designRequests) || 1;
      const generationRequests =
        Number(info.stylingAiGenerationRequests ?? cost.generationRequests) || 1;
      elements.stylingCostNote.textContent =
        `AI 个性化设计：完整流程含 1 次人物事实分析 + 1 次造型设计 + ${generationRequests} 次出图提示词规划，共 ${flowRequests} 次付费模型调用；该照片已有有效分析缓存时复用分析，只计 ${designRequests + generationRequests} 次。无图像输入能力的模型（如 DeepSeek）不会看到原图，只依据本地程序测量的 FaceFacts 设计，并未直接做视觉看图分析。免费模板预览 0 次调用。出图只对选中的一套发起 ${cost.requestsPerLook} 次图像编辑请求。`;
    }
  } catch (error) {
    elements.stylingSource.textContent = "风格库状态读取失败";
  }
}


// --- Photo upload + honest detection state --------------------------------
// The inspector card used to be hard-coded to "质量良好 / 面部 1 / 可见度 98%",
// so it advertised a well-detected portrait even before a photo was chosen.
// Everything below is driven by the real upload and analysis results only.

const FACE_VISIBILITY_LABELS = {
  clear: "清晰",
  visible: "清晰",
  good: "清晰",
  partial: "部分可见",
  obscured: "被遮挡",
  unknown: "未识别",
};

let photoUploadToken = 0;

function setCanvasStatus(text) {
  if (!elements.canvasStatus) return;
  const dot = elements.canvasStatus.querySelector("span");
  elements.canvasStatus.textContent = "";
  if (dot) elements.canvasStatus.appendChild(dot);
  elements.canvasStatus.append(text);
}

function updateDetectionCard({ status = "idle", label, faceCount, visibility, light }) {
  if (elements.detectionLabel) {
    elements.detectionLabel.textContent = label;
    elements.detectionLabel.classList.toggle("is-error", status === "error");
    elements.detectionLabel.classList.toggle("is-pending", status !== "analyzed");
  }
  if (elements.detectionFaceCount) elements.detectionFaceCount.textContent = faceCount;
  if (elements.detectionVisibility) elements.detectionVisibility.textContent = visibility;
  if (elements.detectionLight) elements.detectionLight.textContent = light;
}

function markDetectionIdle(label = "未选择照片") {
  updateDetectionCard({ status: "idle", label, faceCount: "—", visibility: "—", light: "—" });
}

function markDetectionAwaitingAnalysis(label = "照片已载入") {
  updateDetectionCard({
    status: "loaded",
    label,
    faceCount: "待分析",
    visibility: "待分析",
    light: "待分析",
  });
}

// Only show what the analyzer reported for this photo. The free template flow
// runs the local rule-based analyzer, so its numbers are labelled as an estimate
// instead of being dressed up as a vision-model verdict.
function renderDetectionFromAnalysis(analysis) {
  if (!analysis) {
    markDetectionAwaitingAnalysis();
    return;
  }
  const derived = String(analysis.brainProvider || "local") === "local";
  const subjects = analysis.subjects || {};
  const count = Number.isFinite(subjects.count) ? String(subjects.count) : "未识别";
  const rawVisibility = subjects.faceVisibility;
  const visibility =
    FACE_VISIBILITY_LABELS[String(rawVisibility || "").toLowerCase()] ||
    rawVisibility ||
    "未识别";
  const issues = Array.isArray(analysis.lightingIssues) ? analysis.lightingIssues : [];
  updateDetectionCard({
    status: "analyzed",
    label: derived ? "本地估算" : "AI 分析完成",
    faceCount: count,
    visibility,
    light: issues.length ? issues.join(" / ") : "未见明显问题",
  });
  if (elements.portraitMeta) {
    elements.portraitMeta.textContent = derived
      ? "本地估算 · 未调用视觉模型"
      : `已识别 ${count} 个人像`;
  }
}

function resetPhotoSelection() {
  state.stylingPhoto = null;
  state.stylingAnalysis = null;
  state.stylingProposals = [];
  state.selectedStylingId = "";
  state.generatedStylingJobId = null;
  state.backendPhoto = null;
  elements.stylingGrid.hidden = true;
  elements.stylingRefine.hidden = true;
  elements.stylingResult.hidden = true;
}

async function acceptSelectedPhoto(file) {
  // Show the picked file immediately so the click feels instant...
  if (state.sourceObjectUrl) URL.revokeObjectURL(state.sourceObjectUrl);
  const url = URL.createObjectURL(file);
  state.sourceObjectUrl = url;
  state.sourceFile = file;
  state.sourceName = file.name;
  state.aiResultImage = null;
  state.aiResultUrl = "";
  state.displayingAiResult = false;
  resetPhotoSelection();
  elements.beforeImage.src = url;
  elements.exportButton.href = url;
  if (elements.imageWorkbench) elements.imageWorkbench.dataset.loaded = "1";
  if (elements.documentName) elements.documentName.textContent = file.name;
  if (elements.documentMeta) elements.documentMeta.textContent = "正在上传…";
  if (elements.portraitName) elements.portraitName.textContent = "正在上传…";
  if (elements.portraitMeta) elements.portraitMeta.textContent = file.name;
  setCanvasStatus("正在上传照片…");
  markDetectionIdle("正在上传…");
  elements.footerStatus.textContent = "正在把照片上传到本地服务…";

  // ...then actually store it on the backend. The photo id returned here is what
  // retouch and styling jobs are keyed on, so a photo that never reached the
  // backend must not pretend to be "loaded".
  const token = ++photoUploadToken;
  try {
    const photo = await uploadPhoto(file);
    if (token !== photoUploadToken) return;
    state.backendPhoto = photo;
    console.info("[photo] uploaded to backend", photo);
    if (elements.documentMeta) {
      elements.documentMeta.textContent = `已加载 ${photo.width}×${photo.height}`;
    }
    if (elements.portraitName) elements.portraitName.textContent = "人物 1";
    if (elements.portraitMeta) elements.portraitMeta.textContent = `照片 ID ${photo.imageId}`;
    setCanvasStatus(
      state.mode === "ai" ? "AI 造型设计 · 照片已就绪" : "实时预览已开启",
    );
    markDetectionAwaitingAnalysis();
    elements.footerStatus.textContent = `照片已上传 · 图片 ID ${photo.imageId}`;
  } catch (error) {
    if (token !== photoUploadToken) return;
    state.backendPhoto = null;
    if (elements.documentMeta) elements.documentMeta.textContent = "上传失败";
    if (elements.portraitName) elements.portraitName.textContent = "上传失败";
    setCanvasStatus("上传失败");
    updateDetectionCard({
      status: "error",
      label: "上传失败",
      faceCount: "—",
      visibility: "—",
      light: "—",
    });
    elements.footerStatus.textContent = `照片上传失败：${error.message}`;
    elements.aiFeedback.textContent = `照片上传失败：${error.message}`;
    elements.aiFeedback.className = "ai-feedback error";
  }
}


function init() {
  document.querySelectorAll("[data-category]").forEach((btn) => {
    btn.addEventListener("click", () => {
      state.category = btn.dataset.category;
      elements.toolSearch.value = "";
      document.querySelectorAll("[data-category]").forEach((b) => b.classList.toggle("active", b === btn));
      renderCategory();
    });
  });

  document.querySelectorAll("[data-mode]").forEach((btn) => {
    btn.addEventListener("click", () => setMode(btn.dataset.mode));
  });

  elements.toolSearch.addEventListener("input", renderCategory);
  elements.compareRange.addEventListener("input", () => {
    const v = Number(elements.compareRange.value);
    elements.afterLayer.style.clipPath = `inset(0 0 0 ${v}%)`;
    elements.compareLine.style.left = `${v}%`;
    elements.compareRange.style.setProperty("--range-progress", `${v}%`);
  });

  elements.compareButton.addEventListener("click", () => {
    state.originalOnly = !state.originalOnly;
    elements.afterLayer.style.opacity = state.originalOnly ? "0" : "1";
    elements.compareLine.hidden = state.originalOnly;
    elements.compareButton.lastChild.textContent = state.originalOnly ? " 显示美化" : " 查看原图";
  });

  elements.resetCategoryButton.addEventListener("click", () => {
    resetTools(categories[state.category].tools);
  });
  elements.resetButton.addEventListener("click", () => {
    resetTools(allTools);
  });
  elements.undoButton.addEventListener("click", () => {
    const prev = state.history.pop();
    if (!prev) return;
    if (prev.snapshot) {
      state.values = { ...state.values, ...prev.snapshot };
    } else {
      state.values = { ...state.values, [prev.id]: prev.value };
    }
    renderCategory();
    updatePreview();
  });

  elements.uploadButton.addEventListener("click", () => elements.fileInput.click());
  elements.fileInput.addEventListener("change", () => {
    const [file] = elements.fileInput.files;
    // Reset so re-picking the same file still fires a change event.
    elements.fileInput.value = "";
    if (!file) return;
    acceptSelectedPhoto(file);
  });

  // The central frame doubles as the upload area: while no photo has reached the
  // backend, clicking it opens the picker and dropping a file starts the same
  // flow. Once a photo is stored, the frame goes back to being a compare canvas.
  if (elements.imageWorkbench) {
    elements.imageWorkbench.addEventListener("click", () => {
      if (state.backendPhoto) return;
      elements.fileInput.click();
    });
    ["dragenter", "dragover"].forEach((type) => {
      elements.imageWorkbench.addEventListener(type, (event) => {
        event.preventDefault();
      });
    });
    elements.imageWorkbench.addEventListener("drop", (event) => {
      event.preventDefault();
      const [file] = event.dataTransfer?.files || [];
      if (file) acceptSelectedPhoto(file);
    });
  }

  elements.beforeImage.addEventListener("error", () => {
    if (elements.documentName) elements.documentName.textContent = "请选择照片";
    if (elements.documentMeta) elements.documentMeta.textContent = "未加载";
    elements.footerStatus.textContent = "照片无法显示，请重新选择图片";
  });

  elements.beforeImage.addEventListener("load", scheduleLocalRender);

  document.querySelectorAll("[data-preset]").forEach((btn) => {
    btn.addEventListener("click", () => {
      state.selectedPreset = btn.dataset.preset;
      document.querySelectorAll("[data-preset]").forEach((b) => b.classList.toggle("active", b === btn));
    });
  });

  elements.aiPrompt.addEventListener("input", () => {
    elements.promptCount.textContent = String(elements.aiPrompt.value.length);
  });

  elements.brainProviderSelect.addEventListener("change", syncModelConfig);
  elements.actionProviderSelect.addEventListener("change", syncModelConfig);
  elements.brainApiKeyInput.addEventListener("input", syncModelConfig);
  elements.actionApiKeyInput.addEventListener("input", syncModelConfig);
  elements.exportButton.addEventListener("click", exportCurrentImage);

  state.stylingProposals = state.stylingProposals || [];
  state.selectedStylingId = state.selectedStylingId ?? null;
  state.stylingBusy = false;
  state.stylingPhoto = state.stylingPhoto || null;
  state.generatedStylingJobId = state.generatedStylingJobId ?? null;
  state.lastGenerationStage = state.lastGenerationStage || "queued";
  state.pendingStylingPlanId = state.pendingStylingPlanId ?? null;
  state.pendingRetouchRun = state.pendingRetouchRun ?? null;
  elements.stylingDesignButton.addEventListener("click", () => designStylingFlow("ai"));
  if (elements.stylingTemplateButton) {
    elements.stylingTemplateButton.addEventListener("click", () => designStylingFlow("template"));
  }
  elements.stylingRefineButton.addEventListener("click", refineStylingFlow);

  // The cost confirmation dialog is the only path to a paid generation: the
  // card button merely opens it, and RunningHub is only called on confirm.
  if (elements.costConfirmButton) {
    elements.costConfirmButton.addEventListener("click", () => {
      const planId = state.pendingStylingPlanId;
      const retouchRun = state.pendingRetouchRun;
      closeCostDialog();
      state.pendingRetouchRun = null;
      if (retouchRun) retouchRun();
      else if (planId) runStylingGeneration(planId);
    });
  }
  if (elements.costCancelButton) {
    elements.costCancelButton.addEventListener("click", closeCostDialog);
  }
  elements.costDialog?.addEventListener("cancel", (event) => {
    event.preventDefault();
    closeCostDialog();
  });
  if (elements.resultReselect) {
    elements.resultReselect.addEventListener("click", () => {
      hideStylingResult();
      elements.stylingGrid.scrollIntoView({ behavior: "smooth", block: "start" });
      elements.stylingFeedback.textContent = "请重新选择一套造型，再次点击「生成这套造型」。";
      elements.stylingFeedback.className = "ai-feedback";
    });
  }
  if (elements.historyRefresh) {
    elements.historyRefresh.addEventListener("click", loadJobHistory);
  }

  elements.generateButton.addEventListener("click", () => {
    if (state.aiBusy) return;
    const brain = modelRequest("brain");
    const action = modelRequest("action");
    try {
      validateModelSelection(state.capabilities, brain, action);
    } catch (error) {
      elements.footerStatus.textContent = "AI 美化无法开始";
      elements.aiFeedback.textContent = error.message;
      elements.aiFeedback.className = "ai-feedback error";
      return;
    }
    // The button never submits the paid request itself: it opens the same
    // confirmation dialog so the provider and request count are seen first.
    openRetouchCostDialog(brain, action);
  });

  document.querySelector("#aiShortcut").addEventListener("click", () => setMode("ai"));
  if (elements.floatingTip) {
    elements.floatingTip.querySelector("button")?.addEventListener("click", () => {
      elements.floatingTip.hidden = true;
    });
  }

  // The markup ships with the AI styling workspace open, so the JS state has to
  // start in the same mode; otherwise the first status message ("照片已就绪"
  // vs "实时预览已开启") and the local render loop disagree with the visible UI.
  setMode(state.mode);
  renderCategory();
  updatePreview();
  elements.compareRange.dispatchEvent(new Event("input"));
  if (elements.beforeImage.complete && elements.beforeImage.naturalWidth) scheduleLocalRender();
  loadProviderCapabilities();
  loadStylingVocabulary();
  loadJobHistory();
}

init();
