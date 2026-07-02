import { state, on, categories, allTools, toolMap } from "./state.js";

import {
  analyzePhoto,
  assetUrl,
  createJob,
  createPlans,
  getProviderCapabilities,
  uploadPhoto,
} from "./api.js";

import {
  executeAiRetouch,
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
};


const aiApi = { uploadPhoto, analyzePhoto, createPlans, createJob };


function populateProviderSelect(select, providers, preferredId) {
  const previous = select.value;
  select.replaceChildren(
    ...providers.map((provider) => {
      const option = document.createElement("option");
      option.value = provider.id;
      option.textContent = `${provider.label} · ${provider.model}`;
      return option;
    }),
  );
  const availableIds = new Set(providers.map((p) => p.id));
  select.value = availableIds.has(previous)
    ? previous
    : availableIds.has(preferredId)
      ? preferredId
      : providers[0]?.id || "";
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
  const canReuseBrainKey =
    !isBrain &&
    ((elements.brainProviderSelect.value === "qwen" &&
      ["qwen", "wan"].includes(providerName)) ||
      (elements.brainProviderSelect.value === "doubao" && providerName === "seedream"));
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
    populateProviderSelect(elements.brainProviderSelect, state.capabilities.brainProviders || [], "qwen");
    populateProviderSelect(elements.actionProviderSelect, state.capabilities.actionProviders || [], "qwen");
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
    : '<span class="local-dot"></span>云端 AI 精修 · 生成前会确认参数';
  elements.canvasStatus.classList.toggle("ai", !isLocal);
  elements.canvasStatus.innerHTML = isLocal
    ? "<span></span>实时预览已开启"
    : state.displayingAiResult ? "<span></span>AI 美化结果" : "<span></span>AI 模式 · 当前显示手动预览";
  elements.footerStatus.textContent = isLocal ? "实时美化已就绪" : "AI 精修已就绪";
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
    if (!file) return;
    const url = URL.createObjectURL(file);
    state.sourceFile = file;
    state.sourceName = file.name;
    state.aiResultImage = null;
    state.aiResultUrl = "";
    state.displayingAiResult = false;
    elements.beforeImage.src = url;
    elements.exportButton.href = url;
    if (elements.documentName) elements.documentName.textContent = file.name;
    elements.footerStatus.textContent = "新照片已载入";
  });

  elements.beforeImage.addEventListener("error", () => {
    if (elements.documentName) elements.documentName.textContent = "请选择照片";
    elements.footerStatus.textContent = "请通过左侧 + 按钮选择照片";
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

  elements.generateButton.addEventListener("click", async () => {
    if (state.aiBusy) return;
    const label = elements.generateButton.querySelector("b");
    const brain = modelRequest("brain");
    const action = modelRequest("action");
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
    } catch (error) {
      elements.footerStatus.textContent = "AI 美化失败";
      elements.aiFeedback.textContent = error.message;
      elements.aiFeedback.className = "ai-feedback error";
      label.textContent = "重新尝试 AI 美化";
    } finally {
      state.aiBusy = false;
      elements.generateButton.classList.remove("loading");
    }
  });

  document.querySelector("#aiShortcut").addEventListener("click", () => setMode("ai"));
  if (elements.floatingTip) {
    elements.floatingTip.querySelector("button")?.addEventListener("click", () => {
      elements.floatingTip.hidden = true;
    });
  }

  renderCategory();
  updatePreview();
  elements.compareRange.dispatchEvent(new Event("input"));
  if (elements.beforeImage.complete && elements.beforeImage.naturalWidth) scheduleLocalRender();
  loadProviderCapabilities();
}

init();
