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
import { renderLocalRetouch } from "./local-retouch.mjs";

const categories = {
  skin: {
    title: "皮肤质感",
    tools: [
      { id: "smooth", label: "光滑肌肤", short: "滑", value: 24, hint: "保留纹理" },
      { id: "skinTone", label: "肤色均匀", short: "肤", value: 12, hint: "校正暗沉" },
      { id: "whiten", label: "自然美白", short: "白", value: 0, hint: "不过曝" },
      { id: "sculpt", label: "立体光影", short: "立", value: 8, hint: "增强轮廓" },
      { id: "acne", label: "面部祛痘", short: "痘", value: 0, hint: "局部净肤" },
      { id: "wrinkle", label: "淡化细纹", short: "纹", value: 0, hint: "保留表情" },
    ],
  },
  face: {
    title: "脸型轮廓",
    tools: [
      { id: "slimFace", label: "瘦脸", short: "瘦", value: 0, hint: "自然收窄" },
      { id: "smallFace", label: "小脸", short: "小", value: 0, hint: "整体缩小" },
      { id: "jawline", label: "下颌线", short: "颌", value: 0, hint: "轮廓清晰" },
      { id: "chin", label: "下巴", short: "下", value: 0, hint: "长度微调" },
      { id: "cheekbone", label: "颧骨", short: "颧", value: 0, hint: "柔和轮廓" },
      { id: "forehead", label: "额头", short: "额", value: 0, hint: "比例微调" },
      { id: "doubleChin", label: "淡化双下巴", short: "颈", value: 0, hint: "收紧轮廓" },
    ],
  },
  eyes: {
    title: "眼部精修",
    tools: [
      { id: "enlargeEyes", label: "放大眼睛", short: "大", value: 0, hint: "双眼同步" },
      { id: "eyeBags", label: "去除眼袋", short: "袋", value: 0, hint: "保留卧蚕" },
      { id: "darkCircles", label: "淡化黑眼圈", short: "黑", value: 0, hint: "自然提亮" },
      { id: "iris", label: "眼眸增强", short: "眸", value: 0, hint: "增强细节" },
      { id: "brightEyes", label: "亮眼", short: "亮", value: 0, hint: "清澈有神" },
      { id: "eyeDistance", label: "眼距", short: "距", value: 0, hint: "比例微调" },
      { id: "eyeAngle", label: "眼角", short: "角", value: 0, hint: "轻微调整" },
    ],
  },
  features: {
    title: "五官细节",
    tools: [
      { id: "noseBridge", label: "立体鼻梁", short: "鼻", value: 0, hint: "自然高光" },
      { id: "slimNose", label: "瘦鼻", short: "窄", value: 0, hint: "鼻翼收窄" },
      { id: "teeth", label: "牙齿美白", short: "牙", value: 0, hint: "中性白" },
      { id: "smile", label: "自然微笑", short: "笑", value: 0, hint: "轻调表情" },
      { id: "lipVolume", label: "唇形", short: "唇", value: 0, hint: "丰盈度" },
      { id: "lipColor", label: "唇色", short: "色", value: 0, hint: "自然气色" },
      { id: "brows", label: "眉形", short: "眉", value: 0, hint: "提升精神" },
    ],
  },
  body: {
    title: "身形塑造",
    tools: [
      { id: "slimBelly", label: "瘦肚子", short: "腹", value: 0, hint: "收紧腹部" },
      { id: "slimWaist", label: "细腰", short: "腰", value: 0, hint: "保护背景" },
      { id: "slimArms", label: "瘦手臂", short: "臂", value: 0, hint: "双侧联动" },
      { id: "slimLegs", label: "瘦腿", short: "腿", value: 0, hint: "自然线条" },
      { id: "longLegs", label: "长腿", short: "长", value: 0, hint: "比例优化" },
      { id: "shoulders", label: "肩宽", short: "肩", value: 0, hint: "轮廓微调" },
      { id: "hips", label: "提臀", short: "臀", value: 0, hint: "自然曲线" },
      { id: "headRatio", label: "头身比", short: "比", value: 0, hint: "整体比例" },
    ],
  },
  makeup: {
    title: "自然妆容",
    tools: [
      { id: "foundation", label: "底妆", short: "底", value: 0, hint: "轻薄服帖" },
      { id: "blush", label: "腮红", short: "腮", value: 0, hint: "提升气色" },
      { id: "contour", label: "修容", short: "修", value: 0, hint: "自然立体" },
      { id: "eyebrowMakeup", label: "眉妆", short: "眉", value: 0, hint: "柔和填充" },
      { id: "eyeliner", label: "眼线", short: "线", value: 0, hint: "眼神增强" },
      { id: "lipstick", label: "口红", short: "唇", value: 0, hint: "自然色泽" },
    ],
  },
  tone: {
    title: "整体氛围",
    tools: [
      { id: "brightness", label: "亮度", short: "亮", value: 0, hint: "整体曝光" },
      { id: "contrast", label: "对比度", short: "对", value: 0, hint: "明暗层次" },
      { id: "warmth", label: "色温", short: "温", value: 0, hint: "冷暖调节" },
      { id: "saturation", label: "饱和度", short: "饱", value: 0, hint: "色彩浓度" },
      { id: "clarity", label: "清晰度", short: "清", value: 0, hint: "细节增强" },
      { id: "background", label: "背景净化", short: "景", value: 0, hint: "弱化干扰" },
    ],
  },
};

const state = {
  category: "skin",
  mode: "local",
  values: Object.fromEntries(
    Object.values(categories)
      .flatMap((category) => category.tools)
      .map((tool) => [tool.id, tool.value]),
  ),
  history: [],
  selectedPreset: "natural",
  originalOnly: false,
  sourceFile: null,
  capabilities: {
    brainProviders: [],
    actionProviders: [],
  },
  aiResultImage: null,
  displayingAiResult: false,
  aiBusy: false,
};

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
};

const aiApi = {
  uploadPhoto,
  analyzePhoto,
  createPlans,
  createJob,
};

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
  const availableIds = new Set(providers.map((provider) => provider.id));
  select.value = availableIds.has(previous)
    ? previous
    : availableIds.has(preferredId)
      ? preferredId
      : providers[0]?.id || "";
}

function selectedProvider(layer) {
  const isBrain = layer === "brain";
  const providers = isBrain
    ? state.capabilities.brainProviders
    : state.capabilities.actionProviders;
  const select = isBrain
    ? elements.brainProviderSelect
    : elements.actionProviderSelect;
  return providers.find((provider) => provider.id === select.value);
}

function modelRequest(layer) {
  const isBrain = layer === "brain";
  const providerName = isBrain
    ? elements.brainProviderSelect.value
    : elements.actionProviderSelect.value;
  const directApiKey = (
    isBrain ? elements.brainApiKeyInput.value : elements.actionApiKeyInput.value
  ).trim();
  const canReuseBrainKey =
    !isBrain &&
    ((elements.brainProviderSelect.value === "qwen" &&
      ["qwen", "wan"].includes(providerName)) ||
      (elements.brainProviderSelect.value === "doubao" &&
        providerName === "seedream"));
  return {
    name: providerName,
    apiKey:
      directApiKey ||
      (canReuseBrainKey ? elements.brainApiKeyInput.value.trim() : ""),
    workspaceId: (
      isBrain
        ? elements.brainWorkspaceInput.value
        : elements.actionWorkspaceInput.value
    ).trim(),
  };
}

function syncModelConfig() {
  const brain = selectedProvider("brain");
  const action = selectedProvider("action");
  elements.brainApiKeyField.hidden = !brain?.requiresApiKey;
  elements.actionApiKeyField.hidden = !action?.requiresApiKey;
  elements.brainWorkspaceField.hidden = !brain?.workspaceSupported;
  elements.actionWorkspaceField.hidden = !action?.workspaceSupported;
  elements.brainApiKeyInput.placeholder = brain?.configured
    ? "服务端已配置，可留空"
    : "请输入本次会话 Key";
  elements.actionApiKeyInput.placeholder = action?.configured
    ? "服务端已配置，可留空"
    : "可留空复用同平台视觉 Key";

  const brainRequest = modelRequest("brain");
  const actionRequest = modelRequest("action");
  const ready =
    brain &&
    action &&
    (!brain.requiresApiKey || brain.configured || brainRequest.apiKey) &&
    (!action.requiresApiKey || action.configured || actionRequest.apiKey);
  elements.modelConfigState.textContent = ready ? "已就绪" : "需要配置";
  elements.modelConfigState.dataset.state = ready ? "ready" : "error";
}

async function loadProviderCapabilities() {
  try {
    state.capabilities = await getProviderCapabilities();
    populateProviderSelect(
      elements.brainProviderSelect,
      state.capabilities.brainProviders || [],
      "qwen",
    );
    populateProviderSelect(
      elements.actionProviderSelect,
      state.capabilities.actionProviders || [],
      "qwen",
    );
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
  const response = await fetch(elements.beforeImage.currentSrc || elements.beforeImage.src);
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
  const context = elements.previewCanvas.getContext("2d", { alpha: false });
  context.imageSmoothingEnabled = true;
  context.imageSmoothingQuality = "high";
  context.drawImage(image, 0, 0, width, height);
}

function showAiResult(url) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.addEventListener("load", () => {
      state.aiResultImage = image;
      state.displayingAiResult = true;
      drawImageToPreview(image);
      resolve();
    });
    image.addEventListener("error", () => reject(new Error("AI 结果图片加载失败。")));
    image.src = url;
  });
}

function currentTools() {
  const query = elements.toolSearch.value.trim().toLowerCase();
  const tools = categories[state.category].tools;
  if (!query) return tools;
  return tools.filter((tool) => `${tool.label}${tool.hint}`.toLowerCase().includes(query));
}

function renderCategory() {
  const category = categories[state.category];
  elements.categoryTitle.textContent = category.title;
  const tools = currentTools();

  elements.quickTags.innerHTML = tools
    .map(
      (tool) => `
        <button
          class="tool-tag ${state.values[tool.id] > 0 ? "active" : ""}"
          data-tool-tag="${tool.id}"
          type="button"
        >
          ${tool.label}
        </button>
      `,
    )
    .join("");

  elements.adjustmentList.innerHTML = tools
    .map((tool) => {
      const value = state.values[tool.id];
      return `
        <label class="adjustment">
          <span class="adjustment-head">
            <span class="adjustment-label">
              <i>${tool.short}</i>
              <span>${tool.label}</span>
            </span>
            <b class="adjustment-value" data-value-for="${tool.id}">${value}</b>
          </span>
          <span class="slider-row">
            <span>−</span>
            <input
              type="range"
              min="0"
              max="100"
              value="${value}"
              data-tool-range="${tool.id}"
              style="--range-progress: ${value}%"
            />
            <span>＋</span>
          </span>
        </label>
      `;
    })
    .join("");

  elements.quickTags.querySelectorAll("[data-tool-tag]").forEach((button) => {
    button.addEventListener("click", () => {
      const id = button.dataset.toolTag;
      const previous = state.values[id];
      state.history.push({ id, value: previous });
      state.values[id] = previous > 0 ? 0 : 20;
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
      state.values[id] = Number(input.value);
      input.style.setProperty("--range-progress", `${input.value}%`);
      elements.adjustmentList.querySelector(`[data-value-for="${id}"]`).textContent =
        input.value;
      const tag = elements.quickTags.querySelector(`[data-tool-tag="${id}"]`);
      tag?.classList.toggle("active", Number(input.value) > 0);
      updatePreview();
    });
    input.addEventListener("change", () => {
      const startValue = Number(input.dataset.startValue ?? 0);
      if (startValue !== Number(input.value)) {
        state.history.push({ id: input.dataset.toolRange, value: startValue });
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
  const toolMap = Object.fromEntries(
    Object.values(categories)
      .flatMap((category) => category.tools)
      .map((tool) => [tool.id, tool]),
  );
  const changes = Object.entries(state.values)
    .filter(([, value]) => value > 0)
    .sort((a, b) => b[1] - a[1]);

  elements.changeCount.textContent = `${changes.length} 项`;
  elements.changeList.innerHTML = changes.length
    ? changes
        .slice(0, 5)
        .map(([id, value]) => {
          const tool = toolMap[id];
          return `
            <div class="change-item">
              <span>${tool.short}</span>
              <div>
                <strong>${tool.label}</strong>
                <small>${tool.hint}</small>
              </div>
              <b>+${value}</b>
            </div>
          `;
        })
        .join("")
    : `
        <div class="change-item">
          <span>原</span>
          <div><strong>尚未调整</strong><small>选择左侧标签开始美化</small></div>
          <b>0</b>
        </div>
      `;

  elements.historyState.textContent = changes.length
    ? changes
        .slice(0, 4)
        .map(([id, value]) => `${toolMap[id].label} ${value}`)
        .join(" · ")
    : "尚未添加美化参数";
}

function setMode(mode) {
  state.mode = mode;
  document.querySelectorAll("[data-mode]").forEach((button) => {
    button.classList.toggle("active", button.dataset.mode === mode);
  });
  const isLocal = mode === "local";
  elements.localPanel.hidden = !isLocal;
  elements.aiPanel.hidden = isLocal;
  elements.modeNote.innerHTML = isLocal
    ? '<span class="local-dot"></span>本地实时预览 · 图片无需上传模型'
    : '<span class="local-dot"></span>云端 AI 精修 · 生成前会确认参数';
  elements.canvasStatus.classList.toggle("ai", !isLocal);
  elements.canvasStatus.innerHTML = isLocal
    ? "<span></span>实时预览已开启"
    : state.displayingAiResult
      ? "<span></span>AI 美化结果"
      : "<span></span>AI 模式 · 当前显示手动预览";
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
  tools.forEach((tool) => {
    snapshot[tool.id] = state.values[tool.id];
    state.values[tool.id] = 0;
  });
  state.history.push({ snapshot });
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
  Object.assign(state.values, presets[preset]);
  updatePreview();
}

document.querySelectorAll("[data-category]").forEach((button) => {
  button.addEventListener("click", () => {
    state.category = button.dataset.category;
    elements.toolSearch.value = "";
    document.querySelectorAll("[data-category]").forEach((item) => {
      item.classList.toggle("active", item === button);
    });
    renderCategory();
  });
});

document.querySelectorAll("[data-mode]").forEach((button) => {
  button.addEventListener("click", () => setMode(button.dataset.mode));
});

elements.toolSearch.addEventListener("input", renderCategory);

elements.compareRange.addEventListener("input", () => {
  const value = Number(elements.compareRange.value);
  elements.afterLayer.style.clipPath = `inset(0 0 0 ${value}%)`;
  elements.compareLine.style.left = `${value}%`;
  elements.compareRange.style.setProperty("--range-progress", `${value}%`);
});

elements.compareButton.addEventListener("click", () => {
  state.originalOnly = !state.originalOnly;
  elements.afterLayer.style.opacity = state.originalOnly ? "0" : "1";
  elements.compareLine.hidden = state.originalOnly;
  elements.compareButton.lastChild.textContent = state.originalOnly
    ? " 显示美化"
    : " 查看原图";
});

elements.resetCategoryButton.addEventListener("click", () => {
  resetTools(categories[state.category].tools);
});

elements.resetButton.addEventListener("click", () => {
  resetTools(Object.values(categories).flatMap((category) => category.tools));
});

elements.undoButton.addEventListener("click", () => {
  const previous = state.history.pop();
  if (!previous) return;
  if (previous.snapshot) {
    Object.assign(state.values, previous.snapshot);
  } else {
    state.values[previous.id] = previous.value;
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
  state.aiResultImage = null;
  state.displayingAiResult = false;
  elements.beforeImage.src = url;
  elements.exportButton.href = url;
  document.querySelector(".document-name > span:nth-child(2)").textContent = file.name;
  elements.footerStatus.textContent = "新照片已载入";
});

elements.beforeImage.addEventListener("error", () => {
  const fallback = "/data/outputs/out_de5f8e95e54a.png";
  if (!elements.beforeImage.src.endsWith(fallback)) {
    elements.beforeImage.src = fallback;
    elements.exportButton.href = fallback;
  }
});

elements.beforeImage.addEventListener("load", scheduleLocalRender);

document.querySelectorAll("[data-preset]").forEach((button) => {
  button.addEventListener("click", () => {
    state.selectedPreset = button.dataset.preset;
    document.querySelectorAll("[data-preset]").forEach((item) => {
      item.classList.toggle("active", item === button);
    });
  });
});

elements.aiPrompt.addEventListener("input", () => {
  elements.promptCount.textContent = String(elements.aiPrompt.value.length);
});

elements.brainProviderSelect.addEventListener("change", syncModelConfig);
elements.actionProviderSelect.addEventListener("change", syncModelConfig);
elements.brainApiKeyInput.addEventListener("input", syncModelConfig);
elements.actionApiKeyInput.addEventListener("input", syncModelConfig);

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
      file,
      values: state.values,
      preset: state.selectedPreset,
      userInstruction: elements.aiPrompt.value,
      brain,
      action,
      api: aiApi,
    });
    const resultUrl = assetUrl(result.job.outputUrls[0]);
    await showAiResult(resultUrl);
    elements.exportButton.href = resultUrl;
    elements.canvasStatus.innerHTML =
      `<span></span>AI 结果 · ${result.job.modelName}`;
    elements.footerStatus.textContent = "AI 美化结果已生成";
    elements.aiFeedback.textContent =
      `${result.job.brainModel} + ${result.job.modelName} · 生成成功`;
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

elements.floatingTip.querySelector("button").addEventListener("click", () => {
  elements.floatingTip.hidden = true;
});

renderCategory();
updatePreview();
elements.compareRange.dispatchEvent(new Event("input"));
if (elements.beforeImage.complete) scheduleLocalRender();
loadProviderCapabilities();
