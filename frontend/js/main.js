import {
  analyzePhoto,
  assetUrl,
  createJob,
  createPlans,
  getProviderCapabilities,
  refineJob,
  uploadPhoto,
} from "./api.js?v=0.4.0";

const fallbackCapabilities = {
  brainProviders: [
    { id: "qwen", label: "Qwen Vision", model: "qwen3-vl-plus", configured: false, visionMode: "direct", requiresApiKey: true, workspaceSupported: true },
    { id: "glm", label: "GLM Vision", model: "glm-5v-turbo", configured: false, visionMode: "direct", requiresApiKey: true, workspaceSupported: false },
    { id: "doubao", label: "豆包 Vision", model: "doubao-seed-2.0", configured: false, visionMode: "direct", requiresApiKey: true, workspaceSupported: false },
    { id: "deepseek", label: "DeepSeek", model: "deepseek-v4-flash", configured: false, visionMode: "derived", requiresApiKey: true, workspaceSupported: false },
    { id: "local", label: "本地规则", model: "rule-based-planner", configured: true, visionMode: "derived", requiresApiKey: false, workspaceSupported: false },
  ],
  actionProviders: [
    { id: "qwen", label: "Qwen Image", model: "qwen-image-2.0-pro", configured: false, requiresApiKey: true, workspaceSupported: true },
    { id: "wan", label: "Wan Image", model: "wan2.7-image-pro", configured: false, requiresApiKey: true, workspaceSupported: true },
    { id: "seedream", label: "Seedream", model: "doubao-seedream-5.0", configured: false, requiresApiKey: true, workspaceSupported: false },
    { id: "mock", label: "本地模拟", model: "pillow-retouch-mock", configured: true, requiresApiKey: false, workspaceSupported: false },
  ],
};

const state = {
  photo: null,
  analysis: null,
  plans: [],
  selectedPlan: null,
  currentJob: null,
  history: [],
  isBusy: false,
  capabilities: fallbackCapabilities,
  brain: {
    name: "",
    apiKey: "",
    workspaceId: "",
  },
  action: {
    name: "",
    apiKey: "",
    workspaceId: "",
  },
};

const $ = (selector) => document.querySelector(selector);

const els = {
  status: $("#status"),
  brainProviderSelect: $("#brainProviderSelect"),
  brainApiKeyField: $("#brainApiKeyField"),
  brainApiKeyInput: $("#brainApiKeyInput"),
  brainWorkspaceField: $("#brainWorkspaceField"),
  brainWorkspaceInput: $("#brainWorkspaceInput"),
  brainProviderState: $("#brainProviderState"),
  brainProviderHint: $("#brainProviderHint"),
  actionProviderSelect: $("#actionProviderSelect"),
  actionApiKeyField: $("#actionApiKeyField"),
  actionApiKeyInput: $("#actionApiKeyInput"),
  actionWorkspaceField: $("#actionWorkspaceField"),
  actionWorkspaceInput: $("#actionWorkspaceInput"),
  actionProviderState: $("#actionProviderState"),
  actionProviderHint: $("#actionProviderHint"),
  workflowGate: $("#workflowGate"),
  fileInput: $("#fileInput"),
  uploadButton: $("#uploadButton"),
  sourceImage: $("#sourceImage"),
  sourceEmpty: $("#sourceEmpty"),
  photoMeta: $("#photoMeta"),
  analyzeButton: $("#analyzeButton"),
  analysisEmpty: $("#analysisEmpty"),
  analysisContent: $("#analysisContent"),
  sceneType: $("#sceneType"),
  subjectSummary: $("#subjectSummary"),
  riskSummary: $("#riskSummary"),
  analysisModelSummary: $("#analysisModelSummary"),
  analysisGroups: $("#analysisGroups"),
  planCount: $("#planCount"),
  planList: $("#planList"),
  resultImage: $("#resultImage"),
  resultEmpty: $("#resultEmpty"),
  executionNote: $("#executionNote"),
  regenerateButton: $("#regenerateButton"),
  downloadLink: $("#downloadLink"),
  history: $("#history"),
  refineForm: $("#refineForm"),
  instructionInput: $("#instructionInput"),
  refineButton: $("#refineButton"),
};

els.uploadButton.addEventListener("click", () => els.fileInput.click());
els.fileInput.addEventListener("change", handleUpload);
els.analyzeButton.addEventListener("click", runAnalysis);
els.regenerateButton.addEventListener("click", () => {
  if (state.photo && state.selectedPlan) runRetouch(state.selectedPlan, "");
});
els.refineForm.addEventListener("submit", handleRefine);
els.brainProviderSelect.addEventListener("change", () => {
  handleModelSelection("brain", els.brainProviderSelect.value);
});
els.actionProviderSelect.addEventListener("change", () => {
  handleModelSelection("action", els.actionProviderSelect.value);
});
els.brainApiKeyInput.addEventListener("input", () => {
  state.brain.apiKey = els.brainApiKeyInput.value;
  renderModelControls();
});
els.actionApiKeyInput.addEventListener("input", () => {
  state.action.apiKey = els.actionApiKeyInput.value;
  renderModelControls();
});
els.brainWorkspaceInput.addEventListener("input", () => {
  state.brain.workspaceId = els.brainWorkspaceInput.value;
});
els.actionWorkspaceInput.addEventListener("input", () => {
  state.action.workspaceId = els.actionWorkspaceInput.value;
});

initializeModelControls();

async function handleUpload(event) {
  const file = event.target.files?.[0];
  if (!file || !workflowReady()) return;

  await runTask("上传中", async () => {
    const photo = await uploadPhoto(file);
    state.photo = photo;
    state.analysis = null;
    state.plans = [];
    state.selectedPlan = null;
    state.currentJob = null;
    state.history = [`上传 ${file.name}`];
    renderPhoto(photo);
    resetAnalysis();
    resetResult();
    renderHistory();
    setStatus("已上传");
  });
}

async function runAnalysis() {
  if (!state.photo || !workflowReady()) return;

  await runTask("识别与分析中", async () => {
    state.analysis = await analyzePhoto(
      state.photo.imageId,
      brainRequest(),
      actionRequest(),
    );
    state.plans = await createPlans(state.analysis);
    state.selectedPlan = state.plans[0];
    state.history.push(`${state.analysis.sceneType} · ${state.plans.length} 个方案`);
    renderAnalysis();
    renderPlans();
    renderHistory();
    setStatus("方案就绪");
  });
}

async function runRetouch(plan, instruction) {
  if (!state.photo || !plan || !workflowReady()) return;

  await runTask(instruction ? "修改中" : "生成中", async () => {
    const job = await createJob(
      state.photo.imageId,
      plan,
      instruction,
      brainRequest(),
      actionRequest(),
    );
    state.currentJob = job;
    state.selectedPlan = plan;
    state.history.push(instruction ? `修改：${instruction}` : `生成：${plan.title}`);
    renderResult(job, plan, instruction);
    renderHistory();
    setStatus("结果完成");
  });
}

async function handleRefine(event) {
  event.preventDefault();
  const instruction = els.instructionInput.value.trim();
  if (!instruction || !state.currentJob || !workflowReady()) return;

  await runTask("修改中", async () => {
    const job = await refineJob(
      state.currentJob.jobId,
      instruction,
      brainRequest(),
      actionRequest(),
    );
    state.currentJob = job;
    state.history.push(`修改：${instruction}`);
    renderResult(job, job.plan || state.selectedPlan, instruction);
    renderHistory();
    els.instructionInput.value = "";
    setStatus("结果完成");
  });
}

function handleModelSelection(layer, name) {
  if (state[layer].name === name) return;
  state[layer].name = name;
  state[layer].apiKey = "";
  state[layer].workspaceId = "";

  const apiKeyInput = layer === "brain" ? els.brainApiKeyInput : els.actionApiKeyInput;
  const workspaceInput = layer === "brain" ? els.brainWorkspaceInput : els.actionWorkspaceInput;
  apiKeyInput.value = "";
  workspaceInput.value = "";
  invalidateModelDependentState();
  renderModelControls();
  setStatus(workflowReady() ? "模型已就绪" : "请选择模型");
}

function invalidateModelDependentState() {
  if (!state.photo || (!state.analysis && !state.currentJob)) return;
  state.analysis = null;
  state.plans = [];
  state.selectedPlan = null;
  state.currentJob = null;
  state.history.push("模型已变更");
  resetAnalysis();
  resetResult();
  renderHistory();
}

function renderPhoto(photo) {
  els.sourceImage.src = assetUrl(photo.url);
  els.sourceImage.hidden = false;
  els.sourceEmpty.hidden = true;
  setMeta([
    photo.imageId,
    `${photo.width} × ${photo.height}`,
    formatBytes(photo.sizeBytes),
    photo.contentType,
  ]);
}

function resetAnalysis() {
  els.analysisEmpty.hidden = false;
  els.analysisEmpty.textContent = "待分析";
  els.analysisContent.hidden = true;
  els.planCount.textContent = "0";
  els.planList.innerHTML = "";
}

function resetResult() {
  els.resultImage.hidden = true;
  els.resultEmpty.hidden = false;
  els.executionNote.textContent = "--";
  els.downloadLink.classList.add("disabled");
  els.downloadLink.removeAttribute("href");
}

function renderAnalysis() {
  const analysis = state.analysis;
  els.analysisEmpty.hidden = true;
  els.analysisContent.hidden = false;
  els.sceneType.textContent = analysis.sceneType;
  els.subjectSummary.textContent = `${analysis.subjects.count} 个 · ${analysis.subjects.faceVisibility}`;
  els.riskSummary.textContent = analysis.riskFlags.length ? `${analysis.riskFlags.length} 项` : "低";
  els.analysisModelSummary.textContent = `${analysis.brainProvider} / ${analysis.brainModel}`;

  const groups = [
    ["光线问题", analysis.lightingIssues, "amber"],
    ["背景问题", analysis.backgroundIssues, "blue"],
    ["主体优化", analysis.portraitSuggestions, ""],
    ["构图建议", analysis.compositionSuggestions, "blue"],
    ["风格建议", analysis.recommendedStyles, ""],
    ["风险标记", analysis.riskFlags.length ? analysis.riskFlags : ["暂无明显风险"], analysis.riskFlags.length ? "rose" : ""],
  ];
  els.analysisGroups.innerHTML = groups.map(renderGroup).join("");
}

function renderPlans() {
  const ready = workflowReady() && !state.isBusy;
  els.planCount.textContent = String(state.plans.length);
  els.planList.innerHTML = state.plans
    .map(
      (plan) => `
        <article class="plan">
          <div>
            <strong>${escapeHtml(plan.title)}</strong>
            <p>${escapeHtml(plan.description)}</p>
            <div class="tags">
              ${plan.expectedChanges.map((item) => `<span class="tag">${escapeHtml(item)}</span>`).join("")}
            </div>
          </div>
          <button type="button" data-plan-id="${escapeHtml(plan.planId)}" ${ready ? "" : "disabled"}>生成</button>
        </article>
      `,
    )
    .join("");

  els.planList.querySelectorAll("button[data-plan-id]").forEach((button) => {
    button.addEventListener("click", () => {
      const plan = state.plans.find((item) => item.planId === button.dataset.planId);
      runRetouch(plan, "");
    });
  });
}

function renderResult(job, plan, instruction) {
  const outputUrl = job.outputUrls?.[0];
  if (!outputUrl) throw new Error(job.errorMessage || "生成结果为空");

  const absoluteUrl = assetUrl(outputUrl);
  els.resultImage.src = absoluteUrl;
  els.resultImage.hidden = false;
  els.resultEmpty.hidden = true;
  els.downloadLink.href = absoluteUrl;
  els.downloadLink.download = `${job.outputImageIds[0]}.png`;
  els.downloadLink.classList.remove("disabled");
  els.executionNote.textContent = instruction
    ? `二次修改：${instruction}。基于上一版结果继续生成。`
    : (
      `方案：${plan?.title || job.planId}。` +
      `大脑：${job.brainProvider}/${job.brainModel}；` +
      `行动：${job.modelProvider}/${job.modelName}。`
    );
}

function renderGroup([title, items, tone]) {
  return `
    <section class="group">
      <h3>${escapeHtml(title)}</h3>
      <div class="tags">
        ${items.map((item) => `<span class="tag ${tone}">${escapeHtml(item)}</span>`).join("")}
      </div>
    </section>
  `;
}

function renderHistory() {
  els.history.textContent = state.history.length ? state.history.slice(-4).join(" · ") : "等待任务";
}

async function initializeModelControls() {
  try {
    state.capabilities = await getProviderCapabilities();
  } catch (error) {
    state.capabilities = fallbackCapabilities;
  }
  populateModelOptions(
    els.brainProviderSelect,
    state.capabilities.brainProviders || fallbackCapabilities.brainProviders,
    "brain",
  );
  populateModelOptions(
    els.actionProviderSelect,
    state.capabilities.actionProviders || fallbackCapabilities.actionProviders,
    "action",
  );
  renderModelControls();
  setStatus("请选择模型");
}

function populateModelOptions(select, providers, layer) {
  const placeholder = document.createElement("option");
  placeholder.value = "";
  placeholder.textContent = "请选择";
  select.replaceChildren(placeholder);

  providers.forEach((provider) => {
    const option = document.createElement("option");
    option.value = provider.id;
    const suffix = layer === "brain" && provider.visionMode === "derived"
      ? "（分析辅助）"
      : provider.id === "mock"
        ? "（演示）"
        : "";
    option.textContent = `${provider.label} · ${provider.model}${suffix}`;
    select.append(option);
  });
}

function renderModelControls() {
  renderLayerControls("brain");
  renderLayerControls("action");

  const ready = workflowReady();
  const brainMeta = selectedModel("brain");
  const actionMeta = selectedModel("action");
  els.workflowGate.dataset.ready = String(ready);
  els.workflowGate.textContent = ready
    ? `${brainMeta.label} + ${actionMeta.label} 已就绪`
    : workflowGateMessage();
  syncControlState();
  if (state.plans.length) renderPlans();
}

function renderLayerControls(layer) {
  const config = state[layer];
  const meta = selectedModel(layer);
  const isBrain = layer === "brain";
  const apiKeyField = isBrain ? els.brainApiKeyField : els.actionApiKeyField;
  const workspaceField = isBrain ? els.brainWorkspaceField : els.actionWorkspaceField;
  const stateLabel = isBrain ? els.brainProviderState : els.actionProviderState;
  const hint = isBrain ? els.brainProviderHint : els.actionProviderHint;

  apiKeyField.hidden = !meta || !meta.requiresApiKey;
  workspaceField.hidden = !meta || !meta.workspaceSupported;

  if (!meta) {
    stateLabel.textContent = "待选择";
    hint.textContent = "--";
    return;
  }

  if (!meta.requiresApiKey) {
    stateLabel.textContent = "可用";
  } else if (config.apiKey.trim()) {
    stateLabel.textContent = "使用本次会话 Key";
  } else if (meta.configured) {
    stateLabel.textContent = "服务端已配置";
  } else {
    stateLabel.textContent = "需要 API Key";
  }

  if (isBrain) {
    hint.textContent = meta.visionMode === "direct"
      ? `${meta.model} · 直接视觉`
      : `${meta.model} · 结构化分析辅助`;
  } else {
    hint.textContent = `${meta.model} · 图像编辑`;
  }
}

function selectedModel(layer) {
  const key = layer === "brain" ? "brainProviders" : "actionProviders";
  return (state.capabilities[key] || []).find((item) => item.id === state[layer].name) || null;
}

function layerReady(layer) {
  const meta = selectedModel(layer);
  if (!meta) return false;
  if (!meta.requiresApiKey) return true;
  return Boolean(state[layer].apiKey.trim() || meta.configured);
}

function workflowReady() {
  return layerReady("brain") && layerReady("action");
}

function workflowGateMessage() {
  if (!state.brain.name && !state.action.name) {
    return "请选择 Agent 大脑和 Agent 行动";
  }
  if (!state.brain.name) return "请选择 Agent 大脑";
  if (!state.action.name) return "请选择 Agent 行动";
  if (!layerReady("brain")) return "Agent 大脑需要 API Key";
  if (!layerReady("action")) return "Agent 行动需要 API Key";
  return "模型配置未就绪";
}

function brainRequest() {
  return {
    name: state.brain.name,
    apiKey: state.brain.apiKey.trim(),
    workspaceId: state.brain.workspaceId.trim(),
  };
}

function actionRequest() {
  return {
    name: state.action.name,
    apiKey: state.action.apiKey.trim(),
    workspaceId: state.action.workspaceId.trim(),
  };
}

function setMeta(values) {
  [...els.photoMeta.querySelectorAll("dd")].forEach((node, index) => {
    node.textContent = values[index] || "--";
  });
}

async function runTask(label, task) {
  if (!workflowReady()) {
    setStatus(workflowGateMessage());
    return;
  }
  setBusy(true);
  setStatus(label);
  try {
    await task();
  } catch (error) {
    setStatus(error.message || "操作失败");
  } finally {
    setBusy(false);
  }
}

function setBusy(isBusy) {
  state.isBusy = isBusy;
  syncControlState();
  if (state.plans.length) renderPlans();
}

function syncControlState() {
  const ready = workflowReady();
  const disabled = state.isBusy;
  els.uploadButton.disabled = disabled || !ready;
  els.analyzeButton.disabled = disabled || !ready || !state.photo;
  els.regenerateButton.disabled = disabled || !ready || !state.currentJob;
  els.refineButton.disabled = disabled || !ready || !state.currentJob;
  els.instructionInput.disabled = disabled || !ready || !state.currentJob;
  els.brainProviderSelect.disabled = disabled;
  els.brainApiKeyInput.disabled = disabled;
  els.brainWorkspaceInput.disabled = disabled;
  els.actionProviderSelect.disabled = disabled;
  els.actionApiKeyInput.disabled = disabled;
  els.actionWorkspaceInput.disabled = disabled;
}

function setStatus(text) {
  els.status.textContent = text;
}

function formatBytes(bytes) {
  if (!bytes) return "--";
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}
