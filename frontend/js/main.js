import {
  analyzePhoto,
  assetUrl,
  createJob,
  createPlans,
  getProviderCapabilities,
  refineJob,
  uploadPhoto,
} from "./api.js?v=0.3.0";

const state = {
  photo: null,
  analysis: null,
  plans: [],
  selectedPlan: null,
  currentJob: null,
  history: [],
  provider: {
    name: "auto",
    apiKey: "",
    workspaceId: "",
    capabilities: null,
  },
  brain: {
    name: "local",
    apiKey: "",
  },
};

const $ = (selector) => document.querySelector(selector);

const els = {
  status: $("#status"),
  brainProviderSelect: $("#brainProviderSelect"),
  brainApiKeyField: $("#brainApiKeyField"),
  brainApiKeyInput: $("#brainApiKeyInput"),
  brainProviderState: $("#brainProviderState"),
  brainProviderHint: $("#brainProviderHint"),
  providerSelect: $("#providerSelect"),
  apiKeyField: $("#apiKeyField"),
  apiKeyInput: $("#apiKeyInput"),
  workspaceField: $("#workspaceField"),
  workspaceInput: $("#workspaceInput"),
  providerState: $("#providerState"),
  providerHint: $("#providerHint"),
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
els.providerSelect.addEventListener("change", () => {
  state.provider.name = els.providerSelect.value;
  renderProviderControls();
});
els.apiKeyInput.addEventListener("input", () => {
  state.provider.apiKey = els.apiKeyInput.value;
  renderProviderControls();
});
els.workspaceInput.addEventListener("input", () => {
  state.provider.workspaceId = els.workspaceInput.value;
});
els.brainProviderSelect.addEventListener("change", () => {
  state.brain.name = els.brainProviderSelect.value;
  renderBrainControls();
});
els.brainApiKeyInput.addEventListener("input", () => {
  state.brain.apiKey = els.brainApiKeyInput.value;
  renderBrainControls();
});

initializeProviderControls();

async function handleUpload(event) {
  const file = event.target.files?.[0];
  if (!file) return;

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
    els.analyzeButton.disabled = false;
    setStatus("已上传");
  });
}

async function runAnalysis() {
  if (!state.photo) return;

  await runTask("分析中", async () => {
    state.analysis = await analyzePhoto(state.photo.imageId);
    state.plans = await createPlans(state.analysis);
    state.selectedPlan = state.plans[0];
    state.history.push("生成 3 个方案");
    renderAnalysis();
    renderPlans();
    renderHistory();
    setStatus("方案就绪");
  });
}

async function runRetouch(plan, instruction) {
  if (!state.photo || !plan) return;

  await runTask(instruction ? "修改中" : "生成中", async () => {
    const job = await createJob(
      state.photo.imageId,
      plan,
      instruction,
      providerRequest(),
      brainRequest(),
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
  if (!instruction || !state.currentJob) return;

  await runTask("修改中", async () => {
    const job = await refineJob(
      state.currentJob.jobId,
      instruction,
      providerRequest(),
      brainRequest(),
    );
    state.currentJob = job;
    state.history.push(`修改：${instruction}`);
    renderResult(job, job.plan || state.selectedPlan, instruction);
    renderHistory();
    els.instructionInput.value = "";
    setStatus("结果完成");
  });
}

function renderPhoto(photo) {
  els.sourceImage.src = assetUrl(photo.url);
  els.sourceImage.hidden = false;
  els.sourceEmpty.hidden = true;
  setMeta([photo.imageId, `${photo.width} × ${photo.height}`, formatBytes(photo.sizeBytes), photo.contentType]);
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
  els.regenerateButton.disabled = true;
  els.downloadLink.classList.add("disabled");
  els.downloadLink.removeAttribute("href");
  els.instructionInput.disabled = true;
  els.refineButton.disabled = true;
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
  els.analysisGroups.innerHTML = groups.map(renderGroup).join("");
}

function renderPlans() {
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
          <button type="button" data-plan-id="${escapeHtml(plan.planId)}">生成</button>
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
  els.regenerateButton.disabled = false;
  els.downloadLink.href = absoluteUrl;
  els.downloadLink.download = `${job.outputImageIds[0]}.jpg`;
  els.downloadLink.classList.remove("disabled");
  els.instructionInput.disabled = false;
  els.refineButton.disabled = false;
  els.executionNote.textContent = instruction
    ? `二次修改：${instruction}。基于上一版结果继续生成。`
    : (
      `方案：${plan?.title || job.planId}。` +
      `Agent：${job.brainProvider}/${job.brainModel}；` +
      `修图：${job.modelProvider}/${job.modelName}。`
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

async function initializeProviderControls() {
  try {
    state.provider.capabilities = await getProviderCapabilities();
  } catch (error) {
    state.provider.capabilities = null;
  }
  renderProviderControls();
  renderBrainControls();
}

function renderProviderControls() {
  const isMock = state.provider.name === "mock";
  const capabilities = state.provider.capabilities;
  els.apiKeyField.hidden = isMock;
  els.workspaceField.hidden = isMock;

  if (isMock) {
    els.providerState.textContent = "本地 Pillow 模拟";
    els.providerHint.textContent = "不调用外部模型，不需要 API Key";
    return;
  }

  const hasUserKey = Boolean(state.provider.apiKey.trim());
  const serverConfigured = Boolean(capabilities?.qwenConfigured);
  if (hasUserKey) {
    els.providerState.textContent = "使用本次会话 API Key";
  } else if (serverConfigured) {
    els.providerState.textContent = `服务端已配置 ${capabilities.qwenModel}`;
  } else if (state.provider.name === "qwen") {
    els.providerState.textContent = "需要 Qwen API Key";
  } else {
    els.providerState.textContent = "无 Key 时使用本地模拟";
  }
  els.providerHint.textContent = "API Key 仅随生成请求发送，不会写入任务记录";
}

function providerRequest() {
  return {
    name: state.provider.name,
    apiKey: state.provider.apiKey.trim(),
    workspaceId: state.provider.workspaceId.trim(),
  };
}

function renderBrainControls() {
  const isLocal = state.brain.name === "local";
  const capabilities = state.provider.capabilities;
  els.brainApiKeyField.hidden = isLocal;

  if (isLocal) {
    els.brainProviderState.textContent = "本地规则规划";
    els.brainProviderHint.textContent = "不调用外部文本模型";
    return;
  }

  const hasUserKey = Boolean(state.brain.apiKey.trim());
  const isDeepSeek = state.brain.name === "deepseek";
  const serverConfigured = isDeepSeek
    ? Boolean(capabilities?.deepseekConfigured)
    : Boolean(capabilities?.glmConfigured);
  const model = isDeepSeek
    ? capabilities?.deepseekModel || "DeepSeek"
    : capabilities?.glmModel || "GLM";

  if (hasUserKey) {
    els.brainProviderState.textContent = `使用本次会话 ${model} Key`;
  } else if (serverConfigured) {
    els.brainProviderState.textContent = `服务端已配置 ${model}`;
  } else {
    els.brainProviderState.textContent = `需要 ${model} API Key`;
  }
  els.brainProviderHint.textContent = "仅优化修图指令，不直接生成或编辑图片";
}

function brainRequest() {
  return {
    name: state.brain.name,
    apiKey: state.brain.apiKey.trim(),
  };
}

function setMeta(values) {
  [...els.photoMeta.querySelectorAll("dd")].forEach((node, index) => {
    node.textContent = values[index] || "--";
  });
}

async function runTask(label, task) {
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
  els.uploadButton.disabled = isBusy;
  els.analyzeButton.disabled = isBusy || !state.photo;
  els.regenerateButton.disabled = isBusy || !state.currentJob;
  els.refineButton.disabled = isBusy || !state.currentJob;
  els.instructionInput.disabled = isBusy || !state.currentJob;
  els.providerSelect.disabled = isBusy;
  els.apiKeyInput.disabled = isBusy;
  els.workspaceInput.disabled = isBusy;
  els.brainProviderSelect.disabled = isBusy;
  els.brainApiKeyInput.disabled = isBusy;
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
