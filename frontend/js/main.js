import { analyzePhoto, assetUrl, createJob, createPlans, refineJob, uploadPhoto } from "./api.js";

const state = {
  photo: null,
  analysis: null,
  plans: [],
  selectedPlan: null,
  currentJob: null,
  history: [],
};

const $ = (selector) => document.querySelector(selector);

const els = {
  status: $("#status"),
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
    const job = await createJob(state.photo.imageId, plan, instruction);
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
    const job = await refineJob(state.currentJob.jobId, instruction);
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
    : `方案：${plan?.title || job.planId}。模型：${job.modelProvider}/${job.modelName}。`;
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
