const localApi = "http://127.0.0.1:8000/api";
const staticPreviewPorts = new Set(["4173", "5500"]);

export const API_BASE =
  window.PHOTO_AGENT_API_BASE ||
  (staticPreviewPorts.has(window.location.port) ? localApi : "/api");

export function assetUrl(path) {
  if (!path) return "";
  if (/^https?:\/\//.test(path)) return path;
  const apiOrigin = API_BASE.startsWith("http") ? new URL(API_BASE).origin : window.location.origin;
  return `${apiOrigin}${path}`;
}

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, options);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.detail || `Request failed: ${response.status}`);
  }
  return payload;
}

export function uploadPhoto(file) {
  const formData = new FormData();
  formData.append("file", file);
  return request("/photos/upload", {
    method: "POST",
    body: formData,
  });
}

export function analyzePhoto(imageId) {
  return request("/photos/analyze", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ imageId, domainType: "portrait" }),
  });
}

export function createPlans(analysis) {
  return request("/retouch/plans", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ analysis }),
  });
}

export function getProviderCapabilities() {
  return request("/retouch/providers");
}

function providerHeaders(provider = {}, brain = {}) {
  const headers = { "Content-Type": "application/json" };
  if (provider.name) headers["X-AI-Provider"] = provider.name;
  if (provider.apiKey) headers["X-AI-API-Key"] = provider.apiKey;
  if (provider.workspaceId) headers["X-AI-Workspace-Id"] = provider.workspaceId;
  if (brain.name) headers["X-Agent-Provider"] = brain.name;
  if (brain.apiKey) headers["X-Agent-API-Key"] = brain.apiKey;
  return headers;
}

export function createJob(
  sourceImageId,
  plan,
  userInstruction = "",
  provider = {},
  brain = {},
) {
  return request("/retouch/jobs", {
    method: "POST",
    headers: providerHeaders(provider, brain),
    body: JSON.stringify({ sourceImageId, plan, userInstruction }),
  });
}

export function refineJob(jobId, userInstruction, provider = {}, brain = {}) {
  return request(`/retouch/jobs/${jobId}/refine`, {
    method: "POST",
    headers: providerHeaders(provider, brain),
    body: JSON.stringify({ userInstruction }),
  });
}
