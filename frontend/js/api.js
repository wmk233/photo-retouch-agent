const localApi = "http://127.0.0.1:8000/api";

export const API_BASE =
  window.PHOTO_AGENT_API_BASE || (window.location.origin.includes(":8000") ? "/api" : localApi);

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

export function createJob(sourceImageId, plan, userInstruction = "") {
  return request("/retouch/jobs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sourceImageId, plan, userInstruction }),
  });
}

export function refineJob(jobId, userInstruction) {
  return request(`/retouch/jobs/${jobId}/refine`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ userInstruction }),
  });
}
