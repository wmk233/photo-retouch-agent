function safeBaseName(fileName) {
  const baseName = String(fileName || "portrait")
    .replace(/\.[^.]+$/, "")
    .trim()
    .replace(/[\\/:*?"<>|]+/g, "-")
    .replace(/\s+/g, "-");
  return baseName || "portrait";
}

export function buildExportFilename(
  sourceName,
  mode = "local",
  mimeType = "image/png",
) {
  const extension = mimeType === "image/jpeg" ? "jpg" : "png";
  const suffix = mode === "ai" ? "ai-retouched" : "retouched";
  return `${safeBaseName(sourceName)}-${suffix}.${extension}`;
}

export function canvasToBlob(canvas, mimeType = "image/png", quality = 0.94) {
  if (!canvas || typeof canvas.toBlob !== "function") {
    return Promise.reject(new Error("当前浏览器无法导出画布。"));
  }

  return new Promise((resolve, reject) => {
    canvas.toBlob(
      (blob) => {
        if (blob) resolve(blob);
        else reject(new Error("图片导出失败。"));
      },
      mimeType,
      quality,
    );
  });
}

export function triggerBlobDownload(
  blob,
  fileName,
  {
    documentRef = globalThis.document,
    urlApi = globalThis.URL,
    schedule = globalThis.setTimeout,
  } = {},
) {
  if (!blob || !documentRef || !urlApi) {
    throw new Error("当前环境无法下载图片。");
  }

  const objectUrl = urlApi.createObjectURL(blob);
  const anchor = documentRef.createElement("a");
  anchor.href = objectUrl;
  anchor.download = fileName;
  anchor.hidden = true;
  documentRef.body.append(anchor);
  anchor.click();
  anchor.remove();
  schedule(() => urlApi.revokeObjectURL(objectUrl), 1000);
  return objectUrl;
}
