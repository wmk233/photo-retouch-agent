const clamp = (value, min, max) => Math.min(max, Math.max(min, value));
const amount = (values, key) => clamp(Number(values[key]) || 0, 0, 100);

export function createRenderRecipe(values = {}) {
  const brightness =
    1 +
    (amount(values, "whiten") +
      amount(values, "skinTone") * 0.65 +
      amount(values, "brightness") +
      amount(values, "brightEyes") * 0.2) /
      720;
  const saturation =
    1 +
    (amount(values, "skinTone") +
      amount(values, "iris") +
      amount(values, "lipColor") +
      amount(values, "lipstick") +
      amount(values, "saturation")) /
      900;
  const contrast =
    1 +
    (amount(values, "sculpt") +
      amount(values, "contour") +
      amount(values, "contrast") +
      amount(values, "clarity")) /
      850;
  const warmth =
    (amount(values, "warmth") +
      amount(values, "blush") * 0.6 +
      amount(values, "foundation") * 0.35) /
    1300;
  const smoothing =
    (amount(values, "smooth") +
      amount(values, "wrinkle") * 0.8 +
      amount(values, "acne")) /
    260;
  const faceSlimming =
    (amount(values, "slimFace") + amount(values, "smallFace") * 0.75) /
    2100;
  const bodySlimming =
    (amount(values, "slimBelly") +
      amount(values, "slimWaist") +
      amount(values, "slimArms") * 0.35 +
      amount(values, "slimLegs") * 0.2) /
    1700;
  const verticalStretch =
    (amount(values, "longLegs") + amount(values, "headRatio") * 0.45) / 1800;

  return {
    brightness: clamp(brightness, 0.8, 1.35),
    saturation: clamp(saturation, 0.75, 1.45),
    contrast: clamp(contrast, 0.8, 1.4),
    warmth: clamp(warmth, 0, 0.18),
    smoothingOpacity: clamp(smoothing, 0, 0.58),
    smoothingRadius: clamp(
      1.2 +
        amount(values, "smooth") * 0.09 +
        amount(values, "acne") * 0.05,
      1.2,
      12,
    ),
    faceScaleX: clamp(1 - faceSlimming, 0.92, 1),
    faceScaleY: clamp(1 - amount(values, "smallFace") / 2500, 0.96, 1),
    jawScaleX: clamp(1 - amount(values, "jawline") / 1600, 0.935, 1),
    cheekScaleX: clamp(1 - amount(values, "cheekbone") / 2000, 0.95, 1),
    chinScaleY: clamp(1 + amount(values, "chin") / 1800, 1, 1.06),
    foreheadScaleY: clamp(
      1 - amount(values, "forehead") / 2000,
      0.95,
      1,
    ),
    doubleChinScaleY: clamp(
      1 - amount(values, "doubleChin") / 1800,
      0.94,
      1,
    ),
    bodyScaleX: clamp(1 - bodySlimming, 0.9, 1),
    bodyScaleY: clamp(1 + verticalStretch, 1, 1.08),
    eyeGlow: clamp(
      (amount(values, "iris") +
        amount(values, "brightEyes") +
        amount(values, "eyeBags") * 0.25) /
        220,
      0,
      0.65,
    ),
    eyeScale: clamp(1 + amount(values, "enlargeEyes") / 1250, 1, 1.08),
    sculpt: clamp(
      (amount(values, "sculpt") +
        amount(values, "noseBridge") * 0.45 +
        amount(values, "contour") * 0.55) /
        180,
      0,
      0.75,
    ),
  };
}

export function buildCanvasFilter(recipe) {
  return [
    `brightness(${recipe.brightness.toFixed(3)})`,
    `saturate(${recipe.saturation.toFixed(3)})`,
    `contrast(${recipe.contrast.toFixed(3)})`,
    `sepia(${recipe.warmth.toFixed(3)})`,
  ].join(" ");
}

export function activeAdjustmentCount(values = {}) {
  return Object.values(values).filter((value) => Number(value) > 0).length;
}

function createCanvas(width, height) {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  return canvas;
}

export function mapWarpPosition(position, scale) {
  const normalized = clamp(position, 0, 1) * 2 - 1;
  const warped =
    normalized *
    (scale + (1 - scale) * normalized * normalized);
  return (warped + 1) / 2;
}

function drawAxisWarp(context, source, region, scale, axis) {
  const horizontal = axis === "x";
  const dimension = horizontal ? region.width : region.height;
  const sliceCount = clamp(Math.ceil(dimension / 6), 48, 96);

  for (let index = 0; index < sliceCount; index += 1) {
    const start = index / sliceCount;
    const end = (index + 1) / sliceCount;
    const sourceStart =
      (horizontal ? region.x : region.y) + dimension * start;
    const sourceEnd =
      (horizontal ? region.x : region.y) + dimension * end;
    const destinationStart =
      (horizontal ? region.x : region.y) +
      dimension * mapWarpPosition(start, scale);
    const destinationEnd =
      (horizontal ? region.x : region.y) +
      dimension * mapWarpPosition(end, scale);

    if (horizontal) {
      context.drawImage(
        source,
        sourceStart,
        region.y,
        sourceEnd - sourceStart,
        region.height,
        destinationStart,
        region.y,
        destinationEnd - destinationStart + 0.6,
        region.height,
      );
    } else {
      context.drawImage(
        source,
        region.x,
        sourceStart,
        region.width,
        sourceEnd - sourceStart,
        region.x,
        destinationStart,
        region.width,
        destinationEnd - destinationStart + 0.6,
      );
    }
  }
}

function applyFeatherMask(
  canvas,
  region,
  featherStart = 0.82,
  opacity = 1,
) {
  const context = canvas.getContext("2d");
  const radiusX = region.width / 2;
  const radiusY = region.height / 2;

  context.save();
  context.globalCompositeOperation = "destination-in";
  context.translate(region.x + radiusX, region.y + radiusY);
  context.scale(radiusX, radiusY);
  const gradient = context.createRadialGradient(0, 0, 0, 0, 0, 1);
  gradient.addColorStop(0, `rgba(0, 0, 0, ${opacity})`);
  gradient.addColorStop(
    featherStart,
    `rgba(0, 0, 0, ${opacity})`,
  );
  gradient.addColorStop(1, "rgba(0, 0, 0, 0)");
  context.fillStyle = gradient;
  context.fillRect(-1, -1, 2, 2);
  context.restore();
}

function drawScaledRegion(
  context,
  source,
  region,
  scaleX,
  scaleY,
) {
  if (Math.abs(scaleX - 1) < 0.001 && Math.abs(scaleY - 1) < 0.001) return;

  let warped = source;

  if (Math.abs(scaleX - 1) >= 0.001) {
    const horizontalWarp = createCanvas(source.width, source.height);
    drawAxisWarp(
      horizontalWarp.getContext("2d"),
      source,
      region,
      scaleX,
      "x",
    );
    warped = horizontalWarp;
  }

  if (Math.abs(scaleY - 1) >= 0.001) {
    const verticalWarp = createCanvas(source.width, source.height);
    drawAxisWarp(
      verticalWarp.getContext("2d"),
      warped,
      region,
      scaleY,
      "y",
    );
    warped = verticalWarp;
  }

  applyFeatherMask(warped, region);
  context.drawImage(warped, 0, 0);
}

function drawCurrentWarp(context, region, scaleX, scaleY) {
  if (Math.abs(scaleX - 1) < 0.001 && Math.abs(scaleY - 1) < 0.001) return;

  const snapshot = createCanvas(context.canvas.width, context.canvas.height);
  snapshot.getContext("2d").drawImage(context.canvas, 0, 0);
  drawScaledRegion(context, snapshot, region, scaleX, scaleY);
}

function drawEyeGlow(context, width, height, recipe) {
  if (recipe.eyeGlow <= 0) return;

  const eyes = [
    { x: width * 0.43, y: height * 0.365 },
    { x: width * 0.57, y: height * 0.365 },
  ];
  context.save();
  context.globalCompositeOperation = "screen";
  eyes.forEach((eye) => {
    const radius = width * 0.055 * recipe.eyeScale;
    const gradient = context.createRadialGradient(
      eye.x,
      eye.y,
      0,
      eye.x,
      eye.y,
      radius,
    );
    gradient.addColorStop(0, `rgba(205, 236, 230, ${recipe.eyeGlow * 0.22})`);
    gradient.addColorStop(0.35, `rgba(178, 216, 209, ${recipe.eyeGlow * 0.12})`);
    gradient.addColorStop(1, "rgba(255, 255, 255, 0)");
    context.fillStyle = gradient;
    context.beginPath();
    context.ellipse(
      eye.x,
      eye.y,
      radius,
      radius * 0.52,
      0,
      0,
      Math.PI * 2,
    );
    context.fill();
  });
  context.restore();
}

function drawSculptLight(context, width, height, recipe) {
  if (recipe.sculpt <= 0) return;

  context.save();
  context.globalCompositeOperation = "soft-light";
  const highlight = context.createRadialGradient(
    width * 0.48,
    height * 0.38,
    0,
    width * 0.48,
    height * 0.38,
    width * 0.2,
  );
  highlight.addColorStop(0, `rgba(255, 244, 229, ${recipe.sculpt * 0.2})`);
  highlight.addColorStop(1, "rgba(255, 244, 229, 0)");
  context.fillStyle = highlight;
  context.fillRect(width * 0.25, height * 0.18, width * 0.5, height * 0.48);
  context.restore();
}

export function renderLocalRetouch(canvas, image, values = {}) {
  const width = image.naturalWidth || image.width;
  const height = image.naturalHeight || image.height;
  if (!width || !height) return null;

  if (canvas.width !== width) canvas.width = width;
  if (canvas.height !== height) canvas.height = height;

  const context = canvas.getContext("2d", { alpha: false });
  context.imageSmoothingEnabled = true;
  context.imageSmoothingQuality = "high";

  const source = createCanvas(width, height);
  source.getContext("2d").drawImage(image, 0, 0, width, height);
  const recipe = createRenderRecipe(values);

  context.save();
  context.filter = buildCanvasFilter(recipe);
  context.drawImage(source, 0, 0, width, height);
  context.restore();

  if (recipe.smoothingOpacity > 0) {
    const softened = createCanvas(width, height);
    const softenedContext = softened.getContext("2d");
    softenedContext.filter = `blur(${recipe.smoothingRadius.toFixed(1)}px)`;
    softenedContext.drawImage(source, 0, 0, width, height);

    applyFeatherMask(
      softened,
      {
        x: width * 0.29,
        y: height * 0.22,
        width: width * 0.42,
        height: height * 0.34,
      },
      0.68,
      recipe.smoothingOpacity,
    );
    context.drawImage(softened, 0, 0);
  }

  drawCurrentWarp(
    context,
    {
      x: width * 0.16,
      y: height * 0.09,
      width: width * 0.68,
      height: height * 0.59,
    },
    recipe.faceScaleX,
    recipe.faceScaleY,
  );
  drawCurrentWarp(
    context,
    {
      x: width * 0.22,
      y: height * 0.29,
      width: width * 0.56,
      height: height * 0.26,
    },
    recipe.cheekScaleX,
    1,
  );
  drawCurrentWarp(
    context,
    {
      x: width * 0.24,
      y: height * 0.37,
      width: width * 0.52,
      height: height * 0.25,
    },
    recipe.jawScaleX,
    1,
  );
  drawCurrentWarp(
    context,
    {
      x: width * 0.27,
      y: height * 0.14,
      width: width * 0.46,
      height: height * 0.22,
    },
    1,
    recipe.foreheadScaleY,
  );
  drawCurrentWarp(
    context,
    {
      x: width * 0.34,
      y: height * 0.45,
      width: width * 0.32,
      height: height * 0.18,
    },
    1,
    recipe.chinScaleY,
  );
  drawCurrentWarp(
    context,
    {
      x: width * 0.29,
      y: height * 0.5,
      width: width * 0.42,
      height: height * 0.18,
    },
    1,
    recipe.doubleChinScaleY,
  );
  drawCurrentWarp(
    context,
    {
      x: width * 0.12,
      y: height * 0.49,
      width: width * 0.76,
      height: height * 0.62,
    },
    recipe.bodyScaleX,
    recipe.bodyScaleY,
  );

  drawEyeGlow(context, width, height, recipe);
  drawSculptLight(context, width, height, recipe);
  return recipe;
}
