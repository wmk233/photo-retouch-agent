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
    (amount(values, "slimFace") +
      amount(values, "smallFace") * 0.65 +
      amount(values, "jawline") * 0.25) /
    1500;
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
    faceScaleY: clamp(
      1 + (amount(values, "chin") - amount(values, "forehead") * 0.4) / 1800,
      0.96,
      1.06,
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

function ellipsePath(context, region) {
  context.beginPath();
  context.ellipse(
    region.x + region.width / 2,
    region.y + region.height / 2,
    region.width / 2,
    region.height / 2,
    0,
    0,
    Math.PI * 2,
  );
}

function drawScaledRegion(context, source, region, scaleX, scaleY, opacity = 1) {
  if (Math.abs(scaleX - 1) < 0.001 && Math.abs(scaleY - 1) < 0.001) return;

  const destinationWidth = region.width * scaleX;
  const destinationHeight = region.height * scaleY;
  const destinationX = region.x + (region.width - destinationWidth) / 2;
  const destinationY = region.y + (region.height - destinationHeight) / 2;

  context.save();
  ellipsePath(context, region);
  context.clip();
  context.globalAlpha = opacity;
  context.drawImage(
    source,
    region.x,
    region.y,
    region.width,
    region.height,
    destinationX,
    destinationY,
    destinationWidth,
    destinationHeight,
  );
  context.restore();
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

    context.save();
    ellipsePath(context, {
      x: width * 0.245,
      y: height * 0.165,
      width: width * 0.51,
      height: height * 0.43,
    });
    context.clip();
    context.globalAlpha = recipe.smoothingOpacity;
    context.drawImage(softened, 0, 0);
    context.restore();
  }

  const snapshot = createCanvas(width, height);
  snapshot.getContext("2d").drawImage(canvas, 0, 0);
  drawScaledRegion(
    context,
    snapshot,
    {
      x: width * 0.245,
      y: height * 0.16,
      width: width * 0.51,
      height: height * 0.45,
    },
    recipe.faceScaleX,
    recipe.faceScaleY,
    0.92,
  );
  drawScaledRegion(
    context,
    snapshot,
    {
      x: width * 0.12,
      y: height * 0.49,
      width: width * 0.76,
      height: height * 0.62,
    },
    recipe.bodyScaleX,
    recipe.bodyScaleY,
    0.96,
  );

  drawEyeGlow(context, width, height, recipe);
  drawSculptLight(context, width, height, recipe);
  return recipe;
}
