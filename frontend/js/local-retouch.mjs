const clamp = (value, min, max) => Math.min(max, Math.max(min, value));
const amount = (values, key) => clamp(Number(values[key]) || 0, 0, 100);

export function createRenderRecipe(values = {}) {
  const brightness =
    1 +
    (amount(values, "whiten") +
      amount(values, "skinTone") * 0.18 +
      amount(values, "brightness")) /
      720;
  const saturation =
    1 + amount(values, "saturation") / 900;
  const contrast =
    1 +
    (amount(values, "contrast") +
      amount(values, "clarity")) /
      850;
  const warmth =
    amount(values, "warmth") / 1300;
  const faceSlimming =
    (amount(values, "slimFace") + amount(values, "smallFace") * 0.75) /
    2100;
  return {
    brightness: clamp(brightness, 0.8, 1.35),
    saturation: clamp(saturation, 0.75, 1.45),
    contrast: clamp(contrast, 0.8, 1.4),
    warmth: clamp(warmth, 0, 0.18),
    skinToneStrength: amount(values, "skinTone") / 100,
    whiteningStrength: amount(values, "whiten") / 100,
    smoothingOpacity: clamp(amount(values, "smooth") / 240, 0, 0.42),
    smoothingRadius: clamp(
      1.2 + amount(values, "smooth") * 0.075,
      1.2,
      9,
    ),
    acneOpacity: clamp(amount(values, "acne") / 260, 0, 0.38),
    acneRadius: clamp(1 + amount(values, "acne") * 0.065, 1, 7.5),
    wrinkleOpacity: clamp(amount(values, "wrinkle") / 250, 0, 0.4),
    wrinkleRadius: clamp(
      0.8 + amount(values, "wrinkle") * 0.045,
      0.8,
      5.3,
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
    bellyScaleX: clamp(1 - amount(values, "slimBelly") / 1000, 0.9, 1),
    waistScaleX: clamp(1 - amount(values, "slimWaist") / 820, 0.875, 1),
    armScaleX: clamp(1 - amount(values, "slimArms") / 1200, 0.915, 1),
    legScaleX: clamp(1 - amount(values, "slimLegs") / 1100, 0.905, 1),
    legScaleY: clamp(1 + amount(values, "longLegs") / 1000, 1, 1.1),
    shoulderScaleX: clamp(
      1 + amount(values, "shoulders") / 1250,
      1,
      1.08,
    ),
    hipScaleX: clamp(1 + amount(values, "hips") / 1400, 1, 1.075),
    hipScaleY: clamp(1 - amount(values, "hips") / 2500, 0.96, 1),
    headRatioScale: clamp(
      1 - amount(values, "headRatio") / 1600,
      0.935,
      1,
    ),
    eyeScale: clamp(1 + amount(values, "enlargeEyes") / 550, 1, 1.18),
    eyeBagStrength: amount(values, "eyeBags") / 100,
    darkCircleStrength: amount(values, "darkCircles") / 100,
    irisStrength: amount(values, "iris") / 100,
    brightEyeStrength: amount(values, "brightEyes") / 100,
    eyeGlow: clamp(
      amount(values, "brightEyes") / 185 +
        amount(values, "iris") / 1000,
      0,
      0.65,
    ),
    eyeDistanceScale: clamp(
      1 + amount(values, "eyeDistance") / 1800,
      1,
      1.06,
    ),
    eyeAngle: clamp(
      (amount(values, "eyeAngle") / 100) * (Math.PI / 45),
      0,
      Math.PI / 45,
    ),
    noseBridgeStrength: amount(values, "noseBridge") / 100,
    noseScaleX: clamp(1 - amount(values, "slimNose") / 1300, 0.92, 1),
    teethStrength: amount(values, "teeth") / 100,
    smileAngle: clamp(
      (amount(values, "smile") / 100) * (Math.PI / 45),
      0,
      Math.PI / 45,
    ),
    lipScaleY: clamp(1 + amount(values, "lipVolume") / 600, 1, 1.17),
    lipColorStrength: amount(values, "lipColor") / 100,
    browStrength: amount(values, "brows") / 100,
    foundationStrength: amount(values, "foundation") / 100,
    blushStrength: amount(values, "blush") / 100,
    contourStrength: amount(values, "contour") / 100,
    eyebrowMakeupStrength: amount(values, "eyebrowMakeup") / 100,
    eyelinerStrength: amount(values, "eyeliner") / 100,
    lipstickStrength: amount(values, "lipstick") / 100,
    sculpt: clamp(amount(values, "sculpt") / 140, 0, 0.72),
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

function drawFilteredRegion(
  context,
  source,
  region,
  filter,
  opacity,
  featherStart = 0.68,
) {
  if (opacity <= 0) return;

  const filtered = createCanvas(source.width, source.height);
  const filteredContext = filtered.getContext("2d");
  filteredContext.filter = filter;
  filteredContext.drawImage(source, 0, 0);
  applyFeatherMask(filtered, region, featherStart, opacity);
  context.drawImage(filtered, 0, 0);
}

function drawTransformedRegion(
  context,
  source,
  region,
  {
    scaleX = 1,
    scaleY = 1,
    rotation = 0,
    opacity = 1,
  } = {},
) {
  if (
    Math.abs(scaleX - 1) < 0.001 &&
    Math.abs(scaleY - 1) < 0.001 &&
    Math.abs(rotation) < 0.001
  ) {
    return;
  }

  const layer = createCanvas(source.width, source.height);
  const layerContext = layer.getContext("2d");
  const centerX = region.x + region.width / 2;
  const centerY = region.y + region.height / 2;
  layerContext.save();
  layerContext.translate(centerX, centerY);
  layerContext.rotate(rotation);
  layerContext.scale(scaleX, scaleY);
  layerContext.drawImage(
    source,
    region.x,
    region.y,
    region.width,
    region.height,
    -region.width / 2,
    -region.height / 2,
    region.width,
    region.height,
  );
  layerContext.restore();
  applyFeatherMask(layer, region, 0.62, opacity);
  context.drawImage(layer, 0, 0);
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
    gradient.addColorStop(0, `rgba(240, 247, 248, ${recipe.eyeGlow * 0.2})`);
    gradient.addColorStop(0.35, `rgba(220, 234, 236, ${recipe.eyeGlow * 0.1})`);
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

function drawEyeRetouch(context, width, height, recipe) {
  const eyeRegions = [
    {
      x: width * 0.345,
      y: height * 0.315,
      width: width * 0.17,
      height: height * 0.105,
      angle: recipe.eyeAngle,
    },
    {
      x: width * 0.485,
      y: height * 0.315,
      width: width * 0.17,
      height: height * 0.105,
      angle: -recipe.eyeAngle,
    },
  ];

  if (recipe.eyeScale > 1 || recipe.eyeAngle > 0) {
    const eyeSource = createCanvas(width, height);
    eyeSource.getContext("2d").drawImage(context.canvas, 0, 0);
    eyeRegions.forEach((region) => {
      drawTransformedRegion(context, eyeSource, region, {
        scaleX: recipe.eyeScale,
        scaleY: 1 + (recipe.eyeScale - 1) * 0.7,
        rotation: region.angle,
      });
    });
  }

  drawCurrentWarp(
    context,
    {
      x: width * 0.27,
      y: height * 0.29,
      width: width * 0.46,
      height: height * 0.17,
    },
    recipe.eyeDistanceScale,
    1,
  );

  const detailSource = createCanvas(width, height);
  detailSource.getContext("2d").drawImage(context.canvas, 0, 0);
  const underEyeRegions = [
    {
      x: width * 0.33,
      y: height * 0.355,
      width: width * 0.19,
      height: height * 0.105,
    },
    {
      x: width * 0.48,
      y: height * 0.355,
      width: width * 0.19,
      height: height * 0.105,
    },
  ];
  underEyeRegions.forEach((region) => {
    drawFilteredRegion(
      context,
      detailSource,
      region,
      `blur(${(1 + recipe.eyeBagStrength * 4).toFixed(
        1,
      )}px) brightness(${(1 + recipe.eyeBagStrength * 0.09).toFixed(3)})`,
      recipe.eyeBagStrength * 0.36,
      0.55,
    );
    drawFilteredRegion(
      context,
      detailSource,
      region,
      `brightness(${(1 + recipe.darkCircleStrength * 0.16).toFixed(
        3,
      )}) saturate(${(1 - recipe.darkCircleStrength * 0.06).toFixed(3)})`,
      recipe.darkCircleStrength * 0.38,
      0.52,
    );
  });

  const irisRegions = [
    {
      x: width * 0.375,
      y: height * 0.338,
      width: width * 0.11,
      height: height * 0.055,
    },
    {
      x: width * 0.515,
      y: height * 0.338,
      width: width * 0.11,
      height: height * 0.055,
    },
  ];
  irisRegions.forEach((region) => {
    drawFilteredRegion(
      context,
      detailSource,
      region,
      `contrast(${(1 + recipe.irisStrength * 0.28).toFixed(
        3,
      )}) saturate(${(1 + recipe.irisStrength * 0.2).toFixed(
        3,
      )}) brightness(${(1 + recipe.irisStrength * 0.035).toFixed(3)})`,
      recipe.irisStrength * 0.55,
      0.45,
    );
  });

  drawEyeGlow(context, width, height, recipe);
}

function drawNoseBridge(context, width, height, strength) {
  if (strength <= 0) return;

  context.save();
  context.globalCompositeOperation = "soft-light";
  const highlight = context.createLinearGradient(
    width * 0.5,
    height * 0.31,
    width * 0.5,
    height * 0.5,
  );
  highlight.addColorStop(0, "rgba(255, 244, 226, 0)");
  highlight.addColorStop(
    0.45,
    `rgba(255, 244, 226, ${strength * 0.32})`,
  );
  highlight.addColorStop(1, "rgba(255, 244, 226, 0)");
  context.strokeStyle = highlight;
  context.lineWidth = width * 0.018;
  context.lineCap = "round";
  context.beginPath();
  context.moveTo(width * 0.5, height * 0.32);
  context.lineTo(width * 0.5, height * 0.49);
  context.stroke();

  context.strokeStyle = `rgba(76, 48, 39, ${strength * 0.09})`;
  context.lineWidth = width * 0.012;
  [
    [width * 0.475, width * 0.485],
    [width * 0.525, width * 0.515],
  ].forEach(([startX, endX]) => {
    context.beginPath();
    context.moveTo(startX, height * 0.34);
    context.lineTo(endX, height * 0.49);
    context.stroke();
  });
  context.restore();
}

function drawLipColor(context, width, height, strength) {
  if (strength <= 0) return;

  context.save();
  context.globalCompositeOperation = "soft-light";
  const lipGradient = context.createRadialGradient(
    width * 0.5,
    height * 0.525,
    0,
    width * 0.5,
    height * 0.525,
    width * 0.11,
  );
  lipGradient.addColorStop(
    0,
    `rgba(188, 48, 72, ${strength * 0.52})`,
  );
  lipGradient.addColorStop(
    0.58,
    `rgba(164, 42, 63, ${strength * 0.3})`,
  );
  lipGradient.addColorStop(1, "rgba(164, 42, 63, 0)");
  context.fillStyle = lipGradient;
  context.beginPath();
  context.ellipse(
    width * 0.5,
    height * 0.525,
    width * 0.105,
    height * 0.032,
    0,
    0,
    Math.PI * 2,
  );
  context.fill();
  context.restore();
}

function drawBrows(context, source, width, height, strength) {
  if (strength <= 0) return;

  [
    {
      x: width * 0.34,
      y: height * 0.285,
      width: width * 0.15,
      height: height * 0.065,
    },
    {
      x: width * 0.51,
      y: height * 0.285,
      width: width * 0.15,
      height: height * 0.065,
    },
  ].forEach((region) => {
    drawFilteredRegion(
      context,
      source,
      region,
      `brightness(${(1 - strength * 0.18).toFixed(3)}) contrast(${(
        1 +
        strength * 0.32
      ).toFixed(3)})`,
      strength * 0.58,
      0.5,
    );
  });
}

function drawFeatureRetouch(context, width, height, recipe) {
  drawCurrentWarp(
    context,
    {
      x: width * 0.41,
      y: height * 0.31,
      width: width * 0.18,
      height: height * 0.23,
    },
    recipe.noseScaleX,
    1,
  );

  if (recipe.lipScaleY > 1) {
    const lipSource = createCanvas(width, height);
    lipSource.getContext("2d").drawImage(context.canvas, 0, 0);
    drawTransformedRegion(
      context,
      lipSource,
      {
        x: width * 0.39,
        y: height * 0.475,
        width: width * 0.22,
        height: height * 0.105,
      },
      {
        scaleX: 1 + (recipe.lipScaleY - 1) * 0.2,
        scaleY: recipe.lipScaleY,
      },
    );
  }

  if (recipe.smileAngle > 0) {
    const smileSource = createCanvas(width, height);
    smileSource.getContext("2d").drawImage(context.canvas, 0, 0);
    [
      {
        x: width * 0.385,
        y: height * 0.48,
        width: width * 0.125,
        height: height * 0.1,
        rotation: recipe.smileAngle,
      },
      {
        x: width * 0.49,
        y: height * 0.48,
        width: width * 0.125,
        height: height * 0.1,
        rotation: -recipe.smileAngle,
      },
    ].forEach((region) => {
      drawTransformedRegion(context, smileSource, region, {
        rotation: region.rotation,
      });
    });
  }

  const featureSource = createCanvas(width, height);
  featureSource.getContext("2d").drawImage(context.canvas, 0, 0);
  drawFilteredRegion(
    context,
    featureSource,
    {
      x: width * 0.42,
      y: height * 0.505,
      width: width * 0.16,
      height: height * 0.055,
    },
    `brightness(${(1 + recipe.teethStrength * 0.24).toFixed(
      3,
    )}) saturate(${(1 - recipe.teethStrength * 0.55).toFixed(3)})`,
    recipe.teethStrength * 0.52,
    0.48,
  );

  drawNoseBridge(context, width, height, recipe.noseBridgeStrength);
  drawLipColor(context, width, height, recipe.lipColorStrength);
  drawBrows(context, featureSource, width, height, recipe.browStrength);
}

function drawBodyRetouch(context, width, height, recipe) {
  drawCurrentWarp(
    context,
    {
      x: width * 0.14,
      y: height * 0.06,
      width: width * 0.72,
      height: height * 0.58,
    },
    recipe.headRatioScale,
    recipe.headRatioScale,
  );
  drawCurrentWarp(
    context,
    {
      x: width * 0.08,
      y: height * 0.46,
      width: width * 0.84,
      height: height * 0.23,
    },
    recipe.shoulderScaleX,
    1,
  );
  drawCurrentWarp(
    context,
    {
      x: width * 0.17,
      y: height * 0.54,
      width: width * 0.66,
      height: height * 0.35,
    },
    recipe.bellyScaleX,
    1,
  );
  drawCurrentWarp(
    context,
    {
      x: width * 0.2,
      y: height * 0.62,
      width: width * 0.6,
      height: height * 0.27,
    },
    recipe.waistScaleX,
    1,
  );
  [
    {
      x: width * 0.035,
      y: height * 0.52,
      width: width * 0.28,
      height: height * 0.46,
    },
    {
      x: width * 0.685,
      y: height * 0.52,
      width: width * 0.28,
      height: height * 0.46,
    },
  ].forEach((region) => {
    drawCurrentWarp(context, region, recipe.armScaleX, 1);
  });
  drawCurrentWarp(
    context,
    {
      x: width * 0.16,
      y: height * 0.72,
      width: width * 0.68,
      height: height * 0.25,
    },
    recipe.hipScaleX,
    recipe.hipScaleY,
  );
  drawCurrentWarp(
    context,
    {
      x: width * 0.15,
      y: height * 0.73,
      width: width * 0.7,
      height: height * 0.36,
    },
    recipe.legScaleX,
    recipe.legScaleY,
  );
}

function drawBlush(context, width, height, strength) {
  if (strength <= 0) return;

  context.save();
  context.globalCompositeOperation = "soft-light";
  [
    { x: width * 0.37, y: height * 0.43 },
    { x: width * 0.63, y: height * 0.43 },
  ].forEach((point) => {
    const blush = context.createRadialGradient(
      point.x,
      point.y,
      0,
      point.x,
      point.y,
      width * 0.095,
    );
    blush.addColorStop(0, `rgba(231, 104, 116, ${strength * 0.4})`);
    blush.addColorStop(0.58, `rgba(222, 118, 126, ${strength * 0.2})`);
    blush.addColorStop(1, "rgba(222, 118, 126, 0)");
    context.fillStyle = blush;
    context.fillRect(
      point.x - width * 0.11,
      point.y - height * 0.075,
      width * 0.22,
      height * 0.15,
    );
  });
  context.restore();
}

function drawMakeupContour(context, width, height, strength) {
  if (strength <= 0) return;

  context.save();
  context.globalCompositeOperation = "soft-light";
  [
    { x: width * 0.31, y: height * 0.43 },
    { x: width * 0.69, y: height * 0.43 },
  ].forEach((point) => {
    const shadow = context.createRadialGradient(
      point.x,
      point.y,
      0,
      point.x,
      point.y,
      width * 0.14,
    );
    shadow.addColorStop(0, `rgba(78, 48, 43, ${strength * 0.22})`);
    shadow.addColorStop(1, "rgba(78, 48, 43, 0)");
    context.fillStyle = shadow;
    context.fillRect(
      point.x - width * 0.15,
      point.y - height * 0.13,
      width * 0.3,
      height * 0.26,
    );
  });

  const centerLight = context.createRadialGradient(
    width * 0.5,
    height * 0.39,
    0,
    width * 0.5,
    height * 0.39,
    width * 0.16,
  );
  centerLight.addColorStop(
    0,
    `rgba(255, 236, 218, ${strength * 0.16})`,
  );
  centerLight.addColorStop(1, "rgba(255, 236, 218, 0)");
  context.fillStyle = centerLight;
  context.fillRect(
    width * 0.34,
    height * 0.24,
    width * 0.32,
    height * 0.32,
  );
  context.restore();
}

function drawEyeliner(context, source, width, height, strength) {
  if (strength <= 0) return;

  [
    {
      x: width * 0.36,
      y: height * 0.335,
      width: width * 0.14,
      height: height * 0.06,
    },
    {
      x: width * 0.5,
      y: height * 0.335,
      width: width * 0.14,
      height: height * 0.06,
    },
  ].forEach((region) => {
    drawFilteredRegion(
      context,
      source,
      region,
      `brightness(${(1 - strength * 0.24).toFixed(3)}) contrast(${(
        1 +
        strength * 0.42
      ).toFixed(3)})`,
      strength * 0.48,
      0.46,
    );
  });
}

function drawLipstick(context, width, height, strength) {
  if (strength <= 0) return;

  context.save();
  context.globalCompositeOperation = "soft-light";
  const lipstick = context.createRadialGradient(
    width * 0.5,
    height * 0.525,
    0,
    width * 0.5,
    height * 0.525,
    width * 0.105,
  );
  lipstick.addColorStop(0, `rgba(184, 42, 68, ${strength * 0.72})`);
  lipstick.addColorStop(0.62, `rgba(143, 31, 55, ${strength * 0.42})`);
  lipstick.addColorStop(1, "rgba(143, 31, 55, 0)");
  context.fillStyle = lipstick;
  context.beginPath();
  context.ellipse(
    width * 0.5,
    height * 0.525,
    width * 0.105,
    height * 0.031,
    0,
    0,
    Math.PI * 2,
  );
  context.fill();
  context.restore();
}

function drawMakeupRetouch(context, width, height, recipe) {
  const makeupSource = createCanvas(width, height);
  makeupSource.getContext("2d").drawImage(context.canvas, 0, 0);
  drawFilteredRegion(
    context,
    makeupSource,
    {
      x: width * 0.27,
      y: height * 0.2,
      width: width * 0.46,
      height: height * 0.38,
    },
    `blur(${(recipe.foundationStrength * 2.2).toFixed(
      1,
    )}px) brightness(${(1 + recipe.foundationStrength * 0.055).toFixed(
      3,
    )}) saturate(${(1 + recipe.foundationStrength * 0.045).toFixed(
      3,
    )}) sepia(${(recipe.foundationStrength * 0.025).toFixed(3)})`,
    recipe.foundationStrength * 0.34,
    0.7,
  );
  drawBlush(context, width, height, recipe.blushStrength);
  drawMakeupContour(context, width, height, recipe.contourStrength);

  const eyebrowSource = createCanvas(width, height);
  eyebrowSource.getContext("2d").drawImage(context.canvas, 0, 0);
  drawBrows(
    context,
    eyebrowSource,
    width,
    height,
    recipe.eyebrowMakeupStrength * 0.9,
  );
  drawEyeliner(
    context,
    eyebrowSource,
    width,
    height,
    recipe.eyelinerStrength,
  );
  drawLipstick(context, width, height, recipe.lipstickStrength);
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

  const shadowOpacity = recipe.sculpt * 0.16;
  [
    { x: width * 0.34, y: height * 0.41 },
    { x: width * 0.66, y: height * 0.41 },
  ].forEach((point) => {
    const shadow = context.createRadialGradient(
      point.x,
      point.y,
      0,
      point.x,
      point.y,
      width * 0.13,
    );
    shadow.addColorStop(0, `rgba(78, 48, 38, ${shadowOpacity})`);
    shadow.addColorStop(1, "rgba(78, 48, 38, 0)");
    context.fillStyle = shadow;
    context.fillRect(
      point.x - width * 0.15,
      point.y - height * 0.11,
      width * 0.3,
      height * 0.22,
    );
  });
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

  const skinSource = createCanvas(width, height);
  skinSource.getContext("2d").drawImage(canvas, 0, 0);
  drawFilteredRegion(
    context,
    skinSource,
    {
      x: width * 0.27,
      y: height * 0.2,
      width: width * 0.46,
      height: height * 0.38,
    },
    `brightness(${(1 + recipe.skinToneStrength * 0.06).toFixed(3)}) saturate(${(
      1 +
      recipe.skinToneStrength * 0.1
    ).toFixed(3)}) sepia(${(recipe.skinToneStrength * 0.035).toFixed(3)})`,
    recipe.skinToneStrength * 0.52,
    0.72,
  );
  drawFilteredRegion(
    context,
    skinSource,
    {
      x: width * 0.29,
      y: height * 0.22,
      width: width * 0.42,
      height: height * 0.34,
    },
    `blur(${recipe.smoothingRadius.toFixed(1)}px)`,
    recipe.smoothingOpacity,
  );
  drawFilteredRegion(
    context,
    skinSource,
    {
      x: width * 0.3,
      y: height * 0.25,
      width: width * 0.4,
      height: height * 0.31,
    },
    `blur(${recipe.acneRadius.toFixed(1)}px) contrast(0.985)`,
    recipe.acneOpacity,
    0.62,
  );
  const wrinkleRegions = [
    {
      x: width * 0.32,
      y: height * 0.21,
      width: width * 0.36,
      height: height * 0.15,
    },
    {
      x: width * 0.31,
      y: height * 0.33,
      width: width * 0.18,
      height: height * 0.1,
    },
    {
      x: width * 0.51,
      y: height * 0.33,
      width: width * 0.18,
      height: height * 0.1,
    },
  ];
  wrinkleRegions.forEach((region) => {
    drawFilteredRegion(
      context,
      skinSource,
      region,
      `blur(${recipe.wrinkleRadius.toFixed(1)}px)`,
      recipe.wrinkleOpacity,
      0.58,
    );
  });

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
  drawBodyRetouch(context, width, height, recipe);
  drawFeatureRetouch(context, width, height, recipe);
  drawEyeRetouch(context, width, height, recipe);
  drawMakeupRetouch(context, width, height, recipe);
  drawSculptLight(context, width, height, recipe);
  return recipe;
}
