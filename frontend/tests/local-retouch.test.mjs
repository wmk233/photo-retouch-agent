import assert from "node:assert/strict";
import test from "node:test";

import {
  activeAdjustmentCount,
  buildCanvasFilter,
  createRenderRecipe,
  mapWarpPosition,
} from "../js/local-retouch.mjs";

test("creates neutral recipe for empty adjustments", () => {
  const recipe = createRenderRecipe({});

  assert.equal(recipe.brightness, 1);
  assert.equal(recipe.saturation, 1);
  assert.equal(recipe.contrast, 1);
  assert.equal(recipe.faceScaleX, 1);
  assert.equal(recipe.bellyScaleX, 1);
  assert.equal(recipe.legScaleY, 1);
  assert.equal(recipe.eyeGlow, 0);
});

test("maps every skin control to an independent render effect", () => {
  const smooth = createRenderRecipe({ smooth: 100 });
  const skinTone = createRenderRecipe({ skinTone: 100 });
  const whiten = createRenderRecipe({ whiten: 100 });
  const sculpt = createRenderRecipe({ sculpt: 100 });
  const acne = createRenderRecipe({ acne: 100 });
  const wrinkle = createRenderRecipe({ wrinkle: 100 });

  assert.ok(smooth.smoothingOpacity > 0);
  assert.ok(smooth.smoothingRadius > 1.2);
  assert.equal(skinTone.skinToneStrength, 1);
  assert.equal(whiten.whiteningStrength, 1);
  assert.ok(whiten.brightness > 1);
  assert.ok(sculpt.sculpt > 0);
  assert.ok(acne.acneOpacity > 0);
  assert.ok(acne.acneRadius > 1);
  assert.ok(wrinkle.wrinkleOpacity > 0);
  assert.ok(wrinkle.wrinkleRadius > 0.8);
});

test("maps face controls to independent bounded geometry", () => {
  const recipe = createRenderRecipe({
    slimFace: 100,
    smallFace: 100,
    jawline: 100,
    cheekbone: 100,
    chin: 100,
    forehead: 100,
    doubleChin: 100,
  });

  assert.equal(recipe.faceScaleX, 0.92);
  assert.equal(recipe.faceScaleY, 0.96);
  assert.equal(recipe.jawScaleX, 0.9375);
  assert.equal(recipe.cheekScaleX, 0.95);
  assert.ok(recipe.chinScaleY > 1);
  assert.equal(recipe.foreheadScaleY, 0.95);
  assert.ok(recipe.doubleChinScaleY < 1);
});

test("maps every eye control to an independent local effect", () => {
  const enlarge = createRenderRecipe({ enlargeEyes: 100 });
  const eyeBags = createRenderRecipe({ eyeBags: 100 });
  const darkCircles = createRenderRecipe({ darkCircles: 100 });
  const iris = createRenderRecipe({ iris: 100 });
  const brightEyes = createRenderRecipe({ brightEyes: 100 });
  const eyeDistance = createRenderRecipe({ eyeDistance: 100 });
  const eyeAngle = createRenderRecipe({ eyeAngle: 100 });

  assert.equal(enlarge.eyeScale, 1.18);
  assert.equal(eyeBags.eyeBagStrength, 1);
  assert.equal(darkCircles.darkCircleStrength, 1);
  assert.equal(iris.irisStrength, 1);
  assert.ok(iris.eyeGlow > 0);
  assert.equal(brightEyes.brightEyeStrength, 1);
  assert.ok(brightEyes.eyeGlow > iris.eyeGlow);
  assert.ok(eyeDistance.eyeDistanceScale > 1);
  assert.ok(eyeDistance.eyeDistanceScale <= 1.06);
  assert.ok(eyeAngle.eyeAngle > 0);
});

test("maps every facial feature control to an independent local effect", () => {
  const noseBridge = createRenderRecipe({ noseBridge: 100 });
  const slimNose = createRenderRecipe({ slimNose: 100 });
  const teeth = createRenderRecipe({ teeth: 100 });
  const smile = createRenderRecipe({ smile: 100 });
  const lipVolume = createRenderRecipe({ lipVolume: 100 });
  const lipColor = createRenderRecipe({ lipColor: 100 });
  const brows = createRenderRecipe({ brows: 100 });

  assert.equal(noseBridge.noseBridgeStrength, 1);
  assert.ok(slimNose.noseScaleX < 1);
  assert.equal(teeth.teethStrength, 1);
  assert.ok(smile.smileAngle > 0);
  assert.ok(lipVolume.lipScaleY > 1);
  assert.equal(lipColor.lipColorStrength, 1);
  assert.equal(brows.browStrength, 1);
});

test("maps every body control to an independent regional warp", () => {
  const belly = createRenderRecipe({ slimBelly: 100 });
  const waist = createRenderRecipe({ slimWaist: 100 });
  const arms = createRenderRecipe({ slimArms: 100 });
  const legs = createRenderRecipe({ slimLegs: 100 });
  const longLegs = createRenderRecipe({ longLegs: 100 });
  const shoulders = createRenderRecipe({ shoulders: 100 });
  const hips = createRenderRecipe({ hips: 100 });
  const headRatio = createRenderRecipe({ headRatio: 100 });

  assert.equal(belly.bellyScaleX, 0.9);
  assert.ok(waist.waistScaleX < 1);
  assert.ok(arms.armScaleX < 1);
  assert.ok(legs.legScaleX < 1);
  assert.equal(longLegs.legScaleY, 1.1);
  assert.equal(shoulders.shoulderScaleX, 1.08);
  assert.ok(hips.hipScaleX > 1);
  assert.equal(hips.hipScaleY, 0.96);
  assert.ok(headRatio.headRatioScale < 1);
});

test("maps every makeup control to an independent local overlay", () => {
  const foundation = createRenderRecipe({ foundation: 100 });
  const blush = createRenderRecipe({ blush: 100 });
  const contour = createRenderRecipe({ contour: 100 });
  const brows = createRenderRecipe({ eyebrowMakeup: 100 });
  const eyeliner = createRenderRecipe({ eyeliner: 100 });
  const lipstick = createRenderRecipe({ lipstick: 100 });

  assert.equal(foundation.foundationStrength, 1);
  assert.equal(blush.blushStrength, 1);
  assert.equal(contour.contourStrength, 1);
  assert.equal(brows.eyebrowMakeupStrength, 1);
  assert.equal(eyeliner.eyelinerStrength, 1);
  assert.equal(lipstick.lipstickStrength, 1);
});

test("maps every tone control to a visible global or background effect", () => {
  const brightness = createRenderRecipe({ brightness: 100 });
  const contrast = createRenderRecipe({ contrast: 100 });
  const warmth = createRenderRecipe({ warmth: 100 });
  const saturation = createRenderRecipe({ saturation: 100 });
  const clarity = createRenderRecipe({ clarity: 100 });
  const background = createRenderRecipe({ background: 100 });

  assert.ok(brightness.brightness > 1);
  assert.equal(brightness.toneBrightnessStrength, 1);
  assert.ok(contrast.contrast > 1);
  assert.equal(contrast.toneContrastStrength, 1);
  assert.ok(warmth.warmth > 0);
  assert.equal(warmth.toneWarmthStrength, 1);
  assert.ok(saturation.saturation > 1);
  assert.equal(saturation.toneSaturationStrength, 1);
  assert.equal(clarity.clarityStrength, 1);
  assert.equal(background.backgroundStrength, 1);
});

test("clamps invalid and excessive adjustment values", () => {
  const recipe = createRenderRecipe({
    whiten: 999,
    contrast: -50,
    smooth: "not-a-number",
  });

  assert.ok(recipe.brightness <= 1.35);
  assert.equal(recipe.smoothingOpacity, 0);
  assert.equal(recipe.contrast, 1);
});

test("builds a deterministic canvas filter", () => {
  const recipe = createRenderRecipe({ whiten: 20, warmth: 10 });
  const filter = buildCanvasFilter(recipe);

  assert.match(filter, /^brightness\(\d+\.\d{3}\) /);
  assert.match(filter, /saturate\(\d+\.\d{3}\)/);
  assert.match(filter, /contrast\(\d+\.\d{3}\)/);
  assert.match(filter, /sepia\(\d+\.\d{3}\)$/);
});

test("counts active adjustments", () => {
  assert.equal(activeAdjustmentCount({ smooth: 20, whiten: 0, iris: 5 }), 2);
});

test("anchors warp edges while moving the face inward continuously", () => {
  assert.equal(mapWarpPosition(0, 0.92), 0);
  assert.equal(mapWarpPosition(0.5, 0.92), 0.5);
  assert.equal(mapWarpPosition(1, 0.92), 1);
  assert.ok(mapWarpPosition(0.25, 0.92) > 0.25);
  assert.ok(mapWarpPosition(0.75, 0.92) < 0.75);

  const positions = Array.from(
    { length: 21 },
    (_, index) => mapWarpPosition(index / 20, 0.92),
  );
  assert.ok(
    positions.every(
      (position, index) => index === 0 || position > positions[index - 1],
    ),
  );
});
