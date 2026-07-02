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
  assert.equal(recipe.bodyScaleX, 1);
  assert.equal(recipe.eyeGlow, 0);
});

test("maps skin controls to local pixel processing", () => {
  const recipe = createRenderRecipe({
    smooth: 40,
    skinTone: 30,
    whiten: 20,
    acne: 10,
    sculpt: 25,
  });

  assert.ok(recipe.brightness > 1);
  assert.ok(recipe.saturation > 1);
  assert.ok(recipe.contrast > 1);
  assert.ok(recipe.smoothingOpacity > 0);
  assert.ok(recipe.smoothingRadius > 1.2);
  assert.ok(recipe.sculpt > 0);
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

test("maps eye and body controls to bounded geometry", () => {
  const recipe = createRenderRecipe({
    enlargeEyes: 100,
    iris: 100,
    brightEyes: 100,
    slimBelly: 100,
    slimWaist: 100,
    longLegs: 100,
  });

  assert.equal(recipe.bodyScaleX, 0.9);
  assert.equal(recipe.eyeScale, 1.08);
  assert.equal(recipe.eyeGlow, 0.65);
  assert.ok(recipe.bodyScaleY > 1);
  assert.ok(recipe.bodyScaleY <= 1.08);
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
