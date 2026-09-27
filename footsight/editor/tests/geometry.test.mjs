import { test } from "node:test";
import assert from "node:assert/strict";

import {
  acrossPitchLine,
  arrowHead,
  catmullRom,
  densify,
  invert3,
  makeProjector,
  parseOverlays,
  projectPolygon,
  ribbonPolygon,
  ringPolygon,
  serializeOverlays,
  simplifyRDP,
  smoothPath,
  snapPlayer,
} from "../geometry.js";

// A broadcast-like camera: pitch metres -> image pixels, with perspective
// (far touchline at the top, horizon above the image).
const PITCH_TO_IMAGE = [
  [8, 1.2, 450],
  [0, 3, 400],
  [0, 0.004, 1],
];
const SCENE = {
  version: 1,
  image: { width: 900, height: 600 },
  pitch: { length: 105, width: 68 },
  pitch_to_image: PITCH_TO_IMAGE,
  image_to_pitch: invert3(PITCH_TO_IMAGE),
  players: [
    { id: 0, feet_m: [0, 0], feet: null, box: null },
    { id: 1, feet_m: [10, 5], feet: null, box: null },
  ],
};
const P = makeProjector(SCENE);
for (const player of SCENE.players) {
  player.feet = P.toImage(player.feet_m);
  const [x, y] = player.feet;
  player.box = [x - 10, y - 60, x + 10, y];
}

const close = (a, b, eps = 1e-6) => assert.ok(Math.abs(a - b) <= eps, `${a} != ${b}`);
const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);

test("projection round-trips between pitch metres and image pixels", () => {
  const px = P.toImage([12.5, -7]);
  const back = P.toPitch(px);
  close(back[0], 12.5);
  close(back[1], -7);
});

test("points beyond the horizon have no image position", () => {
  // w = 0.004*y + 1 is negative for y < -250: behind the camera
  assert.equal(P.toImage([0, -400]), null);
});

test("a click inside a player's box snaps to that player", () => {
  const [x, y] = SCENE.players[1].feet;
  assert.equal(snapPlayer(SCENE, P, [x, y - 30]), 1);
});

test("a click near a player's feet on the pitch snaps to them", () => {
  const near = P.toImage([2, 1]); // ~2.2 m from player 0
  assert.equal(snapPlayer(SCENE, P, near), 0);
  const far = P.toImage([-20, -20]);
  assert.equal(snapPlayer(SCENE, P, far), null);
});

test("a ring is a circle of the given radius around the feet", () => {
  const ring = ringPolygon([3, 4], 1.3, 48);
  assert.equal(ring.length, 48);
  for (const p of ring) close(dist(p, [3, 4]), 1.3, 1e-9);
});

test("densify keeps points no further apart than the step", () => {
  const path = densify([[0, 0], [1, 0], [1, 2]], 0.25);
  for (let i = 1; i < path.length; i++) assert.ok(dist(path[i], path[i - 1]) <= 0.25 + 1e-9);
  assert.deepEqual(path[0], [0, 0]);
  assert.deepEqual(path.at(-1), [1, 2]);
});

test("a ribbon is as wide as asked", () => {
  const ribbon = ribbonPolygon([[0, 0], [5, 0], [10, 0]], 0.55);
  assert.equal(ribbon.length, 6);
  close(dist(ribbon[1], ribbon[4]), 0.55);
});

test("an arrow head points along the direction", () => {
  const [tip, left, right] = arrowHead([10, 0], [1, 0], 2);
  assert.ok(tip[0] > left[0] && tip[0] > right[0]);
  close(left[1], -right[1]);
});

test("simplifyRDP drops points that add nothing", () => {
  const simple = simplifyRDP([[0, 0], [1, 0.01], [2, 0], [3, 0.02], [4, 0]], 0.5);
  assert.deepEqual(simple, [[0, 0], [4, 0]]);
});

test("catmullRom passes through the points", () => {
  const curve = catmullRom([[0, 0], [5, 5], [10, 0]], 8);
  assert.deepEqual(curve[0], [0, 0]);
  assert.deepEqual(curve.at(-1), [10, 0]);
  assert.ok(curve.some((p) => dist(p, [5, 5]) < 1e-9));
});

test("smoothPath keeps a shaky drag's start and end", () => {
  const shaky = [[0, 0], [1, 0.2], [2, -0.1], [3, 0.15], [6, 3], [9, 6.1], [12, 9]];
  const smooth = smoothPath(shaky);
  assert.deepEqual(smooth[0], [0, 0]);
  assert.deepEqual(smooth.at(-1), [12, 9]);
});

test("a line across the pitch runs touchline to touchline", () => {
  assert.deepEqual(acrossPitchLine(-12.5, SCENE.pitch), [[-12.5, -34], [-12.5, 34]]);
});

test("projectPolygon leaves out points beyond the horizon", () => {
  const projected = projectPolygon(P, [[0, 0], [0, -400], [5, 5]]);
  assert.equal(projected.length, 2);
});

test("overlays survive a save and reload", () => {
  const overlays = [
    { type: "ring", player: 0, color: "yellow" },
    { type: "tag", player: 1, text: "PEDRI", color: "yellow" },
    { type: "arrow", style: "run", points_m: [[0, 0], [5, 2]], color: "white" },
    { type: "arrow", style: "pass", points_m: [[0, 0], [9, 1]], color: "yellow" },
    { type: "link", players: [0, 1], color: "cyan" },
    { type: "line", from_m: [-10, -34], to_m: [-10, 34], dashed: true, color: "cyan" },
    { type: "zone", points_m: [[0, 0], [5, 0], [5, 5]], color: "yellow" },
  ];
  const parsed = parseOverlays(serializeOverlays(overlays, true));
  assert.deepEqual(parsed, { overlays, spotlight: true });
});

test("parseOverlays rejects a file it doesn't understand", () => {
  assert.throws(() => parseOverlays(JSON.stringify({ version: 2, overlays: [] })));
  assert.throws(() => parseOverlays(JSON.stringify({ version: 1, overlays: [{ type: "laser" }] })));
  assert.throws(() => parseOverlays(JSON.stringify({ version: 1, overlays: [{ type: "ring" }] })));
});
