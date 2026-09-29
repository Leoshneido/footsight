// Pure geometry for the overlay editor: projection between pitch metres
// and camera-view pixels, player snapping, and the shapes the tools draw.
// Everything is built in pitch metres and projected last, so graphics lie on
// the grass in perspective. No DOM here -- unit-tested with `node --test`.

export function invert3(m) {
  const [[a, b, c], [d, e, f], [g, h, i]] = m;
  const A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g;
  const det = a * A + b * B + c * C;
  return [
    [A / det, -(b * i - c * h) / det, (b * f - c * e) / det],
    [B / det, (a * i - c * g) / det, -(a * f - c * d) / det],
    [C / det, -(a * h - b * g) / det, (a * e - b * d) / det],
  ];
}

function applyW(m, [x, y]) {
  return [m[0][0] * x + m[0][1] * y + m[0][2], m[1][0] * x + m[1][1] * y + m[1][2], m[2][0] * x + m[2][1] * y + m[2][2]];
}

// Projections both ways. A point on the far side of the horizon has a
// homogeneous w of the opposite sign to the pitch center's; it would divide
// through to a mirrored, plausible-looking position, so it maps to null.
export function makeProjector(scene) {
  const toImageM = scene.pitch_to_image;
  const toPitchM = scene.image_to_pitch;
  const pitchSign = Math.sign(applyW(toImageM, [0, 0])[2]);
  const centerPx = project(toImageM, [0, 0]);
  const imageSign = Math.sign(applyW(toPitchM, centerPx)[2]);

  function project(m, p) {
    const [x, y, w] = applyW(m, p);
    return [x / w, y / w];
  }
  return {
    toImage(m) {
      const w = applyW(toImageM, m)[2];
      return w === 0 || Math.sign(w) !== pitchSign ? null : project(toImageM, m);
    },
    toPitch(px) {
      const w = applyW(toPitchM, px)[2];
      return w === 0 || Math.sign(w) !== imageSign ? null : project(toPitchM, px);
    },
  };
}

const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);

// Which player a click means: one whose box contains it (the nearest to the
// camera if boxes overlap), else the closest whose feet are within maxMeters
// of the clicked spot on the pitch.
export function snapPlayer(scene, projector, px, maxMeters = 3) {
  const inside = scene.players.filter(({ box: [x1, y1, x2, y2] }) => px[0] >= x1 && px[0] <= x2 && px[1] >= y1 && px[1] <= y2);
  if (inside.length) return inside.reduce((a, b) => (b.feet[1] > a.feet[1] ? b : a)).id;
  const spot = projector.toPitch(px);
  if (!spot) return null;
  let best = null;
  for (const player of scene.players) {
    const d = dist(player.feet_m, spot);
    if (d <= maxMeters && (!best || d < best.d)) best = { id: player.id, d };
  }
  return best ? best.id : null;
}

export function ringPolygon(center, radius = 1.3, n = 48) {
  return Array.from({ length: n }, (_, k) => {
    const t = (2 * Math.PI * k) / n;
    return [center[0] + radius * Math.cos(t), center[1] + radius * Math.sin(t)];
  });
}

// Extra points along a path so no two are more than `step` apart: straight
// lines on the pitch bend under perspective only if they are sampled.
export function densify(path, step = 0.25) {
  const out = [path[0]];
  for (let i = 1; i < path.length; i++) {
    const [a, b] = [path[i - 1], path[i]];
    const n = Math.max(1, Math.ceil(dist(a, b) / step));
    for (let k = 1; k <= n; k++) out.push([a[0] + ((b[0] - a[0]) * k) / n, a[1] + ((b[1] - a[1]) * k) / n]);
  }
  return out;
}

// A strip of the given width along a path (arrow bodies, lines): the left
// edge out, then the right edge back.
export function ribbonPolygon(path, width) {
  const left = [], right = [];
  for (let i = 0; i < path.length; i++) {
    const a = path[Math.max(0, i - 1)], b = path[Math.min(path.length - 1, i + 1)];
    const len = dist(a, b) || 1;
    const nx = -(b[1] - a[1]) / len, ny = (b[0] - a[0]) / len;
    left.push([path[i][0] + (nx * width) / 2, path[i][1] + (ny * width) / 2]);
    right.push([path[i][0] - (nx * width) / 2, path[i][1] - (ny * width) / 2]);
  }
  return [...left, ...right.reverse()];
}

export function arrowHead(tip, direction, size = 2) {
  const len = Math.hypot(direction[0], direction[1]) || 1;
  const d = [direction[0] / len, direction[1] / len];
  const n = [-d[1], d[0]];
  return [
    [tip[0] + d[0] * size * 0.3, tip[1] + d[1] * size * 0.3],
    [tip[0] - d[0] * size + n[0] * size * 0.6, tip[1] - d[1] * size + n[1] * size * 0.6],
    [tip[0] - d[0] * size - n[0] * size * 0.6, tip[1] - d[1] * size - n[1] * size * 0.6],
  ];
}

function pointToSegment(p, a, b) {
  const dx = b[0] - a[0], dy = b[1] - a[1];
  const len2 = dx * dx + dy * dy;
  const t = len2 ? Math.max(0, Math.min(1, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / len2)) : 0;
  return dist(p, [a[0] + t * dx, a[1] + t * dy]);
}

export function simplifyRDP(points, eps = 0.5) {
  if (points.length < 3) return points.slice();
  let worst = 0, index = 0;
  for (let i = 1; i < points.length - 1; i++) {
    const d = pointToSegment(points[i], points[0], points.at(-1));
    if (d > worst) [worst, index] = [d, i];
  }
  if (worst <= eps) return [points[0], points.at(-1)];
  const left = simplifyRDP(points.slice(0, index + 1), eps);
  return [...left.slice(0, -1), ...simplifyRDP(points.slice(index), eps)];
}

// A smooth curve through every point (centripetal feel is not needed at
// this scale; uniform Catmull-Rom with the ends repeated).
export function catmullRom(points, samples = 8) {
  if (points.length < 3) return points.slice();
  const p = [points[0], ...points, points.at(-1)];
  const out = [points[0]];
  for (let i = 1; i < p.length - 2; i++) {
    for (let s = 1; s <= samples; s++) {
      const t = s / samples, t2 = t * t, t3 = t2 * t;
      const f = (k) =>
        0.5 * (2 * p[i][k] + (-p[i - 1][k] + p[i + 1][k]) * t + (2 * p[i - 1][k] - 5 * p[i][k] + 4 * p[i + 1][k] - p[i + 2][k]) * t2 +
          (-p[i - 1][k] + 3 * p[i][k] - 3 * p[i + 1][k] + p[i + 2][k]) * t3);
      out.push(s === samples ? p[i + 1].slice() : [f(0), f(1)]);
    }
  }
  return out;
}

// A shaky hand-drawn drag in metres -> a clean curve with the same ends.
export function smoothPath(points, eps = 0.5) {
  return catmullRom(simplifyRDP(points, eps));
}

// Centres of evenly spaced dots along a path (a dotted pass), the first half
// a spacing in from the start.
export function dotsAlong(path, spacing = 0.9) {
  const dots = [];
  let next = spacing / 2, travelled = 0;
  for (let i = 1; i < path.length; i++) {
    const [a, b] = [path[i - 1], path[i]];
    const length = dist(a, b);
    while (next <= travelled + length + 1e-9) {
      const t = (next - travelled) / length;
      dots.push([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]);
      next += spacing;
    }
    travelled += length;
  }
  return dots;
}

// Whether a click (image px) lands on a zone's first corner -- which closes
// the zone -- once it has at least three corners.
export function nearFirstPoint(projector, points, clickPx, thresholdPx) {
  if (points.length < 3) return false;
  const first = projector.toImage(points[0]);
  return first !== null && dist(first, clickPx) <= thresholdPx;
}

export function acrossPitchLine(x, pitch) {
  return [[x, -pitch.width / 2], [x, pitch.width / 2]];
}

export function projectPolygon(projector, points) {
  return points.map((p) => projector.toImage(p)).filter((p) => p !== null);
}

const COLORS = new Set(["yellow", "cyan", "white", "red"]);
const isPoint = (p) => Array.isArray(p) && p.length === 2 && p.every(Number.isFinite);
const isPoints = (ps, min) => Array.isArray(ps) && ps.length >= min && ps.every(isPoint);
const REQUIRED = {
  ring: (o) => Number.isInteger(o.player),
  tag: (o) => Number.isInteger(o.player) && typeof o.text === "string",
  arrow: (o) => (o.style === "run" || o.style === "pass") && isPoints(o.points_m, 2),
  link: (o) => Array.isArray(o.players) && o.players.length >= 2 && o.players.every(Number.isInteger),
  line: (o) => isPoint(o.from_m) && isPoint(o.to_m) && typeof o.dashed === "boolean",
  zone: (o) => isPoints(o.points_m, 3),
};

export function serializeOverlays(overlays, spotlight) {
  return JSON.stringify({ version: 1, overlays, spotlight: Boolean(spotlight) });
}

export function parseOverlays(text) {
  const data = typeof text === "string" ? JSON.parse(text) : text;
  if (!data || data.version !== 1 || !Array.isArray(data.overlays)) throw new Error("Not a footsight overlays file (version 1)");
  for (const o of data.overlays) {
    const check = REQUIRED[o && o.type];
    if (!check || !check(o) || !COLORS.has(o.color)) throw new Error(`Invalid overlay: ${JSON.stringify(o)}`);
  }
  return { overlays: data.overlays, spotlight: Boolean(data.spotlight) };
}

// ---- studio fixes ----

// The camera view is the still enlarged `scale` times; corrections are in
// the original still's pixels.
export function imageToStill([x, y], scale) {
  return [x / scale, y / scale];
}

// The corrections after one fix, as a new object (the old one stays in the
// undo history). Added players get their ids from the studio; removing one
// takes it back out of `added` instead of listing it as removed.
export function nextCorrections(corrections, action) {
  const next = {
    removed: [...(corrections.removed || [])],
    added: (corrections.added || []).map((a) => ({ ...a })),
    ball: { ...(corrections.ball || { mode: "auto" }) },
    sides: { ...(corrections.sides || {}) },
  };
  switch (action.type) {
    case "remove":
      if (next.added.some((a) => a.id === action.id)) next.added = next.added.filter((a) => a.id !== action.id);
      else if (!next.removed.includes(action.id)) next.removed.push(action.id);
      break;
    case "add":
      next.added.push({ feet: action.feet, category: action.category });
      break;
    case "ball":
      next.ball = { mode: "set", pixel: action.pixel };
      break;
    case "ball-none":
      next.ball = { mode: "none" };
      break;
    case "side": {
      const added = next.added.find((a) => a.id === action.id);
      if (added) added.category = action.category;
      else next.sides[action.id] = action.category;
      break;
    }
    default:
      throw new Error(`Unknown fix: ${action.type}`);
  }
  return next;
}

// Drawings tied to players who no longer exist are dropped; a link keeps
// its remaining players while at least two are left.
export function dropOverlaysFor(overlays, keepIds) {
  const out = [];
  for (const o of overlays) {
    if ((o.type === "ring" || o.type === "tag") && !keepIds.has(o.player)) continue;
    if (o.type === "link") {
      const players = o.players.filter((id) => keepIds.has(id));
      if (players.length < 2) continue;
      out.push({ ...o, players });
      continue;
    }
    out.push(o);
  }
  return out;
}
