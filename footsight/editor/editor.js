// footsight overlay editor: a live telestrator over the camera view.
// Graphics are kept in pitch metres (geometry.js) and drawn as SVG between
// the background and figures layers, so they lie on the grass under the
// players. Nothing is saved unless the user exports (E) or saves (Cmd/Ctrl+S).
import {
  acrossPitchLine,
  arrowHead,
  densify,
  makeProjector,
  parseOverlays,
  projectPolygon,
  ribbonPolygon,
  ringPolygon,
  serializeOverlays,
  smoothPath,
  snapPlayer,
} from "/editor/geometry.js";

const SVG_NS = "http://www.w3.org/2000/svg";
const COLORS = { yellow: "#FFD60A", cyan: "#00DCFF", white: "#FFFFFF", red: "#FF3B30" };
const COLOR_KEYS = { 1: "yellow", 2: "cyan", 3: "white", 4: "red" };
const TOOL_KEYS = { h: "highlight", t: "tag", a: "run", p: "pass", k: "link", l: "line", z: "zone" };
const TOOL_COLORS = { highlight: "yellow", tag: "yellow", run: "white", pass: "yellow", link: "cyan", line: "cyan", zone: "yellow" };
const FILL_OPACITY = 0.27;
const TAG_FILL = "#0F1423";

// Sizes on the pitch, in metres.
const RING_RADIUS = 1.3;
const RING_THICKNESS = 0.22;
const RUN_WIDTH = 0.55;
const PASS_WIDTH = 0.4;
const HEAD_SIZE = 2.0;
const LINK_WIDTH = 0.3;
const LINK_DOT = 0.45;
const LINE_WIDTH = 0.3;
const ZONE_BORDER = 0.25;
const DASH = 1.2;
const GAP = 0.8;
const MIN_ARROW = 1.0;
const DRAG_SAMPLE = 0.2;

const $ = (id) => document.getElementById(id);
const svg = $("stage");
const layers = { background: $("background"), ground: $("ground"), preview: $("preview"), figures: $("figures"), veil: $("veil"), tags: $("tags") };

const app = {
  stills: [],
  index: 0,
  sessions: new Map(), // still id -> { scene, projector, overlays, spotlight, undo, redo }
  tool: "highlight",
  color: null, // null: the tool's default
  drawing: null, // the shape in progress
};

// ---------- small helpers ----------

function el(tag, attrs = {}, parent = null) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  if (parent) parent.appendChild(node);
  return node;
}

const pointsAttr = (points) => points.map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join(" ");
const pathRing = (points) => (points.length ? `M${pointsAttr(points).replaceAll(" ", " L")} Z` : "");
const clone = (value) => JSON.parse(JSON.stringify(value));
const session = () => app.sessions.get(app.stills[app.index]?.id);

let noticeTimer = null;
function notice(text, ms = 2200) {
  const box = $("notice");
  box.textContent = text;
  box.classList.add("show");
  clearTimeout(noticeTimer);
  noticeTimer = setTimeout(() => box.classList.remove("show"), ms);
}

function toImagePoint(event) {
  const pt = svg.createSVGPoint();
  pt.x = event.clientX;
  pt.y = event.clientY;
  const p = pt.matrixTransform(svg.getScreenCTM().inverse());
  return [p.x, p.y];
}

function toClientPoint([x, y]) {
  const pt = svg.createSVGPoint();
  pt.x = x;
  pt.y = y;
  const p = pt.matrixTransform(svg.getScreenCTM());
  return [p.x, p.y];
}

const colorOf = (overlay) => COLORS[overlay.color] || COLORS.yellow;
const toolColor = () => app.color || TOOL_COLORS[app.tool];

// ---------- shapes: overlay -> SVG, in camera-view pixels ----------

function pathLength(path) {
  let total = 0;
  for (let i = 1; i < path.length; i++) total += Math.hypot(path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1]);
  return total;
}

// Cut a path (metres) into dash pieces along its length.
function dashes(path, dash = DASH, gap = GAP) {
  const pts = densify(path, 0.1);
  const pieces = [];
  let current = [pts[0]], travelled = 0, inDash = true, boundary = dash;
  for (let i = 1; i < pts.length; i++) {
    travelled += Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]);
    if (inDash) current.push(pts[i]);
    if (travelled >= boundary) {
      if (inDash && current.length > 1) pieces.push(current);
      inDash = !inDash;
      boundary += inDash ? dash : gap;
      current = [pts[i]];
    }
  }
  if (inDash && current.length > 1) pieces.push(current);
  return pieces;
}

// Drop the last `length` metres of a path (the arrow head covers them).
function trimEnd(path, length) {
  const pts = densify(path, 0.1);
  let remaining = length;
  while (pts.length > 2 && remaining > 0) {
    const [a, b] = [pts.at(-2), pts.at(-1)];
    remaining -= Math.hypot(b[0] - a[0], b[1] - a[1]);
    pts.pop();
  }
  return pts;
}

function ribbonShape(P, path, width, color, opacity = 1) {
  const polygon = projectPolygon(P, ribbonPolygon(densify(path, 0.25), width));
  return polygon.length > 2 ? { tag: "polygon", attrs: { points: pointsAttr(polygon), fill: color, "fill-opacity": opacity } } : null;
}

function groundShapes(overlay, scene, P) {
  const color = colorOf(overlay);
  const feet = (id) => scene.players[id]?.feet_m;
  const shapes = [];
  switch (overlay.type) {
    case "ring": {
      const c = feet(overlay.player);
      if (!c) break;
      const outer = projectPolygon(P, ringPolygon(c, RING_RADIUS + RING_THICKNESS / 2, 64));
      const inner = projectPolygon(P, ringPolygon(c, RING_RADIUS - RING_THICKNESS / 2, 64));
      shapes.push({ tag: "path", attrs: { d: `${pathRing(outer)} ${pathRing(inner)}`, "fill-rule": "evenodd", fill: color } });
      break;
    }
    case "arrow": {
      const path = overlay.points_m;
      if (pathLength(path) < 0.5) break;
      const body = trimEnd(path, HEAD_SIZE * 0.8);
      const width = overlay.style === "pass" ? PASS_WIDTH : RUN_WIDTH;
      if (overlay.style === "pass") dashes(body).forEach((piece) => shapes.push(ribbonShape(P, piece, width, color)));
      else shapes.push(ribbonShape(P, body, width, color));
      const tip = path.at(-1);
      const back = trimEnd(path, 1.0).at(-1);
      const head = projectPolygon(P, arrowHead(tip, [tip[0] - back[0], tip[1] - back[1]], HEAD_SIZE));
      if (head.length === 3) shapes.push({ tag: "polygon", attrs: { points: pointsAttr(head), fill: color } });
      break;
    }
    case "link": {
      const pts = overlay.players.map(feet).filter(Boolean);
      if (pts.length >= 2) shapes.push(ribbonShape(P, pts, LINK_WIDTH, color));
      for (const c of pts) {
        const dot = projectPolygon(P, ringPolygon(c, LINK_DOT, 32));
        if (dot.length > 2) shapes.push({ tag: "polygon", attrs: { points: pointsAttr(dot), fill: color } });
      }
      break;
    }
    case "line": {
      const path = [overlay.from_m, overlay.to_m];
      if (overlay.dashed) dashes(path).forEach((piece) => shapes.push(ribbonShape(P, piece, LINE_WIDTH, color)));
      else shapes.push(ribbonShape(P, path, LINE_WIDTH, color));
      break;
    }
    case "zone": {
      const loop = [...overlay.points_m, overlay.points_m[0]];
      const area = projectPolygon(P, densify(loop, 0.25));
      if (area.length > 2) shapes.push({ tag: "polygon", attrs: { points: pointsAttr(area), fill: color, "fill-opacity": FILL_OPACITY } });
      shapes.push(ribbonShape(P, loop, ZONE_BORDER, color));
      break;
    }
  }
  return shapes.filter(Boolean);
}

// Tags float above the player's head at a fixed size on screen.
function tagLayout(overlay, scene, measure) {
  const player = scene.players[overlay.player];
  if (!player) return null;
  const size = Math.round(scene.image.height * 0.02);
  const width = measure(overlay.text, size) + size * 1.1;
  const height = size * 1.55;
  const [x1, y1, x2] = player.box;
  const cx = (x1 + x2) / 2;
  const bottom = y1 - size * 0.45;
  return { x: cx - width / 2, y: bottom - height, width, height, cx, cy: bottom - height / 2, size, text: overlay.text, color: colorOf(overlay) };
}

let measureCanvas = null;
function measureText(text, size) {
  measureCanvas ||= document.createElement("canvas").getContext("2d");
  measureCanvas.font = `600 ${size}px Kanit, system-ui, sans-serif`;
  return measureCanvas.measureText(text).width;
}

// ---------- render ----------

function drawShapes(group, shapes) {
  for (const shape of shapes) el(shape.tag, shape.attrs, group);
}

function render() {
  const s = session();
  layers.ground.replaceChildren();
  layers.tags.replaceChildren();
  if (!s) return;
  for (const overlay of s.overlays) {
    if (overlay.type === "tag") {
      const t = tagLayout(overlay, s.scene, measureText);
      if (!t) continue;
      const g = el("g", {}, layers.tags);
      el("rect", { x: t.x, y: t.y, width: t.width, height: t.height, rx: t.size * 0.28, fill: TAG_FILL, "fill-opacity": 0.9, stroke: t.color, "stroke-width": t.size * 0.08 }, g);
      const text = el("text", { x: t.cx, y: t.cy, "text-anchor": "middle", "dominant-baseline": "central", fill: "#FFFFFF", "font-family": "Kanit, system-ui, sans-serif", "font-weight": 600, "font-size": t.size }, g);
      text.textContent = t.text;
    } else {
      drawShapes(layers.ground, groundShapes(overlay, s.scene, s.projector));
    }
  }
  renderSpotlight(s);
  renderPreview();
  updateToolbar();
}

function spotlightHoles(s) {
  return s.overlays
    .filter((o) => o.type === "ring" && s.scene.players[o.player])
    .map((o) => {
      const [x1, y1, x2, y2] = s.scene.players[o.player].box;
      const h = y2 - y1;
      return { cx: (x1 + x2) / 2, cy: y2 - h * 0.45, rx: Math.max((x2 - x1) * 0.95, h * 0.55), ry: h * 0.8 };
    });
}

function renderSpotlight(s) {
  const holes = $("spot-holes");
  holes.replaceChildren();
  for (const h of spotlightHoles(s)) el("ellipse", { ...h, fill: "black", filter: "url(#soft)" }, holes);
  layers.veil.setAttribute("visibility", s.spotlight ? "visible" : "hidden");
}

function renderPreview() {
  layers.preview.replaceChildren();
  const s = session();
  const d = app.drawing;
  if (!s || !d) return;
  const overlay = previewOverlay(d);
  if (overlay) drawShapes(layers.preview, groundShapes(overlay, s.scene, s.projector));
}

function previewOverlay(d) {
  const color = d.color;
  if ((d.type === "run" || d.type === "pass") && d.points.length > 1) {
    return { type: "arrow", style: d.type, points_m: smoothPath(d.points), color };
  }
  if (d.type === "line" && d.end) return { type: "line", ...lineEnds(d), color };
  if (d.type === "link" && d.players.length) {
    const players = d.hover !== null && d.hover !== d.players.at(-1) ? [...d.players, d.hover] : d.players;
    return players.length >= 2 ? { type: "link", players, color } : { type: "ring", player: d.players[0], color };
  }
  if (d.type === "zone" && d.points.length) {
    const pts = d.cursor ? [...d.points, d.cursor] : d.points;
    return pts.length >= 3 ? { type: "zone", points_m: pts, color } : { type: "line", from_m: pts[0], to_m: pts.at(-1), dashed: false, color };
  }
  return null;
}

function lineEnds(d) {
  const s = session();
  if (d.across) {
    const [from_m, to_m] = acrossPitchLine(d.end[0], s.scene.pitch);
    return { from_m, to_m, dashed: d.dashed };
  }
  return { from_m: d.start, to_m: d.end, dashed: d.dashed };
}

// ---------- history ----------

function commit(change) {
  const s = session();
  if (!s) return;
  s.undo.push({ overlays: clone(s.overlays), spotlight: s.spotlight });
  s.redo = [];
  change(s);
  render();
}

function undo() {
  const s = session();
  if (!s || !s.undo.length) return notice("Nothing to undo", 1200);
  s.redo.push({ overlays: clone(s.overlays), spotlight: s.spotlight });
  Object.assign(s, s.undo.pop());
  render();
}

function redo() {
  const s = session();
  if (!s || !s.redo.length) return notice("Nothing to redo", 1200);
  s.undo.push({ overlays: clone(s.overlays), spotlight: s.spotlight });
  Object.assign(s, s.redo.pop());
  render();
}

// ---------- tools ----------

function setTool(tool) {
  cancelDrawing();
  app.tool = tool;
  app.color = null;
  updateToolbar();
}

function cancelDrawing() {
  app.drawing = null;
  hideTagInput();
  renderPreview();
}

function finishShape() {
  const d = app.drawing;
  if (!d) return;
  if (d.type === "link" && d.players.length >= 2) {
    commit((s) => s.overlays.push({ type: "link", players: d.players, color: d.color }));
  } else if (d.type === "zone" && d.points.length >= 3) {
    commit((s) => s.overlays.push({ type: "zone", points_m: d.points, color: d.color }));
  }
  app.drawing = null;
  renderPreview();
}

function onPointerDown(event) {
  const s = session();
  if (!s || event.button !== 0) return;
  const px = toImagePoint(event);
  const m = s.projector.toPitch(px);
  const player = snapPlayer(s.scene, s.projector, px);
  const color = toolColor();

  switch (app.tool) {
    case "highlight": {
      if (player === null) return notice("Click a player to highlight", 1400);
      const existing = s.overlays.findIndex((o) => o.type === "ring" && o.player === player);
      commit((st) => (existing >= 0 ? st.overlays.splice(existing, 1) : st.overlays.push({ type: "ring", player, color })));
      return;
    }
    case "tag": {
      if (player === null) return notice("Click a player to tag", 1400);
      return showTagInput(player, color);
    }
    case "run":
    case "pass": {
      if (!m) return;
      const start = player !== null ? s.scene.players[player].feet_m : m;
      app.drawing = { type: app.tool, points: [start], color };
      svg.setPointerCapture(event.pointerId);
      return;
    }
    case "line": {
      if (!m) return;
      app.drawing = { type: "line", start: m, end: null, across: event.shiftKey, dashed: event.altKey, color };
      svg.setPointerCapture(event.pointerId);
      return;
    }
    case "link": {
      if (player === null) return notice("Click players to link — Enter to finish", 1600);
      app.drawing ||= { type: "link", players: [], hover: null, color };
      if (app.drawing.players.at(-1) !== player) app.drawing.players.push(player);
      renderPreview();
      return;
    }
    case "zone": {
      if (!m) return;
      app.drawing ||= { type: "zone", points: [], cursor: null, color };
      const last = app.drawing.points.at(-1);
      if (!last || Math.hypot(last[0] - m[0], last[1] - m[1]) > 0.3) app.drawing.points.push(m);
      renderPreview();
      return;
    }
  }
}

function onPointerMove(event) {
  const s = session();
  const d = app.drawing;
  if (!s || !d) return;
  const px = toImagePoint(event);
  const m = s.projector.toPitch(px);
  if ((d.type === "run" || d.type === "pass") && m) {
    const last = d.points.at(-1);
    if (Math.hypot(m[0] - last[0], m[1] - last[1]) >= DRAG_SAMPLE) d.points.push(m);
  } else if (d.type === "line" && m) {
    d.end = m;
    d.across = event.shiftKey;
    d.dashed = event.altKey;
  } else if (d.type === "link") {
    d.hover = snapPlayer(s.scene, s.projector, px);
  } else if (d.type === "zone") {
    d.cursor = m;
  }
  renderPreview();
}

function onPointerUp(event) {
  const d = app.drawing;
  if (!d) return;
  if (d.type === "run" || d.type === "pass") {
    const points = smoothPath(d.points);
    if (pathLength(points) >= MIN_ARROW) commit((s) => s.overlays.push({ type: "arrow", style: d.type, points_m: points, color: d.color }));
    app.drawing = null;
  } else if (d.type === "line") {
    const long = d.end && (d.across || Math.hypot(d.end[0] - d.start[0], d.end[1] - d.start[1]) >= MIN_ARROW);
    if (long) commit((s) => s.overlays.push({ type: "line", ...lineEnds(d), color: d.color }));
    app.drawing = null;
  }
  if (svg.hasPointerCapture?.(event.pointerId)) svg.releasePointerCapture(event.pointerId);
  renderPreview();
}

// ---------- tag input ----------

function showTagInput(player, color) {
  const s = session();
  const input = $("tag-input");
  const [x1, y1, x2] = s.scene.players[player].box;
  const [cx, cy] = toClientPoint([(x1 + x2) / 2, y1]);
  const existing = s.overlays.find((o) => o.type === "tag" && o.player === player);
  input.value = existing ? existing.text : "";
  input.style.borderColor = COLORS[color];
  input.style.left = `${Math.max(8, cx - 110)}px`;
  input.style.top = `${Math.max(8, cy - 70)}px`;
  input.hidden = false;
  input.dataset.player = player;
  input.dataset.color = color;
  input.focus();
}

function hideTagInput() {
  const input = $("tag-input");
  input.hidden = true;
  input.blur();
}

$("tag-input").addEventListener("keydown", (event) => {
  event.stopPropagation();
  const input = event.currentTarget;
  if (event.key === "Escape") return hideTagInput();
  if (event.key !== "Enter") return;
  const player = Number(input.dataset.player);
  const text = input.value.trim();
  const color = input.dataset.color;
  hideTagInput();
  commit((s) => {
    s.overlays = s.overlays.filter((o) => !(o.type === "tag" && o.player === player));
    if (text) s.overlays.push({ type: "tag", player, text, color });
  });
});

// ---------- stills ----------

async function loadStill(index) {
  cancelDrawing();
  app.index = (index + app.stills.length) % app.stills.length;
  const still = app.stills[app.index];
  if (!app.sessions.has(still.id)) {
    const scene = await (await fetch(`/api/stills/${still.id}/scene`)).json();
    const s = { scene, projector: makeProjector(scene), overlays: [], spotlight: false, undo: [], redo: [] };
    if (still.has_overlays) {
      try {
        const saved = parseOverlays(await (await fetch(`/api/stills/${still.id}/overlays`)).text());
        s.overlays = saved.overlays;
        s.spotlight = saved.spotlight;
      } catch (error) {
        notice(`Couldn't read saved overlays: ${error.message}`, 4000);
      }
    }
    app.sessions.set(still.id, s);
  }
  const { width, height } = session().scene.image;
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  for (const [layer, name] of [[layers.background, "background"], [layers.figures, "figures"]]) {
    layer.setAttribute("href", `/api/stills/${still.id}/${name}`);
    layer.setAttribute("width", width);
    layer.setAttribute("height", height);
  }
  for (const node of [layers.veil, $("spot-mask-bg")]) {
    node.setAttribute("width", width);
    node.setAttribute("height", height);
  }
  $("still-name").textContent = `${still.name} · ${app.index + 1}/${app.stills.length}`;
  render();
}

// ---------- export & save ----------

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error(`Couldn't load ${src}`));
    img.src = src;
  });
}

async function svgLayerImage(inner, width, height) {
  const defs = svg.querySelector("defs").outerHTML;
  const markup = `<svg xmlns="${SVG_NS}" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">${defs}${inner}</svg>`;
  const url = URL.createObjectURL(new Blob([markup], { type: "image/svg+xml" }));
  try {
    return await loadImage(url);
  } finally {
    URL.revokeObjectURL(url);
  }
}

async function exportPng() {
  const s = session();
  if (!s) return;
  notice("Exporting…", 10000);
  try {
    await document.fonts.load("600 40px Kanit");
    const { width, height } = s.scene.image;
    const id = app.stills[app.index].id;
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(await loadImage(`/api/stills/${id}/background`), 0, 0);
    ctx.drawImage(await svgLayerImage(layers.ground.outerHTML, width, height), 0, 0);
    ctx.drawImage(await loadImage(`/api/stills/${id}/figures`), 0, 0);
    if (s.spotlight) ctx.drawImage(await svgLayerImage(layers.veil.outerHTML, width, height), 0, 0);
    // Tags go straight onto the canvas: an SVG drawn as an image can't use the page's fonts.
    for (const overlay of s.overlays.filter((o) => o.type === "tag")) {
      const t = tagLayout(overlay, s.scene, measureText);
      if (!t) continue;
      ctx.beginPath();
      ctx.roundRect(t.x, t.y, t.width, t.height, t.size * 0.28);
      ctx.fillStyle = "rgba(15, 20, 35, 0.9)";
      ctx.fill();
      ctx.lineWidth = t.size * 0.08;
      ctx.strokeStyle = t.color;
      ctx.stroke();
      ctx.font = `600 ${t.size}px Kanit, system-ui, sans-serif`;
      ctx.fillStyle = "#FFFFFF";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(t.text, t.cx, t.cy);
    }
    const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/png"));
    const response = await fetch(`/api/stills/${id}/export`, { method: "POST", headers: { "Content-Type": "image/png" }, body: blob });
    if (!response.ok) throw new Error(await response.text());
    notice(`Exported ${(await response.json()).saved}`);
  } catch (error) {
    notice(`Export failed: ${error.message}`, 5000);
  }
}

async function saveOverlays() {
  const s = session();
  if (!s) return;
  const still = app.stills[app.index];
  try {
    const response = await fetch(`/api/stills/${still.id}/overlays`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: serializeOverlays(s.overlays, s.spotlight),
    });
    if (!response.ok) throw new Error(await response.text());
    still.has_overlays = true;
    notice(`Saved ${(await response.json()).saved}`);
  } catch (error) {
    notice(`Save failed: ${error.message}`, 5000);
  }
}

// ---------- toolbar, keys ----------

function updateToolbar() {
  for (const button of document.querySelectorAll("[data-tool]")) button.classList.toggle("active", button.dataset.tool === app.tool);
  for (const button of document.querySelectorAll("[data-color]")) button.classList.toggle("active", button.dataset.color === toolColor());
  document.querySelector('[data-action="spotlight"]').classList.toggle("active", Boolean(session()?.spotlight));
}

function toggleSpotlight() {
  const s = session();
  if (!s) return;
  if (!s.spotlight && !s.overlays.some((o) => o.type === "ring")) notice("Tip: highlight players (H) to leave them lit", 2200);
  commit((st) => (st.spotlight = !st.spotlight));
}

function toggleFullscreen() {
  if (document.fullscreenElement) document.exitFullscreen();
  else document.documentElement.requestFullscreen?.();
}

const actions = {
  spotlight: toggleSpotlight,
  undo,
  clear: () => commit((s) => { s.overlays = []; s.spotlight = false; }),
  prev: () => loadStill(app.index - 1),
  next: () => loadStill(app.index + 1),
  save: saveOverlays,
  export: exportPng,
  fullscreen: toggleFullscreen,
  help: () => ($("help").hidden = !$("help").hidden),
};

document.addEventListener("keydown", (event) => {
  if (!$("tag-input").hidden) return;
  const key = event.key.toLowerCase();
  const mod = event.metaKey || event.ctrlKey;
  if (mod && key === "z") {
    event.preventDefault();
    return event.shiftKey ? redo() : undo();
  }
  if (mod && key === "s") {
    event.preventDefault();
    return saveOverlays();
  }
  if (mod || event.altKey && key !== "alt") return;
  if (TOOL_KEYS[key]) return setTool(TOOL_KEYS[key]);
  if (COLOR_KEYS[key]) {
    app.color = COLOR_KEYS[key];
    if (app.drawing) app.drawing.color = app.color;
    updateToolbar();
    return renderPreview();
  }
  if (key === "escape") return cancelDrawing();
  if (key === "enter") return finishShape();
  if (key === "arrowleft") return actions.prev();
  if (key === "arrowright") return actions.next();
  if (key === "s") return toggleSpotlight();
  if (key === "c") return actions.clear();
  if (key === "e") return exportPng();
  if (key === "f") return toggleFullscreen();
  if (key === "?") return actions.help();
});

for (const button of document.querySelectorAll("[data-tool]")) button.addEventListener("click", () => setTool(button.dataset.tool));
for (const button of document.querySelectorAll("[data-color]")) {
  button.addEventListener("click", () => {
    app.color = button.dataset.color;
    updateToolbar();
  });
}
for (const button of document.querySelectorAll("[data-action]")) button.addEventListener("click", () => actions[button.dataset.action]());

svg.addEventListener("pointerdown", onPointerDown);
svg.addEventListener("pointermove", onPointerMove);
svg.addEventListener("pointerup", onPointerUp);
svg.addEventListener("dblclick", finishShape);

// The toolbar slides away so the still fills the recording; it comes back
// when the mouse reaches the top edge.
let hideTimer = null;
function scheduleHide() {
  clearTimeout(hideTimer);
  hideTimer = setTimeout(() => $("toolbar").classList.add("hidden"), 1800);
}
document.addEventListener("mousemove", (event) => {
  const toolbar = $("toolbar");
  if (event.clientY < 70 || toolbar.contains(event.target)) {
    toolbar.classList.remove("hidden");
    clearTimeout(hideTimer);
  } else if (!toolbar.classList.contains("hidden")) {
    scheduleHide();
  }
});

// ---------- start ----------

async function start() {
  try {
    const listing = await (await fetch("/api/stills")).json();
    app.stills = listing.stills;
    if (listing.skipped.length) notice(`Skipped (re-run the pipeline): ${listing.skipped.join(", ")}`, 6000);
    if (!app.stills.length) return notice("No editable stills — run the pipeline first", 60000);
    await loadStill(0);
    updateToolbar();
    scheduleHide();
  } catch (error) {
    notice(`Couldn't start the editor: ${error.message}`, 60000);
  }
}

start();
