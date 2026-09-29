import { test } from "node:test";
import assert from "node:assert/strict";

import { captureHeaders, describeResult, fallbackFilename, isMostlyBlack, STUDIO_URL } from "../capture-core.js";

function pixels(n, rgb) {
  const data = new Uint8ClampedArray(n * 4);
  for (let i = 0; i < n; i++) data.set([...rgb, 255], i * 4);
  return data;
}

test("a black frame (protected video) is recognised", () => {
  assert.equal(isMostlyBlack(pixels(100000, [0, 0, 0])), true);
  assert.equal(isMostlyBlack(pixels(100000, [40, 120, 40])), false);
});

test("a mostly dark but real frame is not mistaken for a black one", () => {
  const data = pixels(100000, [0, 0, 0]);
  for (let i = 0; i < 100000; i += 10) data.set([200, 200, 200, 255], i * 4); // 10% lit
  assert.equal(isMostlyBlack(data), false);
});

test("the capture request marks itself as footsight's and carries the page details", () => {
  const meta = { title: "Barça 2–0 Feyenoord", host: "example.org", time: 3391.2, method: "video" };
  const headers = captureHeaders(meta);
  assert.equal(headers["Content-Type"], "image/png");
  assert.equal(headers["X-Footsight-Capture"], "1");
  // headers must be plain ASCII: accented titles are percent-encoded
  assert.match(headers["X-Footsight-Meta"], /^[\x20-\x7e]+$/);
  assert.deepEqual(JSON.parse(decodeURIComponent(headers["X-Footsight-Meta"])), meta);
});

test("the studio address is the local port the studio listens on", () => {
  assert.equal(STUDIO_URL, "http://127.0.0.1:8765");
});

test("fallback captures are named by time and method", () => {
  const name = fallbackFilename(new Date("2026-09-28T20:45:07.123Z"), "video");
  assert.equal(name, "footsight-captures/still-2026-09-28T20-45-07-video.png");
});

test("messages tell the user what happened", () => {
  assert.equal(describeResult({ outcome: "sent", number: 5 }), "Sent to footsight ✓ (still 5)");
  assert.match(describeResult({ outcome: "saved" }), /isn't running.*Downloads\/footsight-captures/);
  assert.match(describeResult({ outcome: "black" }), /protected/);
  assert.match(describeResult({ outcome: "no-video" }), /No playing video/);
  assert.match(describeResult({ outcome: "error", reason: "boom" }), /boom/);
});

import { defaultProjectName, deliverCapture, popupState } from "../capture-core.js";

test("the popup shows setup, off, starting or running", () => {
  assert.equal(popupState(null, new Error("Specified native messaging host not found.")), "setup");
  assert.equal(popupState({ ok: true, running: false, starting: false }), "off");
  assert.equal(popupState({ ok: true, running: false, starting: true }), "starting");
  assert.equal(popupState({ ok: true, running: true, starting: false }), "running");
});

test("a new project is named after today's date by default", () => {
  assert.equal(defaultProjectName(new Date(2026, 8, 29, 20, 0)), "Match 2026-09-29");
});

function fakes(overrides = {}) {
  const calls = [];
  const deps = {
    sendToStudio: async () => ({ ok: false }),
    host: async (msg) => {
      calls.push(msg.cmd);
      if (msg.cmd === "status") return { ok: true, running: false, starting: false };
      if (msg.cmd === "recent") return { ok: true, projects: [{ name: "Barca v Feyenoord", path: "/p/Barca v Feyenoord" }] };
      if (msg.cmd === "open") return { ok: true, starting: true };
      return { ok: false };
    },
    waitForStudio: async () => true,
    saveToDownloads: async () => calls.push("downloads"),
    keepPending: async () => calls.push("pending"),
    openPopup: async () => calls.push("popup"),
    ...overrides,
  };
  return { deps, calls };
}

test("with the studio running, a capture goes straight in", async () => {
  const { deps, calls } = fakes({ sendToStudio: async () => ({ ok: true, name: "008_camera" }) });
  assert.deepEqual(await deliverCapture("frame", deps), { outcome: "sent", number: 8 });
  assert.deepEqual(calls, []);
});

test("with footsight off, the last project is started and then gets the capture", async () => {
  let attempts = 0;
  const { deps, calls } = fakes({
    sendToStudio: async () => (++attempts === 1 ? { ok: false } : { ok: true, name: "003_camera" }),
    onStarting: async (name) => calls.push(`starting ${name}`),
  });
  const result = await deliverCapture("frame", deps);
  assert.deepEqual(result, { outcome: "sent", number: 3, started: "Barca v Feyenoord" });
  assert.deepEqual(calls, ["status", "recent", "open", "starting Barca v Feyenoord"]);
});

test("with no project yet, the frame is kept and the popup opens", async () => {
  const { deps, calls } = fakes({
    host: async (msg) => {
      calls.push(msg.cmd);
      return msg.cmd === "recent" ? { ok: true, projects: [] } : { ok: true, running: false, starting: false };
    },
  });
  assert.deepEqual(await deliverCapture("frame", deps), { outcome: "pending" });
  assert.deepEqual(calls, ["status", "recent", "pending", "popup"]);
});

test("without the helper set up, the frame goes to Downloads", async () => {
  const { deps, calls } = fakes({ host: async () => { throw new Error("Specified native messaging host not found."); } });
  assert.deepEqual(await deliverCapture("frame", deps), { outcome: "saved" });
  assert.deepEqual(calls, ["downloads"]);
});

test("if footsight takes too long to start, the frame goes to Downloads", async () => {
  const { deps, calls } = fakes({ waitForStudio: async () => false });
  assert.deepEqual(await deliverCapture("frame", deps), { outcome: "timeout" });
  assert.equal(calls.at(-1), "downloads");
});

test("a project that is already starting isn't started twice", async () => {
  let attempts = 0;
  const { deps, calls } = fakes({
    sendToStudio: async () => (++attempts === 1 ? { ok: false } : { ok: true, name: "001_camera" }),
    host: async (msg) => {
      calls.push(msg.cmd);
      return { ok: true, running: false, starting: true, project: { name: "p", path: "/p" } };
    },
  });
  assert.deepEqual(await deliverCapture("frame", deps), { outcome: "sent", number: 1 });
  assert.deepEqual(calls, ["status"]);
});

test("the new outcomes have messages", () => {
  assert.match(describeResult({ outcome: "pending" }), /Start a project/);
  assert.match(describeResult({ outcome: "timeout" }), /took too long.*Downloads/);
  assert.match(describeResult({ outcome: "sent", number: 3, started: "Barca v Feyenoord" }), /Barca v Feyenoord.*still 3/);
});

test("while footsight starts for a capture, the page says so", () => {
  assert.match(describeResult({ outcome: "starting", started: "Barca v Feyenoord" }), /Starting footsight.*Barca v Feyenoord/);
});
