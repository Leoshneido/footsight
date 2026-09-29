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
