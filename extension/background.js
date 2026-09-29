// footsight capture extension: ⌘⇧S (or the toolbar button) grabs the
// current video frame and sends it to footsight studio.
//
// 1. "video": draw the page's <video> frame to a canvas -- the full video
//    resolution (the method the user's site allows; see MEMORY.md).
// 2. "screenshot": if the video can't be read, capture the visible tab and
//    crop it to the player.
// Nothing here bypasses any protection: if both come out black, it says so.
import { captureHeaders, describeResult, fallbackFilename, isMostlyBlack, STUDIO_URL } from "./capture-core.js";

chrome.commands.onCommand.addListener(async (command, tab) => {
  if (command === "capture") await capture(tab);
});
chrome.action.onClicked.addListener(capture);

// If Chrome didn't give us ⌘⇧S (something else already uses it), say so on
// the toolbar button; the button itself always captures.
chrome.runtime.onInstalled.addListener(async () => {
  const commands = await chrome.commands.getAll();
  const shortcut = commands.find((c) => c.name === "capture")?.shortcut;
  if (!shortcut) {
    await chrome.action.setBadgeText({ text: "!" });
    await chrome.action.setTitle({ title: "footsight: no shortcut assigned — set one at chrome://extensions/shortcuts (clicking here still captures)" });
  }
});

// Runs inside the page (every frame): the biggest visible video, and its
// current frame when the page lets us read it.
function grabFrame() {
  const videos = [...document.querySelectorAll("video")]
    .map((v) => ({ v, r: v.getBoundingClientRect() }))
    .filter(({ v, r }) => v.readyState >= 2 && r.width > 50 && r.height > 50);
  if (!videos.length) return { found: false };
  const { v, r } = videos.sort((a, b) => b.r.width * b.r.height - a.r.width * a.r.height)[0];
  const info = {
    found: true,
    inTopFrame: window === window.top,
    rect: { x: r.left, y: r.top, width: r.width, height: r.height },
    dpr: window.devicePixelRatio,
    videoSize: [v.videoWidth, v.videoHeight],
    time: v.currentTime,
    title: (window.top === window ? document.title : "") || document.title,
    host: location.hostname,
  };
  try {
    const canvas = document.createElement("canvas");
    canvas.width = v.videoWidth;
    canvas.height = v.videoHeight;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(v, 0, 0);
    const { data } = ctx.getImageData(0, 0, canvas.width, canvas.height);
    return { ...info, pixels: Array.from(data.filter((_, i) => i % (4 * 997) < 4)), dataUrl: canvas.toDataURL("image/png") };
  } catch (error) {
    return { ...info, blocked: error.name };
  }
}

function toast(message, ok) {
  const box = document.createElement("div");
  box.textContent = message;
  Object.assign(box.style, {
    position: "fixed", zIndex: 2147483647, top: "16px", left: "50%", transform: "translateX(-50%)",
    background: ok ? "#0f1423" : "#5a1111", color: "#fff", border: `2px solid ${ok ? "#FFD60A" : "#FF3B30"}`,
    borderRadius: "8px", padding: "10px 16px", font: "600 15px system-ui, sans-serif",
  });
  document.documentElement.appendChild(box);
  setTimeout(() => box.remove(), 3500);
}

async function report(tabId, outcome, details = {}) {
  const ok = outcome === "sent" || outcome === "saved";
  await chrome.scripting.executeScript({ target: { tabId }, func: toast, args: [describeResult({ outcome, ...details }), ok] }).catch(() => {});
}

async function blobPixels(blob) {
  const bitmap = await createImageBitmap(blob);
  const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
  const ctx = canvas.getContext("2d");
  ctx.drawImage(bitmap, 0, 0);
  return ctx.getImageData(0, 0, bitmap.width, bitmap.height).data;
}

async function cropScreenshot(dataUrl, rect, dpr) {
  const bitmap = await createImageBitmap(await (await fetch(dataUrl)).blob());
  const x = Math.max(0, Math.round(rect.x * dpr)), y = Math.max(0, Math.round(rect.y * dpr));
  const w = Math.min(bitmap.width - x, Math.round(rect.width * dpr)), h = Math.min(bitmap.height - y, Math.round(rect.height * dpr));
  const canvas = new OffscreenCanvas(w, h);
  canvas.getContext("2d").drawImage(bitmap, x, y, w, h, 0, 0, w, h);
  return canvas.convertToBlob({ type: "image/png" });
}

async function blobToDataUrl(blob) {
  const bytes = new Uint8Array(await blob.arrayBuffer());
  let binary = "";
  for (let i = 0; i < bytes.length; i += 0x8000) binary += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return `data:image/png;base64,${btoa(binary)}`;
}

async function capture(tab) {
  let results;
  try {
    results = await chrome.scripting.executeScript({ target: { tabId: tab.id, allFrames: true }, func: grabFrame });
  } catch (error) {
    return report(tab.id, "error", { reason: `can't run on this page (${error.message})` });
  }
  const hits = results.map((r) => r.result).filter((r) => r && r.found);
  if (!hits.length) return report(tab.id, "no-video");
  const hit = hits.sort((a, b) => b.rect.width * b.rect.height - a.rect.width * a.rect.height)[0];

  let png, method;
  if (hit.dataUrl && !isMostlyBlack(Uint8ClampedArray.from(hit.pixels), 1)) {
    png = await (await fetch(hit.dataUrl)).blob();
    method = "video";
  } else {
    const shot = await chrome.tabs.captureVisibleTab(tab.windowId, { format: "png" });
    const cropped = hit.inTopFrame ? await cropScreenshot(shot, hit.rect, hit.dpr) : await (await fetch(shot)).blob();
    if (isMostlyBlack(await blobPixels(cropped))) return report(tab.id, "black");
    png = cropped;
    method = "screenshot";
  }

  const meta = { title: hit.title, host: hit.host, time: hit.time, method, captured_at: new Date().toISOString() };
  try {
    const response = await fetch(`${STUDIO_URL}/api/capture`, { method: "POST", headers: captureHeaders(meta), body: png });
    if (!response.ok) throw new Error(await response.text());
    const { name } = await response.json();
    return report(tab.id, "sent", { number: Number(name.split("_")[0]) });
  } catch {
    // studio not running (or refused): keep the still rather than lose it
    await chrome.downloads.download({ url: await blobToDataUrl(png), filename: fallbackFilename(new Date(), method) });
    return report(tab.id, "saved");
  }
}
