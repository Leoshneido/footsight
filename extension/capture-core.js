// The parts of the capture extension that don't need Chrome: unit-tested
// with `node --test extension/tests/*.test.mjs`.

// footsight studio listens here (python -m footsight.studio).
export const STUDIO_URL = "http://127.0.0.1:8765";

// A frame that is almost entirely black means the page's video is
// protected (DRM): the browser hands back black pixels instead of the frame.
// Samples every `stride`-th pixel; a real frame, even a dark night game,
// has plenty of lit pixels.
export function isMostlyBlack(rgba, stride = 997, litThreshold = 0.02) {
  let lit = 0;
  let samples = 0;
  for (let i = 0; i < rgba.length; i += 4 * stride) {
    samples++;
    if (rgba[i] + rgba[i + 1] + rgba[i + 2] > 30) lit++;
  }
  return samples === 0 || lit / samples < litThreshold;
}

// Headers for POST /api/capture. The custom header (with the extension's
// Origin) is how the studio knows the still comes from this extension;
// header values must be ASCII, so the page details are percent-encoded JSON.
export function captureHeaders(meta) {
  return {
    "Content-Type": "image/png",
    "X-Footsight-Capture": "1",
    "X-Footsight-Meta": encodeURIComponent(JSON.stringify(meta)),
  };
}

// Where a capture goes when the studio isn't running.
export function fallbackFilename(date, method) {
  const stamp = date.toISOString().slice(0, 19).replace(/:/g, "-");
  return `footsight-captures/still-${stamp}-${method}.png`;
}

export function describeResult({ outcome, number, reason, started }) {
  switch (outcome) {
    case "sent":
      return started ? `Started "${started}" — sent to footsight ✓ (still ${number})` : `Sent to footsight ✓ (still ${number})`;
    case "starting":
      return `Starting footsight on "${started}"… (this takes a few seconds)`;
    case "pending":
      return "No footsight project yet. Start a project in the footsight popup — this frame will go into it.";
    case "timeout":
      return "footsight took too long to start — saved to Downloads/footsight-captures instead";
    case "saved":
      return "footsight studio isn't running — saved to Downloads/footsight-captures instead";
    case "black":
      return "Both capture methods came out black — this video looks protected. Nothing saved.";
    case "no-video":
      return "No playing video found on this page";
    default:
      return `Capture failed: ${reason || "unknown error"}`;
  }
}

// Which screen the popup shows, from the helper's status reply (or the
// error Chrome gives when the helper isn't registered yet).
export function popupState(reply, error) {
  if (error) return /not found|forbidden/i.test(error.message || "") ? "setup" : "error";
  if (!reply || !reply.ok) return "error";
  if (reply.running) return "running";
  return reply.starting ? "starting" : "off";
}

export function defaultProjectName(date) {
  const pad = (n) => String(n).padStart(2, "0");
  return `Match ${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}`;
}

const stillNumber = (name) => Number(String(name).split("_")[0]);

// ⌘⇧S: get the frame into footsight, starting it if needed. `deps` are the
// Chrome-side pieces (see background.js):
//   sendToStudio(frame) -> {ok, name}   host(message) -> reply (throws if no helper)
//   waitForStudio() -> bool             saveToDownloads(frame)
//   keepPending(frame)                  openPopup()
//   onStarting(projectName)  (optional: tell the page footsight is starting)
export async function deliverCapture(frame, deps) {
  const send = async () => {
    try {
      return await deps.sendToStudio(frame);
    } catch {
      return { ok: false };
    }
  };
  let reply = await send();
  if (reply.ok) return { outcome: "sent", number: stillNumber(reply.name) };

  let status;
  try {
    status = await deps.host({ cmd: "status" });
  } catch {
    await deps.saveToDownloads(frame); // helper not set up: never lose the frame
    return { outcome: "saved" };
  }

  let started;
  if (!status.running && !status.starting) {
    const { projects = [] } = await deps.host({ cmd: "recent" });
    if (!projects.length) {
      await deps.keepPending(frame);
      await deps.openPopup();
      return { outcome: "pending" };
    }
    const opened = await deps.host({ cmd: "open", path: projects[0].path });
    if (!opened.ok) {
      await deps.saveToDownloads(frame);
      return { outcome: "saved" };
    }
    started = projects[0].name;
    await deps.onStarting?.(started);
  }

  if (!(await deps.waitForStudio())) {
    await deps.saveToDownloads(frame);
    return { outcome: "timeout" };
  }
  reply = await send();
  if (!reply.ok) {
    await deps.saveToDownloads(frame);
    return { outcome: "saved" };
  }
  return started ? { outcome: "sent", number: stillNumber(reply.name), started } : { outcome: "sent", number: stillNumber(reply.name) };
}
