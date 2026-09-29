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

export function describeResult({ outcome, number, reason }) {
  switch (outcome) {
    case "sent":
      return `Sent to footsight ✓ (still ${number})`;
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
