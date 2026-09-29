// The footsight popup: start a project, reopen a recent one, or (while
// footsight runs) open the editor, capture, stop. Starting and choosing a
// folder go through background.js, because the popup closes as soon as the
// Finder "Choose folder" dialog takes focus.
import { defaultProjectName, popupState } from "./capture-core.js";

const HOST = "com.footsight.host";
const view = document.getElementById("view");
const messageBox = document.getElementById("message");
let pollTimer;

function el(tag, props = {}, ...children) {
  const node = Object.assign(document.createElement(tag), props);
  node.append(...children);
  return node;
}

function button(label, onclick, className = "") {
  return el("button", { textContent: label, onclick, className, title: label });
}

function showMessage(text, info = false) {
  messageBox.hidden = !text;
  messageBox.textContent = text || "";
  messageBox.className = info ? "info" : "";
}

async function host(message) {
  return chrome.runtime.sendNativeMessage(HOST, message);
}

async function refresh() {
  clearTimeout(pollTimer);
  let reply, error;
  try {
    reply = await host({ cmd: "status" });
  } catch (e) {
    error = e;
  }
  const state = popupState(reply, error);
  if (state === "setup") return renderSetup();
  if (state === "error") return renderError(error?.message || reply?.error);
  if (reply.error) showMessage(reply.error); // the last studio stopped unexpectedly
  if (state === "running") return renderRunning(reply);
  if (state === "starting") return renderStarting(reply);
  return renderOff();
}

function renderSetup() {
  const command = ".venv/bin/python -m footsight.setup_chrome";
  view.replaceChildren(
    el("p", { textContent: "One-time setup: register the footsight helper with Chrome. In Terminal, inside your footsight folder, run:" }),
    el("code", { textContent: command }),
    button("Copy command", async (event) => {
      await navigator.clipboard.writeText(command);
      event.target.textContent = "Copied ✓";
    }),
    el("p", { className: "muted", textContent: "Then reload the extension at chrome://extensions and click the footsight icon again. Until then, ⌘⇧S saves frames to Downloads/footsight-captures." }),
  );
}

function renderError(text) {
  showMessage(`The footsight helper didn't answer: ${text || "unknown error"}`);
  view.replaceChildren(button("Try again", () => { showMessage(""); refresh(); }));
}

async function renderOff(current = null) {
  const [{ projects = [], last_location: lastLocation } = {}, stored] = await Promise.all([
    host({ cmd: "recent" }).catch(() => ({})),
    chrome.storage.session.get(["pendingFrame", "lastError"]),
  ]);
  if (stored.lastError) {
    showMessage(stored.lastError);
    chrome.storage.session.remove("lastError");
  } else if (stored.pendingFrame) {
    showMessage("A captured frame is waiting — it will go into the project you start.", true);
  }

  const name = el("input", { type: "text", value: defaultProjectName(new Date()), placeholder: "Project name" });
  const start = (choose) => startProject({ name: name.value, choose, location: choose ? undefined : lastLocation });
  const children = [el("h2", { textContent: "New project" }), name];
  if (current) {
    // switching from the running view: starting or reopening stops the current project
    children.unshift(
      button("← Back", refresh),
      el("p", { className: "muted", textContent: `Starting another project stops “${current.name}”.` }),
    );
  }
  if (lastLocation) {
    const folder = lastLocation.split("/").filter(Boolean).pop() || lastLocation;
    children.push(Object.assign(button(`Start in ${folder}`, () => start(false), "primary"), { title: lastLocation }));
  }
  children.push(button("Choose location & start…", () => start(true), lastLocation ? "" : "primary"));

  const others = current ? projects.filter((p) => p.path !== current.path) : projects;
  if (others.length) {
    children.push(el("h2", { textContent: "Recent" }));
    children.push(el("ul", {}, ...others.map((p) => el("li", { title: p.path },
      el("span", { className: "name", textContent: p.name }),
      el("span", { className: "count", textContent: `${p.stills} still${p.stills === 1 ? "" : "s"}` }),
      button("Reopen", () => openProject(p.path)),
    ))));
  }
  view.replaceChildren(...children);
}

function renderStarting(reply) {
  view.replaceChildren(
    el("p", { textContent: `Starting footsight${reply?.project ? ` on “${reply.project.name}”` : ""}…` }),
    el("p", { className: "muted", textContent: "Loading the models takes a few seconds. The editor opens when it's ready." }),
  );
  pollTimer = setTimeout(refresh, 1000);
}

function renderRunning(reply) {
  const counts = reply.stills || {};
  const parts = [["ready", "ready"], ["processing", "processing"], ["queued", "queued"], ["failed", "failed"]]
    .filter(([key]) => counts[key])
    .map(([key, label]) => el("span", {}, el("b", { textContent: String(counts[key]) }), ` ${label}`));
  view.replaceChildren(
    el("h2", { textContent: "Project" }),
    el("p", { textContent: reply.project.name, title: reply.project.path }),
    el("div", { className: "counts" }, ...(parts.length ? parts : [el("span", { textContent: "No stills yet — press ⌘⇧S on a video" })])),
    button("Open editor", async () => { await chrome.runtime.sendMessage({ type: "open-editor" }); window.close(); }, "primary"),
    button("Capture now", async () => { await chrome.runtime.sendMessage({ type: "capture-active" }); window.close(); }),
    button("Switch project…", () => { clearTimeout(pollTimer); renderOff(reply.project); }),
    button("Stop footsight", async () => { await host({ cmd: "stop" }); showMessage(""); refresh(); }, "danger"),
  );
  pollTimer = setTimeout(refresh, 3000); // keep the counts current
}

async function startProject(request) {
  showMessage("");
  if (request.choose) {
    // the Finder dialog takes focus and closes this popup; background.js carries on
    view.replaceChildren(el("p", { textContent: "Choose a folder in the Finder window…" }));
  }
  const reply = await chrome.runtime.sendMessage({ type: "start-project", ...request });
  if (reply?.cancelled) return refresh();
  if (!reply?.ok) {
    showMessage(reply?.error || "Couldn't start the project");
    return renderOff();
  }
  renderStarting({ project: { name: reply.path.split("/").pop() } });
}

async function openProject(path) {
  showMessage("");
  const reply = await chrome.runtime.sendMessage({ type: "open-project", path });
  if (!reply?.ok) {
    showMessage(reply?.error || "Couldn't open the project");
    return renderOff();
  }
  renderStarting({ project: { name: path.split("/").pop() } });
}

refresh();
