const form = document.getElementById("generate-form");
const statusCard = document.getElementById("status-card");
const statusText = document.getElementById("status-text");
const generateButton = document.getElementById("generate-button");
const historyList = document.getElementById("history-list");
const historyEmpty = document.getElementById("history-empty");
const template = document.getElementById("history-item-template");
const pendingRuns = new Map();
let lastHistoryItems = [];

function setStatus(kind, message) {
  statusCard.className = `status-card ${kind}`;
  statusText.textContent = message;
}

function formatMs(value) {
  if (typeof value !== "number") return "n/a";
  return `${(value / 1000).toFixed(2)} s`;
}

function createTimingRow(name, value) {
  const row = document.createElement("div");
  row.className = "timing-row";
  row.innerHTML = `<span>${name}</span><strong>${formatMs(value)}</strong>`;
  return row;
}

function buildHistoryNode(item) {
  const fragment = template.content.cloneNode(true);
  fragment.querySelector(".history-title").textContent = item.request_id;
  fragment.querySelector(".history-meta").textContent =
    `${item.request_received_at} · cold start: ${item.cold_start ? "yes" : "no"}`;
  fragment.querySelector(".history-text").textContent = item.parameters.text;

  const pills = fragment.querySelector(".pill-row");
  if (item.voice_description) {
    const voicePill = document.createElement("span");
    voicePill.className = "pill voice-pill";
    voicePill.textContent = `voice: ${item.voice_description}`;
    pills.appendChild(voicePill);
  }

  [
    `device: ${item.parameters.device}`,
    `cfg: ${item.parameters.cfg_value}`,
    `steps: ${item.parameters.inference_timesteps}`,
    `optimize: ${item.parameters.optimize}`,
    `total: ${formatMs(item.timings_ms.total_ms)}`,
  ].forEach((label) => {
    const pill = document.createElement("span");
    pill.className = "pill";
    pill.textContent = label;
    pills.appendChild(pill);
  });

  const player = fragment.querySelector(".audio-player");
  player.src = item.audio_url;

  const logLink = fragment.querySelector(".log-link");
  logLink.href = item.log_url;

  const timingGrid = fragment.querySelector(".timing-grid");
  [
    ["model_load_ms", item.model_load_ms],
    ["model_ready_ms", item.timings_ms.model_ready_ms],
    ["generate_ms", item.timings_ms.generate_ms],
    ["write_wav_ms", item.timings_ms.write_wav_ms],
    ["total_ms", item.timings_ms.total_ms],
  ].forEach(([name, value]) => timingGrid.appendChild(createTimingRow(name, value)));

  const liveStatus = fragment.querySelector(".live-status");
  liveStatus.innerHTML = `<span class="status-badge success">success</span>`;

  return fragment;
}

function buildPendingNode(run) {
  const fragment = template.content.cloneNode(true);
  fragment.querySelector(".history-title").textContent = run.title;
  fragment.querySelector(".history-meta").textContent =
    `${new Date(run.startedAt).toISOString()} · pending`;
  fragment.querySelector(".history-text").textContent = run.payload.text;

  const pills = fragment.querySelector(".pill-row");
  if (run.payload.voice_description) {
    const voicePill = document.createElement("span");
    voicePill.className = "pill voice-pill";
    voicePill.textContent = `voice: ${run.payload.voice_description}`;
    pills.appendChild(voicePill);
  }

  [
    `device: ${run.payload.device}`,
    `cfg: ${run.payload.cfg_value}`,
    `steps: ${run.payload.inference_timesteps}`,
    `optimize: ${run.payload.optimize}`,
  ].forEach((label) => {
    const pill = document.createElement("span");
    pill.className = "pill";
    pill.textContent = label;
    pills.appendChild(pill);
  });

  const player = fragment.querySelector(".audio-player");
  player.style.display = "none";

  const logLink = fragment.querySelector(".log-link");
  logLink.textContent = "Pending";
  logLink.removeAttribute("href");

  const timingDetails = fragment.querySelector(".timing-details");
  timingDetails.open = true;
  const timingGrid = fragment.querySelector(".timing-grid");
  timingGrid.innerHTML = "";
  timingGrid.appendChild(createTimingRow("elapsed", Date.now() - run.startedAt));

  const statusRow = document.createElement("div");
  statusRow.className = "timing-row";
  statusRow.innerHTML = `<span>status</span><strong>${run.status}</strong>`;
  timingGrid.prepend(statusRow);

  const liveStatus = fragment.querySelector(".live-status");
  liveStatus.innerHTML = `
    <span class="status-badge ${run.status === "failed" ? "error" : "working"}">${run.status}</span>
    <span class="live-elapsed">running for ${((Date.now() - run.startedAt) / 1000).toFixed(1)} s</span>
  `;

  return fragment;
}

function renderHistory(items) {
  lastHistoryItems = items;
  historyList.innerHTML = "";
  historyEmpty.style.display =
    items.length || pendingRuns.size ? "none" : "block";

  for (const pending of Array.from(pendingRuns.values()).sort(
    (a, b) => b.startedAt - a.startedAt
  )) {
    historyList.appendChild(buildPendingNode(pending));
  }

  for (const item of items) {
    historyList.appendChild(buildHistoryNode(item));
  }
}

function startPendingRun(payload) {
  const clientId = `pending-${Date.now()}`;
  pendingRuns.set(clientId, {
    clientId,
    title: clientId,
    payload,
    status: "queued",
    startedAt: Date.now(),
  });
  renderHistory(lastHistoryItems);
  return clientId;
}

function markPendingRun(clientId, nextStatus) {
  const run = pendingRuns.get(clientId);
  if (!run) return;
  run.status = nextStatus;
  renderHistory(lastHistoryItems);
}

function finishPendingRun(clientId) {
  pendingRuns.delete(clientId);
}

function buildVoiceDescription(payload) {
  return [
    payload.voice_gender,
    payload.voice_age,
    payload.voice_tone,
    payload.voice_pace,
    payload.voice_extra,
  ]
    .map((value) => value.trim())
    .filter(Boolean)
    .join(", ");
}

async function loadDefaults() {
  const response = await fetch("/api/defaults");
  const defaults = await response.json();
  document.getElementById("text").value = defaults.text;
  document.getElementById("voice_gender").value = defaults.voice_gender;
  document.getElementById("voice_age").value = defaults.voice_age;
  document.getElementById("voice_tone").value = defaults.voice_tone;
  document.getElementById("voice_pace").value = defaults.voice_pace;
  document.getElementById("voice_extra").value = defaults.voice_extra;
  document.getElementById("device").value = defaults.device;
  document.getElementById("cfg_value").value = defaults.cfg_value;
  document.getElementById("inference_timesteps").value =
    defaults.inference_timesteps;
  document.getElementById("optimize").checked = defaults.optimize;
}

async function loadHistory() {
  const response = await fetch("/api/history");
  const payload = await response.json();
  renderHistory(payload.items);
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const payload = {
    text: document.getElementById("text").value.trim(),
    voice_gender: document.getElementById("voice_gender").value,
    voice_age: document.getElementById("voice_age").value,
    voice_tone: document.getElementById("voice_tone").value,
    voice_pace: document.getElementById("voice_pace").value,
    voice_extra: document.getElementById("voice_extra").value.trim(),
    device: document.getElementById("device").value,
    cfg_value: Number(document.getElementById("cfg_value").value),
    inference_timesteps: Number(
      document.getElementById("inference_timesteps").value
    ),
    optimize: document.getElementById("optimize").checked,
  };
  payload.voice_description = buildVoiceDescription(payload);

  if (!payload.text) {
    setStatus("error", "Text is required.");
    return;
  }

  generateButton.disabled = true;
  const pendingId = startPendingRun(payload);
  setStatus(
    "working",
    "Generating audio. The first request on a new model configuration can take several minutes."
  );
  markPendingRun(pendingId, "running");

  try {
    const response = await fetch("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      const errorPayload = await response.json();
      throw new Error(
        errorPayload?.detail?.error || "Generation failed. Check the server logs."
      );
    }

    const result = await response.json();
    finishPendingRun(pendingId);
    setStatus(
      "success",
      `Generated ${result.request_id} in ${formatMs(result.log.timings_ms.total_ms)}.`
    );
    await loadHistory();
  } catch (error) {
    markPendingRun(pendingId, "failed");
    setStatus("error", error.message);
  } finally {
    generateButton.disabled = false;
  }
});

async function bootstrap() {
  await loadDefaults();
  await loadHistory();
  window.setInterval(() => {
    if (pendingRuns.size > 0) {
      renderHistory(lastHistoryItems);
    }
  }, 1000);
}

bootstrap().catch((error) => {
  setStatus("error", error.message);
});
