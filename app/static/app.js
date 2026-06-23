const form = document.getElementById("generate-form");
const statusCard = document.getElementById("status-card");
const statusText = document.getElementById("status-text");
const generateButton = document.getElementById("generate-button");
const uploadReferenceButton = document.getElementById("upload-reference-button");
const historyList = document.getElementById("history-list");
const historyEmpty = document.getElementById("history-empty");
const referenceVoiceLibrary = document.getElementById(
  "reference-voice-library"
);
const template = document.getElementById("history-item-template");
const pendingRuns = new Map();
let lastHistoryItems = [];
let referenceVoices = [];
let parameterLimits = {};

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
  let displayValue = "n/a";
  if (value !== null && value !== undefined) {
    if (typeof value === "number" && name.endsWith("_ms")) {
      displayValue = formatMs(value);
    } else {
      displayValue = String(value);
    }
  }
  row.innerHTML = `<span>${name}</span><strong>${displayValue}</strong>`;
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
  if (item.reference_voice_label) {
    const referencePill = document.createElement("span");
    referencePill.className = "pill reference-pill";
    referencePill.textContent = `reference: ${item.reference_voice_label}`;
    pills.appendChild(referencePill);
  }

  [
    `device: ${item.parameters.device}`,
    `cfg: ${item.parameters.cfg_value}`,
    `steps: ${item.parameters.inference_timesteps}`,
    `normalize: ${item.parameters.normalize}`,
    `denoise: ${item.parameters.denoise}`,
    `retry: ${item.parameters.retry_badcase}`,
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
    ["retry_badcase_max_times", item.parameters.retry_badcase_max_times],
    [
      "retry_ratio_threshold",
      item.parameters.retry_badcase_ratio_threshold,
    ],
    ["min_len", item.parameters.min_len],
    ["max_len", item.parameters.max_len],
  ].forEach(([name, value]) => timingGrid.appendChild(createTimingRow(name, value)));

  const liveStatus = fragment.querySelector(".live-status");
  if (item.success === false) {
    liveStatus.innerHTML = `
      <span class="status-badge error">failed</span>
      <span class="live-elapsed">${item.error || "Generation failed."}</span>
    `;
  } else {
    liveStatus.innerHTML = `<span class="status-badge success">success</span>`;
  }

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
  if (run.payload.reference_voice_label) {
    const referencePill = document.createElement("span");
    referencePill.className = "pill reference-pill";
    referencePill.textContent = `reference: ${run.payload.reference_voice_label}`;
    pills.appendChild(referencePill);
  }

  [
    `device: ${run.payload.device}`,
    `cfg: ${run.payload.cfg_value}`,
    `steps: ${run.payload.inference_timesteps}`,
    `normalize: ${run.payload.normalize}`,
    `denoise: ${run.payload.denoise}`,
    `retry: ${run.payload.retry_badcase}`,
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
  if (run.error) {
    const errorRow = document.createElement("div");
    errorRow.className = "timing-row error-row";
    errorRow.innerHTML = `<span>error</span><strong>${run.error}</strong>`;
    timingGrid.appendChild(errorRow);
  }

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

function renderReferenceVoices(items) {
  referenceVoices = items;
  const select = document.getElementById("reference_voice_id");
  const currentValue = select.value;
  select.innerHTML = '<option value="">不使用 reference</option>';

  for (const item of items) {
    const option = document.createElement("option");
    option.value = item.voice_id;
    option.textContent = `${item.label} (${(item.file_size_bytes / 1024 / 1024).toFixed(1)} MB)`;
    select.appendChild(option);
  }
  select.value = items.some((item) => item.voice_id === currentValue)
    ? currentValue
    : "";

  if (!items.length) {
    referenceVoiceLibrary.className = "reference-list empty-state";
    referenceVoiceLibrary.textContent = "尚未上傳 reference voice。";
    return;
  }

  referenceVoiceLibrary.className = "reference-list";
  referenceVoiceLibrary.innerHTML = "";
  for (const item of items) {
    const card = document.createElement("article");
    card.className = "reference-card";
    card.innerHTML = `
      <div class="reference-card-header">
        <strong>${item.label}</strong>
        <span>${(item.file_size_bytes / 1024 / 1024).toFixed(1)} MB</span>
      </div>
      <p class="reference-meta">${item.created_at}</p>
      <audio controls preload="metadata" src="${item.audio_url}"></audio>
    `;
    referenceVoiceLibrary.appendChild(card);
  }
}

function applyNumberConstraints(fieldId, spec) {
  const input = document.getElementById(fieldId);
  if (!input || !spec) return;
  input.min = spec.min;
  input.max = spec.max;
  input.step = spec.step;
  input.value = spec.default;
}

function validatePayload(payload) {
  if (!payload.text) {
    return "Text is required.";
  }

  if (payload.max_len < payload.min_len) {
    return "Max Len must be greater than or equal to Min Len.";
  }

  for (const [name, spec] of Object.entries(parameterLimits)) {
    if (!(name in payload)) continue;
    const value = payload[name];
    if (typeof value !== "number" || Number.isNaN(value)) {
      return `${name} must be a valid number.`;
    }
    if (value < spec.min || value > spec.max) {
      return `${name} must stay between ${spec.min} and ${spec.max}.`;
    }
  }

  return null;
}

async function loadDefaults() {
  const response = await fetch("/api/defaults");
  const defaults = await response.json();
  parameterLimits = defaults.parameter_limits || {};
  document.getElementById("text").value = defaults.text;
  document.getElementById("voice_gender").value = defaults.voice_gender;
  document.getElementById("voice_age").value = defaults.voice_age;
  document.getElementById("voice_tone").value = defaults.voice_tone;
  document.getElementById("voice_pace").value = defaults.voice_pace;
  document.getElementById("voice_extra").value = defaults.voice_extra;
  document.getElementById("device").value = defaults.device;
  applyNumberConstraints("cfg_value", parameterLimits.cfg_value);
  document.getElementById("cfg_value").value = defaults.cfg_value;
  applyNumberConstraints(
    "inference_timesteps",
    parameterLimits.inference_timesteps
  );
  document.getElementById("inference_timesteps").value = defaults.inference_timesteps;
  document.getElementById("normalize").checked = defaults.normalize;
  document.getElementById("denoise").checked = defaults.denoise;
  document.getElementById("retry_badcase").checked = defaults.retry_badcase;
  applyNumberConstraints(
    "retry_badcase_max_times",
    parameterLimits.retry_badcase_max_times
  );
  document.getElementById("retry_badcase_max_times").value = defaults.retry_badcase_max_times;
  applyNumberConstraints(
    "retry_badcase_ratio_threshold",
    parameterLimits.retry_badcase_ratio_threshold
  );
  document.getElementById("retry_badcase_ratio_threshold").value = defaults.retry_badcase_ratio_threshold;
  applyNumberConstraints("min_len", parameterLimits.min_len);
  document.getElementById("min_len").value = defaults.min_len;
  applyNumberConstraints("max_len", parameterLimits.max_len);
  document.getElementById("max_len").value = defaults.max_len;
  document.getElementById("optimize").checked = defaults.optimize;
  renderReferenceVoices(defaults.reference_voices || []);
}

async function loadHistory() {
  const response = await fetch("/api/history");
  const payload = await response.json();
  renderHistory(payload.items);
}

async function uploadReferenceVoice() {
  const fileInput = document.getElementById("reference_voice_file");
  const labelInput = document.getElementById("reference_voice_label");
  const selectedFile = fileInput.files?.[0];
  if (!selectedFile) {
    setStatus("error", "Please choose a WAV file before uploading.");
    return;
  }

  uploadReferenceButton.disabled = true;
  const formData = new FormData();
  formData.append("file", selectedFile);
  formData.append("label", labelInput.value.trim());

  try {
    const response = await fetch("/api/reference-voices", {
      method: "POST",
      body: formData,
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Reference upload failed.");
    }
    renderReferenceVoices(payload.items || []);
    document.getElementById("reference_voice_id").value = payload.item.voice_id;
    fileInput.value = "";
    labelInput.value = "";
    setStatus("success", `Uploaded reference voice: ${payload.item.label}`);
  } catch (error) {
    setStatus("error", error.message);
  } finally {
    uploadReferenceButton.disabled = false;
  }
}

uploadReferenceButton.addEventListener("click", () => {
  uploadReferenceVoice().catch((error) => setStatus("error", error.message));
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const payload = {
    text: document.getElementById("text").value.trim(),
    voice_gender: document.getElementById("voice_gender").value,
    voice_age: document.getElementById("voice_age").value,
    voice_tone: document.getElementById("voice_tone").value,
    voice_pace: document.getElementById("voice_pace").value,
    voice_extra: document.getElementById("voice_extra").value.trim(),
    reference_voice_id: document.getElementById("reference_voice_id").value,
    device: document.getElementById("device").value,
    cfg_value: Number(document.getElementById("cfg_value").value),
    inference_timesteps: Number(
      document.getElementById("inference_timesteps").value
    ),
    normalize: document.getElementById("normalize").checked,
    denoise: document.getElementById("denoise").checked,
    retry_badcase: document.getElementById("retry_badcase").checked,
    retry_badcase_max_times: Number(
      document.getElementById("retry_badcase_max_times").value
    ),
    retry_badcase_ratio_threshold: Number(
      document.getElementById("retry_badcase_ratio_threshold").value
    ),
    min_len: Number(document.getElementById("min_len").value),
    max_len: Number(document.getElementById("max_len").value),
    optimize: document.getElementById("optimize").checked,
  };
  payload.voice_description = buildVoiceDescription(payload);
  payload.reference_voice_label =
    referenceVoices.find((item) => item.voice_id === payload.reference_voice_id)
      ?.label || "";

  const validationError = validatePayload(payload);
  if (validationError) {
    setStatus("error", validationError);
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
    const run = pendingRuns.get(pendingId);
    if (run) {
      run.error = error.message;
    }
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
