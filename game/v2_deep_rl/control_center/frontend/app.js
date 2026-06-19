// ── Auth helpers ─────────────────────────────────────────────────────────────
const AUTH_TOKEN_KEY = "cc_auth_token";

function getToken() {
  return localStorage.getItem(AUTH_TOKEN_KEY);
}
function setToken(token) {
  localStorage.setItem(AUTH_TOKEN_KEY, token);
}
function clearToken() {
  localStorage.removeItem(AUTH_TOKEN_KEY);
}

function showLoginScreen() {
  const overlay = document.getElementById("loginOverlay");
  const shell = document.getElementById("appShell");
  if (overlay) overlay.style.display = "flex";
  if (shell) shell.style.display = "none";
  setTimeout(() => {
    const u = document.getElementById("loginUsername");
    if (u) u.focus();
  }, 50);
}

function hideLoginScreen() {
  const overlay = document.getElementById("loginOverlay");
  const shell = document.getElementById("appShell");
  if (overlay) overlay.style.display = "none";
  if (shell) shell.style.display = "flex";
}

function logout() {
  clearToken();
  showLoginScreen();
}
// ─────────────────────────────────────────────────────────────────────────────

const state = {
  apiBaseUrl: window.location.protocol.startsWith("http")
    ? window.location.origin
    : "http://127.0.0.1:8000",
  health: null,
  gameConfigs: [],
  trainingConfigs: [],
  runs: [],
  checkpoints: [],
  jobs: [],
  activePage: "rules",
  activeGameConfigId: "",
  activeTrainingConfigId: "",
  activeCheckpointId: "",
  compatibility: null,
  activeGameConfigPayload: null,
  activeTrainingConfigPayload: null,
  visualGameConfig: null,
  playSession: null,
  directEvaluation: null,
  comparisonEvaluation: null,
  playSeatDrafts: [
    { id: "draft_1", type: "human", display_name: "Player" },
    { id: "draft_2", type: "model-expert", display_name: "AI Expert" },
  ],
  activeProgressJobId: null,
  activeProgressRunId: null,
  trainingProgress: null,
  trainingPreflight: null,
  gameConfigValidation: null,
  trainingConfigValidation: null,
  activeRunId: null,
  runDetail: null,
  activeJobDetailId: null,
  jobDetail: null,
  jobLog: null,
  progressPollHandle: null,
  includeCheckpointSelections: false,
  autopilotSettings: null,
  autopilotHistory: [],
  autopilotStopRequested: false,
  campaigns: [],
  activeCampaignId: null,
  playDiceBox: null,
  playDiceBoxReady: false,
  playDiceBoxInitPromise: null,
  runRating: null,
  messageTimer: null,
};

const DICE_BOX_MODULE_URL = "https://unpkg.com/@3d-dice/dice-box@1.1.3/dist/dice-box.es.min.js";
const DICE_BOX_ASSET_PATH = "https://unpkg.com/@3d-dice/dice-box@1.1.3/assets/";

const DEFAULT_GAME_CONFIG = {
  schema_version: "1.0",
  config_name: "Advanced Classical DDQN",
  config_description: "Default advanced single-player Scrum Game rules for the deep-RL branch.",
  players_count: 1,
  product_names: ["Yellow", "Orange", "Red", "Green", "Blue", "Purple", "Black"],
  max_turns: 6,
  starting_money: 25000,
  ring_value: 5000,
  cost_continue: 0,
  cost_switch_mid: 5000,
  cost_switch_after: 0,
  mandatory_loan_amount: 50000,
  loan_interest: 5000,
  penalty_negative: 1000,
  penalty_positive: 1000,
  daily_scrums_per_sprint: 5,
  daily_scrum_target: 12,
  board_ring_values: [
    [4, 2, 1, 1],
    [5, 3, 2, 1],
    [6, 4, 3, 2],
    [5, 3, 2, 1],
    [4, 3, 2, 1],
    [5, 3, 2, 1],
    [7, 4, 3, 2],
  ],
  board_features: [
    [3, 3, 2, 1],
    [2, 2, 1, 1],
    [1, 1, 1, 1],
    [2, 2, 2, 1],
    [3, 2, 2, 1],
    [2, 2, 1, 1],
    [1, 1, 1, 1],
  ],
  dice_rules: [
    { min_features: 1, max_features: 1, dice_count: 1, dice_sides: 20 },
    { min_features: 2, max_features: 2, dice_count: 2, dice_sides: 10 },
    { min_features: 3, max_features: null, dice_count: 3, dice_sides: 6 },
  ],
  refinement: {
    active: true,
    model_name: "Standard (ID 301)",
    die_sides: 20,
    product_rules: [
      { product_key: "yellow", increase_rolls: [1, 2], decrease_rolls: [19, 20] },
      { product_key: "orange", increase_rolls: [1, 2, 3], decrease_rolls: [19, 20] },
      { product_key: "red", increase_rolls: [1, 2], decrease_rolls: [19, 20] },
      { product_key: "green", increase_rolls: [1, 2, 3], decrease_rolls: [19, 20] },
      { product_key: "blue", increase_rolls: [1, 2, 3, 4], decrease_rolls: [19, 20] },
      { product_key: "purple", increase_rolls: [1, 2, 3], decrease_rolls: [19, 20] },
      { product_key: "black", increase_rolls: [1], decrease_rolls: [20] },
    ],
  },
  incident: {
    active: true,
    allow_player_specific_incidents: false,
    draw_probability: 1.0,
    severity_multiplier: 1.0,
    cards: [
      {
        card_id: 401,
        name: "Demand Collapse Red",
        description: "All future red sprints are worth zero.",
        effect_type: "set_future_product_to_zero",
        target_products: ["red"],
        delta_money: 0,
        target_sprint: null,
        set_value_money: null,
        future_only: true,
        weight: 1.0,
      },
      {
        card_id: 402,
        name: "New Competitors",
        description: "Orange and blue products lose value due to new competitors.",
        effect_type: "adjust_future_products",
        target_products: ["orange", "blue"],
        delta_money: -5000,
        target_sprint: null,
        set_value_money: null,
        future_only: true,
        weight: 1.0,
      },
      {
        card_id: 403,
        name: "Government Subsidy",
        description: "All first sprints gain a subsidy bonus.",
        effect_type: "adjust_specific_sprint_globally",
        target_products: [],
        delta_money: 5000,
        target_sprint: 1,
        set_value_money: null,
        future_only: true,
        weight: 1.0,
      },
      {
        card_id: 404,
        name: "Yellow Demand Boost",
        description: "All future yellow sprints gain value.",
        effect_type: "adjust_future_products",
        target_products: ["yellow"],
        delta_money: 5000,
        target_sprint: null,
        set_value_money: null,
        future_only: true,
        weight: 1.0,
      },
      {
        card_id: 405,
        name: "Black Product Breakthrough",
        description: "The fourth black sprint becomes worth 100000.",
        effect_type: "set_specific_sprint_exact",
        target_products: ["black"],
        delta_money: 0,
        target_sprint: 4,
        set_value_money: 100000,
        future_only: true,
        weight: 1.0,
      },
    ],
  },
  reserved_fields: {},
};

const pages = {
  rules: {
    title: "Design",
    subtitle: "Define the blueprint the agent learns against.",
    usage: "Design uses the active blueprint and training profile.",
  },
  training: {
    title: "Train",
    subtitle: "Launch and monitor training jobs for the active workspace.",
    usage: "Train uses the active blueprint, training profile, and active brain for resume or fine-tune.",
  },
  inspect: {
    title: "Inspect",
    subtitle: "Inspect a running or finished job, run, or brain in one place.",
    usage: "Inspect follows the selected job or run and shows details, logs, and learning progress.",
  },
  evaluate: {
    title: "Evaluate",
    subtitle: "Inspect brains, compare performance, and validate blueprint fit.",
    usage: "Evaluate uses the active blueprint and active brain, while the run browser exposes historical runs and robustness jobs.",
  },
  play: {
    title: "Play",
    subtitle: "Run the arena with human and brain-controlled seats.",
    usage: "Play uses the active blueprint and active brain for model-controlled seats.",
  },
};

function $(id) {
  return document.getElementById(id);
}

function showMessage(text, type = "success") {
  const box = $("globalMessage");
  if (state.messageTimer) {
    clearTimeout(state.messageTimer);
    state.messageTimer = null;
  }
  box.innerHTML = `
    <span>${escapeHtml(text)}</span>
    <button class="message-close" type="button" aria-label="Dismiss message">x</button>
  `;
  box.className = `message ${type}`;
  box.querySelector(".message-close")?.addEventListener("click", clearMessage);
  if (type !== "error") {
    state.messageTimer = setTimeout(clearMessage, 2600);
  }
}

function clearMessage() {
  const box = $("globalMessage");
  if (state.messageTimer) {
    clearTimeout(state.messageTimer);
    state.messageTimer = null;
  }
  box.innerHTML = "";
  box.className = "message hidden";
}

function clearEvaluationResults() {
  state.directEvaluation = null;
  state.comparisonEvaluation = null;
  renderDirectEvaluation();
  renderCheckpointComparison();
}

function clearCompatibilityResult() {
  state.compatibility = null;
  renderCompatibility();
  renderContextCard();
}

async function apiRequest(path, options = {}, timeoutMs = 20000) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const token = getToken();
    console.log(`${state.apiBaseUrl}${path}`);
    const response = await fetch(`${state.apiBaseUrl}${path}`, {
      signal: controller.signal,
      headers: {
        "Content-Type": "application/json",
        ...(token ? { "Authorization": `Bearer ${token}` } : {}),
        ...(options.headers || {}),
      },
      ...options,
    });

    if (response.status === 401) {
      clearToken();
      showLoginScreen();
      throw new Error("Session expired. Please log in again.");
    }

    if (!response.ok) {
      const text = await response.text();
      throw new Error(text || `Request failed: ${response.status}`);
    }

    return response.json();
  } catch (err) {
    if (err.name === "AbortError") {
      throw new Error(`Request timed out: ${path}`);
    }
    throw err;
  } finally {
    clearTimeout(timer);
  }
}

function formatJson(value) {
  return JSON.stringify(value, null, 2);
}

function parseJsonEditor(id) {
  return JSON.parse($(id).value);
}

function clone(value) {
  return structuredClone(value);
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function normalizeProductKey(value) {
  return String(value || "").toLowerCase().replace(/[^a-z0-9]/g, "");
}

function numberValue(inputId, fallback = 0) {
  const value = Number($(inputId).value);
  return Number.isFinite(value) ? value : fallback;
}

function parseNumberList(value) {
  return String(value || "")
    .split(",")
    .map((item) => Number(item.trim()))
    .filter((item) => Number.isFinite(item));
}

function checkpointUiLabel(checkpoint) {
  if (!checkpoint) return "";
  let source = checkpoint.source_type || "checkpoint";
  if (checkpoint.source_type === "run" && checkpoint.source_run) {
    source = formatRunSourceLabel(checkpoint.source_run);
  } else if (checkpoint.source_type === "current_artifacts") {
    source = "Current Artifacts";
  } else if (checkpoint.source_type === "reference_v1") {
    source = "Reference V1";
  } else if (checkpoint.source_type === "playable_model_v1") {
    source = "Playable Model V1";
  }

  // Show checkpoint type (best/intermediate/final) instead of format string.
  const typeLabel = checkpoint.checkpoint_type || "checkpoint";
  // Show episode when available (populated for best checkpoints from metadata).
  const episodeLabel = checkpoint.episode != null ? ` ep${Number(checkpoint.episode).toLocaleString()}` : "";
  // Mark legacy checkpoints so users know they may have limited compatibility info.
  const legacySuffix = checkpoint.checkpoint_format === "legacy" ? " [legacy]" : "";
  return `${source} | ${typeLabel}${episodeLabel}${legacySuffix}`;
}

function formatRunSourceLabel(runName) {
  const raw = String(runName || "").trim();
  const match = raw.match(/^run_(\d{4}-\d{2}-\d{2})_(\d{4})(?:_(.+))?$/);
  if (!match) return raw || "Run";

  const [, datePart, timePart, remainder] = match;
  const timeLabel = `${timePart.slice(0, 2)}:${timePart.slice(2)}`;
  const suffixParts = String(remainder || "")
    .split("_")
    .filter(Boolean);

  let iteration = "";
  if (suffixParts.length && /^\d{2}$/.test(suffixParts[suffixParts.length - 1])) {
    iteration = ` (${Number(suffixParts.pop()) + 1})`;
  }

  const customName = suffixParts.join(" ").trim();
  return customName
    ? `${datePart} ${timeLabel}${iteration} | ${customName}`
    : `${datePart} ${timeLabel}${iteration}`;
}

function checkpointCompatibilityTone(status) {
  const value = String(status || "").toLowerCase();
  if (value.includes("mismatch") || value.includes("incompatible")) return "bad";
  if (value.includes("compatible")) return "good";
  if (value.includes("unknown")) return "warn";
  return "";
}

function checkpointCategory(checkpoint) {
  const type = String(checkpoint?.checkpoint_type || "").toLowerCase();
  const label = String(checkpoint?.label || "").toLowerCase();
  if (type.includes("best") || label.includes("best")) return "best";
  if (type.includes("final") || label.includes("final")) return "final";
  if (type.includes("episode") || label.includes("checkpoint_episode")) return "intermediate";
  return "other";
}

function checkpointGroupLabel(checkpoint) {
  if (checkpoint.source_type === "run") {
    return checkpoint.source_run || "run";
  }
  if (checkpoint.source_type === "current_artifacts") {
    return "Current Artifacts";
  }
  if (checkpoint.source_type === "reference_v1") {
    return "Reference V1";
  }
  if (checkpoint.source_type === "playable_model_v1") {
    return "PlayableModelV1";
  }
  return checkpoint.source_type || "Other";
}

function isModelSelectable(checkpoint) {
  return checkpointCategory(checkpoint) !== "intermediate";
}

function sidebarCheckpointOptions() {
  const includeAll = Boolean(state.includeCheckpointSelections);
  return state.checkpoints
    .filter((item) => includeAll || isModelSelectable(item))
    .map((item) => ({ ...item, ui_label: checkpointUiLabel(item) }))
    .sort((left, right) => left.ui_label.localeCompare(right.ui_label));
}

function formatNumber(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "-";
  }
  return Number(value).toFixed(digits);
}

function selectedGameConfig() {
  return state.gameConfigs.find((item) => item.id === state.activeGameConfigId) || null;
}

function selectedTrainingConfig() {
  return state.trainingConfigs.find((item) => item.id === state.activeTrainingConfigId) || null;
}

function selectedCheckpoint() {
  return state.checkpoints.find((item) => item.id === state.activeCheckpointId) || null;
}

function checkpointByPath(pathValue) {
  return state.checkpoints.find((item) => item.path === pathValue) || null;
}

function selectedProgressJob() {
  return state.jobs.find((item) => item.id === state.activeProgressJobId) || null;
}

function selectedProgressRun() {
  return state.runs.find((item) => item.id === state.activeProgressRunId) || null;
}

function jobForRunId(runId) {
  return state.jobs.find((item) => runLabelFromPath(item.run_dir) === runId) || null;
}

function currentTrainingMode() {
  return $("trainModeSelect")?.value || "train";
}

function runLabelFromPath(pathValue) {
  if (!pathValue) return "-";
  const normalized = String(pathValue).replaceAll("\\", "/").split("/");
  return normalized[normalized.length - 1] || pathValue;
}

function downloadJsonFile(fileName, payload) {
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = fileName;
  anchor.click();
  URL.revokeObjectURL(url);
}

function downloadCsvFile(fileName, headers, rows) {
  const escape = (value) => {
    const text = String(value ?? "");
    return /[",\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
  };
  const csv = [headers.map(escape).join(",")]
    .concat(rows.map((row) => row.map(escape).join(",")))
    .join("\n");
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = fileName;
  anchor.click();
  URL.revokeObjectURL(url);
}

function downloadCheckpointFile(checkpoint) {
  if (!checkpoint?.id) {
    throw new Error("Select a brain before downloading.");
  }
  const anchor = document.createElement("a");
  anchor.href = `${state.apiBaseUrl}/checkpoints/${encodeURIComponent(checkpoint.id)}/download`;
  anchor.download = checkpoint.label || "checkpoint.pth";
  anchor.click();
}

// Guard flag: prevents a new poll from starting before the previous one finishes.
// This replaces setInterval (which fires regardless of async completion) and
// prevents request accumulation when the server is slow during training.
let _pollInFlight = false;

async function _runPollCycle() {
  if (_pollInFlight) return;
  _pollInFlight = true;
  try {
    // Refresh the jobs list first. Subsequent fetches depend on the updated job state.
    const jobsPayload = await apiRequest("/jobs").catch(() => null);
    if (jobsPayload) {
      state.jobs = jobsPayload.items || [];
      renderJobs();
      updateSummaryPills();
    }
    await refreshCampaigns().catch(() => {});

    // Auto-advance to a new running training job when the tracked job is completed.
    // Skip if the user is actively inspecting a specific job. Replacing activeProgressJobId
    // would swap out that job's training charts for the running job's, causing a visible jump.
    const trackedJob = state.jobs.find((j) => j.id === state.activeProgressJobId);
    const userInspectingJob = state.activePage === "inspect" && state.activeJobDetailId != null;
    if (!userInspectingJob && (!state.activeProgressJobId || trackedJob?.status === "completed")) {
      const runningJob = state.jobs.find(
        (j) => j.status === "running" && ["train", "fine_tune"].includes(j.job_type)
      );
      if (runningJob) {
        state.activeProgressJobId = runningJob.id;
      }
    }

    // Determine autopilot run ID from existing state (stale-ok for this cycle).
    const runId =
      state.activeRunId ||
      (state.trainingProgress?.run_dir ? runLabelFromPath(state.trainingProgress.run_dir) : null) ||
      (state.jobDetail?.run_dir ? runLabelFromPath(state.jobDetail.run_dir) : null);

    // Fire progress + autopilot requests in parallel. They are independent of each other.
    await Promise.all([
      state.activeProgressJobId
        ? fetchTrainingProgress(state.activeProgressJobId, false).catch(() => {})
        : Promise.resolve(),
      runId
        ? fetchAutopilotData(runId).catch(() => {})
        : Promise.resolve(),
    ]);

    if (runId) renderAutopilotTrainingPanel();
  } finally {
    _pollInFlight = false;
  }
}

function startProgressPolling() {
  if (state.progressPollHandle) {
    clearTimeout(state.progressPollHandle);
  }
  const schedule = async () => {
    await _runPollCycle();
    state.progressPollHandle = setTimeout(schedule, 5000);
  };
  state.progressPollHandle = setTimeout(schedule, 5000);
}

function buildPolyline(points, width, height) {
  if (!points.length) return "";
  return points.map((point) => `${point.x.toFixed(1)},${point.y.toFixed(1)}`).join(" ");
}

function renderTable(hostId, columns, rows) {
  const host = $(hostId);
  if (!rows.length) {
    host.className = "empty-state";
    host.textContent = "No rows yet.";
    return;
  }
  host.className = "data-table-wrap";
  const header = columns.map((column) => `<th>${escapeHtml(column.label)}</th>`).join("");
  const body = rows
    .map(
      (row) =>
        `<tr>${columns
          .map((column) => `<td>${escapeHtml(String(row[column.key] ?? "-"))}</td>`)
          .join("")}</tr>`
    )
    .join("");
  host.innerHTML = `<table class="data-table"><thead><tr>${header}</tr></thead><tbody>${body}</tbody></table>`;
}

function renderLineChart(hostId, series, valueKey, lineColor, caption) {
  const host = $(hostId);
  const filtered = (series || []).filter((item) => Number.isFinite(item?.[valueKey]));
  if (!filtered.length) {
    host.className = "empty-state";
    host.textContent = caption || "No data yet.";
    return;
  }

  const width = 640;
  const height = 180;
  const padding = 18;
  const values = filtered.map((item) => Number(item[valueKey]));
  const minValue = Math.min(...values);
  const maxValue = Math.max(...values);
  const range = maxValue - minValue || 1;
  const points = filtered.map((item, index) => {
    const x =
      padding +
      (filtered.length === 1 ? 0 : (index / (filtered.length - 1)) * (width - padding * 2));
    const normalized = (Number(item[valueKey]) - minValue) / range;
    const y = height - padding - normalized * (height - padding * 2);
    return { x, y };
  });
  const polyline = buildPolyline(points, width, height);
  const last = filtered[filtered.length - 1];
  host.className = "";
  host.innerHTML = `
    <svg class="mini-chart" viewBox="0 0 ${width} ${height}" preserveAspectRatio="none" aria-label="${escapeHtml(valueKey)} chart">
      <line x1="${padding}" y1="${height - padding}" x2="${width - padding}" y2="${height - padding}" stroke="rgba(170,177,195,0.25)" stroke-width="1" />
      <line x1="${padding}" y1="${padding}" x2="${padding}" y2="${height - padding}" stroke="rgba(170,177,195,0.25)" stroke-width="1" />
      <polyline fill="none" stroke="${lineColor}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" points="${polyline}" />
    </svg>
    <div class="chart-caption">Latest ${escapeHtml(valueKey)}: ${escapeHtml(formatNumber(last[valueKey]))}${caption ? ` | ${escapeHtml(caption)}` : ""}</div>
  `;
}

function renderBarChart(hostId, rows, valueKey, labelKey, positiveColor, negativeColor, caption) {
  const host = $(hostId);
  const filtered = (rows || []).filter((item) => Number.isFinite(Number(item?.[valueKey])));
  if (!filtered.length) {
    host.className = "empty-state";
    host.textContent = caption || "No data yet.";
    return;
  }

  const width = 640;
  const height = 220;
  const padding = 24;
  const values = filtered.map((item) => Number(item[valueKey]));
  const maxAbs = Math.max(...values.map((value) => Math.abs(value)), 1);
  const zeroY = height / 2;
  const barWidth = Math.max(18, Math.min(60, (width - padding * 2) / filtered.length - 8));
  const gap = ((width - padding * 2) - barWidth * filtered.length) / Math.max(filtered.length - 1, 1);
  const bars = filtered
    .map((item, index) => {
      const value = Number(item[valueKey]);
      const magnitude = Math.abs(value) / maxAbs;
      const barHeight = magnitude * (height / 2 - padding);
      const x = padding + index * (barWidth + gap);
      const y = value >= 0 ? zeroY - barHeight : zeroY;
      const fill = value >= 0 ? positiveColor : negativeColor;
      const seedLabel = String(item[labelKey] ?? index + 1);
      return `
        <rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${barWidth.toFixed(1)}" height="${barHeight.toFixed(1)}" rx="3" fill="${fill}" />
        <text x="${(x + barWidth / 2).toFixed(1)}" y="${height - 8}" text-anchor="middle" font-size="10" fill="#667085">${escapeHtml(seedLabel)}</text>
      `;
    })
    .join("");
  host.className = "";
  host.innerHTML = `
    <svg class="bar-chart" viewBox="0 0 ${width} ${height}" preserveAspectRatio="none" aria-label="${escapeHtml(valueKey)} bar chart">
      <line x1="${padding}" y1="${zeroY}" x2="${width - padding}" y2="${zeroY}" stroke="rgba(102,112,133,0.35)" stroke-width="1" />
      ${bars}
    </svg>
    <div class="chart-caption">${escapeHtml(caption || `Values by ${labelKey}`)}</div>
  `;
}

function setPage(pageId) {
  state.activePage = pageId;
  document.querySelectorAll(".nav-button").forEach((button) => {
    button.classList.toggle("is-active", button.dataset.page === pageId);
  });
  document.querySelectorAll(".page").forEach((page) => {
    page.classList.toggle("is-active", page.id === `page-${pageId}`);
  });
  $("pageTitle").textContent = pages[pageId].title;
  $("pageSubtitle").textContent = pages[pageId].subtitle;
  $("contextPageUsage").textContent = pages[pageId].usage;
  renderContextCard();
}

function updateStatusCard() {
  const card = $("backendStatusCard");
  if (!state.health) {
    card.className = "status-card status-muted";
    card.innerHTML = "<span>Not connected</span>";
    return;
  }
  card.className = "status-card status-ok";
  card.innerHTML = `<span>${state.health.status} | v${state.health.api_version}</span>`;
}

function updateSummaryPills() {
  const activeGameConfig = state.gameConfigs.find((item) => item.id === state.activeGameConfigId);
  const activeCheckpoint = state.checkpoints.find((item) => item.id === state.activeCheckpointId);
  const visibleJobs = state.jobs.filter((job) => ["queued", "completed", "failed", "stopped"].includes(job.status));
  $("summaryRuleSignature").textContent = `Blueprint: ${activeGameConfig?.rule_signature || "-"}`;
  $("summaryCheckpointStatus").textContent = `Brain: ${activeCheckpoint?.checkpoint_format || "-"}`;
  $("summaryJobCount").textContent = `Jobs: ${visibleJobs.length}`;
}

function renderContextCard() {
  const body = $("contextCardBody");
  const gameConfig = selectedGameConfig();
  const trainingConfig = selectedTrainingConfig();
  const checkpoint = selectedCheckpoint();
  const compatibilityText = state.compatibility
    ? `${state.compatibility.strict_resume_status} / ${state.compatibility.fine_tune_status}`
    : "not checked";
  body.innerHTML = `
    <div class="context-item">
      <span>Active Blueprint</span>
      <strong>${escapeHtml(gameConfig?.label || "-")}</strong>
    </div>
    <div class="context-item">
      <span>Active Training Profile</span>
      <strong>${escapeHtml(trainingConfig?.label || "-")}</strong>
    </div>
    <div class="context-item">
      <span>Active Brain</span>
      <strong>${escapeHtml(checkpoint ? checkpointUiLabel(checkpoint) : "-")}</strong>
    </div>
    <div class="context-item">
      <span>Compatibility Status</span>
      <strong>${escapeHtml(compatibilityText)}</strong>
    </div>
  `;
}

function buildOptions(selectId, items, valueKey = "id", labelKey = "label", emptyLabel = "None") {
  const select = $(selectId);
  const currentValue = select.value;
  select.innerHTML = "";
  if (!items.length) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = emptyLabel;
    select.appendChild(option);
    return;
  }

  items.forEach((item) => {
    const option = document.createElement("option");
    option.value = item[valueKey];
    option.textContent = item[labelKey] || item.label || item.id || emptyLabel;
    select.appendChild(option);
  });

  if (items.some((item) => item[valueKey] === currentValue)) {
    select.value = currentValue;
  }
}

function ensureVisualGameConfig() {
  if (!state.visualGameConfig) {
    state.visualGameConfig = clone(DEFAULT_GAME_CONFIG);
  }
}

function rebuildVisualBoard(productCount, sprintCount) {
  const config = state.visualGameConfig;
  const nextRingValues = [];
  const nextFeatures = [];
  for (let productIndex = 0; productIndex < productCount; productIndex += 1) {
    const ringRow = [];
    const featureRow = [];
    for (let sprintIndex = 0; sprintIndex < sprintCount; sprintIndex += 1) {
      ringRow.push(config.board_ring_values?.[productIndex]?.[sprintIndex] ?? 1);
      featureRow.push(config.board_features?.[productIndex]?.[sprintIndex] ?? 1);
    }
    nextRingValues.push(ringRow);
    nextFeatures.push(featureRow);
  }
  config.board_ring_values = nextRingValues;
  config.board_features = nextFeatures;
}

function rebuildVisualProductNames(productCount) {
  const config = state.visualGameConfig;
  const nextNames = [];
  for (let index = 0; index < productCount; index += 1) {
    nextNames.push(config.product_names?.[index] ?? `Product ${index + 1}`);
  }
  config.product_names = nextNames;
}

function rebuildVisualRefinementRules() {
  state.visualGameConfig.refinement.product_rules = state.visualGameConfig.product_names.map((name) => ({
    product_key: normalizeProductKey(name),
    increase_rolls: [1, 2],
    decrease_rolls: [19, 20],
  }));
}

function ensureVisualShapeConsistency() {
  ensureVisualGameConfig();
  const config = state.visualGameConfig;
  const productCount = Math.max(1, config.product_names?.length || config.board_ring_values?.length || 1);
  const sprintCount = Math.max(1, config.board_ring_values?.[0]?.length || config.board_features?.[0]?.length || 1);
  rebuildVisualProductNames(productCount);
  rebuildVisualBoard(productCount, sprintCount);
  if (!Array.isArray(config.refinement?.product_rules) || config.refinement.product_rules.length !== productCount) {
    rebuildVisualRefinementRules();
  }
}

function syncVisualShapeFromInputs() {
  const productCount = Math.max(1, numberValue("productsCountInput", state.visualGameConfig.product_names.length));
  const sprintCount = Math.max(1, numberValue("sprintsPerProductInput", state.visualGameConfig.board_ring_values[0]?.length || 1));
  rebuildVisualProductNames(productCount);
  rebuildVisualBoard(productCount, sprintCount);
  if (!Array.isArray(state.visualGameConfig.refinement.product_rules) || state.visualGameConfig.refinement.product_rules.length !== productCount) {
    rebuildVisualRefinementRules();
  }
}

function readVisualEditorIntoState() {
  ensureVisualGameConfig();
  const config = state.visualGameConfig;
  config.config_name = $("configNameInput").value.trim();
  config.schema_version = $("schemaVersionInput").value.trim();
  config.config_description = $("configDescriptionInput").value.trim();
  config.players_count = Math.max(1, numberValue("playersCountInput", 1));
  config.max_turns = Math.max(1, numberValue("maxTurnsInput", 1));
  config.starting_money = numberValue("startingMoneyInput");
  config.ring_value = numberValue("ringValueInput");
  config.cost_continue = numberValue("costContinueInput");
  config.cost_switch_mid = numberValue("costSwitchMidInput");
  config.cost_switch_after = numberValue("costSwitchAfterInput");
  config.mandatory_loan_amount = numberValue("mandatoryLoanInput");
  config.loan_interest = numberValue("loanInterestInput");
  config.penalty_negative = numberValue("penaltyNegativeInput");
  config.penalty_positive = numberValue("penaltyPositiveInput");
  config.daily_scrums_per_sprint = Math.max(1, numberValue("dailyScrumsInput", 1));
  config.daily_scrum_target = Math.max(1, numberValue("dailyScrumTargetInput", 1));

  syncVisualShapeFromInputs();

  config.product_names = config.product_names.map((_, index) => {
    const input = $(`productNameInput_${index}`);
    return input ? input.value.trim() || `Product ${index + 1}` : `Product ${index + 1}`;
  });

  config.board_ring_values = config.board_ring_values.map((row, productIndex) =>
    row.map((_, sprintIndex) => numberValue(`ringValue_${productIndex}_${sprintIndex}`, 1))
  );
  config.board_features = config.board_features.map((row, productIndex) =>
    row.map((_, sprintIndex) => numberValue(`featureValue_${productIndex}_${sprintIndex}`, 1))
  );

  config.dice_rules = config.dice_rules.map((rule, index) => ({
    min_features: Math.max(1, numberValue(`diceMin_${index}`, rule.min_features)),
    max_features: (() => {
      const raw = $(`diceMax_${index}`).value.trim();
      return raw === "" ? null : Math.max(1, Number(raw));
    })(),
    dice_count: Math.max(1, numberValue(`diceCount_${index}`, rule.dice_count)),
    dice_sides: Math.max(2, numberValue(`diceSides_${index}`, rule.dice_sides)),
  }));

  config.refinement.active = $("refinementActiveInput").checked;
  config.refinement.model_name = $("refinementModelInput").value.trim();
  config.refinement.die_sides = Math.max(2, numberValue("refinementDieSidesInput", 20));
  config.refinement.product_rules = config.refinement.product_rules.map((rule, index) => ({
    product_key: normalizeProductKey($(`refinementKey_${index}`).value) || normalizeProductKey(config.product_names[index]),
    increase_rolls: parseNumberList($(`refinementIncrease_${index}`).value),
    decrease_rolls: parseNumberList($(`refinementDecrease_${index}`).value),
  }));

  config.incident.active = $("incidentActiveInput").checked;
  config.incident.allow_player_specific_incidents = $("playerSpecificIncidentsInput").checked;
  config.incident.draw_probability = Number($("incidentDrawProbabilityInput").value);
  config.incident.severity_multiplier = Number($("incidentSeverityMultiplierInput").value);
  config.incident.cards = config.incident.cards.map((card, index) => ({
    card_id: numberValue(`incidentId_${index}`, card.card_id),
    name: $(`incidentName_${index}`).value.trim(),
    description: $(`incidentDescription_${index}`).value.trim(),
    effect_type: $(`incidentEffect_${index}`).value.trim(),
    target_products: $(`incidentTargets_${index}`).value.split(",").map((value) => normalizeProductKey(value)).filter(Boolean),
    delta_money: numberValue(`incidentDelta_${index}`, 0),
    target_sprint: (() => {
      const raw = $(`incidentSprint_${index}`).value.trim();
      if (raw === "") return null;
      const sprintCount = config.board_ring_values[0]?.length || 1;
      return Math.max(1, Math.min(sprintCount, Number(raw)));
    })(),
    set_value_money: (() => {
      const raw = $(`incidentExactValue_${index}`).value.trim();
      return raw === "" ? null : Number(raw);
    })(),
    future_only: $(`incidentFutureOnly_${index}`).checked,
    weight: Number($(`incidentWeight_${index}`).value),
  }));

  return clone(config);
}

function renderVisualMetadata() {
  const config = state.visualGameConfig;
  $("configNameInput").value = config.config_name;
  $("schemaVersionInput").value = config.schema_version;
  $("configDescriptionInput").value = config.config_description;
  $("playersCountInput").value = config.players_count;
  $("productsCountInput").value = config.product_names.length;
  $("sprintsPerProductInput").value = config.board_ring_values[0]?.length || 1;
  $("maxTurnsInput").value = config.max_turns;
  $("startingMoneyInput").value = config.starting_money;
  $("ringValueInput").value = config.ring_value;
  $("costContinueInput").value = config.cost_continue;
  $("costSwitchMidInput").value = config.cost_switch_mid;
  $("costSwitchAfterInput").value = config.cost_switch_after;
  $("mandatoryLoanInput").value = config.mandatory_loan_amount;
  $("loanInterestInput").value = config.loan_interest;
  $("penaltyNegativeInput").value = config.penalty_negative;
  $("penaltyPositiveInput").value = config.penalty_positive;
  $("dailyScrumsInput").value = config.daily_scrums_per_sprint;
  $("dailyScrumTargetInput").value = config.daily_scrum_target;
}

function renderVisualProductNames() {
  const host = $("productNamesGrid");
  host.innerHTML = "";
  state.visualGameConfig.product_names.forEach((name, index) => {
    const label = document.createElement("label");
    label.className = "field";
    label.innerHTML = `
      <span>Product ${index + 1}</span>
      <input id="productNameInput_${index}" type="text" value="${escapeHtml(name)}" />
    `;
    host.appendChild(label);
  });
}

function renderVisualBoardMatrix() {
  const host = $("boardMatrixContainer");
  const sprintCount = state.visualGameConfig.board_ring_values[0]?.length || 1;
  let html = '<table class="matrix-table"><thead><tr><th>Product</th>';
  for (let sprintIndex = 0; sprintIndex < sprintCount; sprintIndex += 1) {
    html += `<th>Sprint ${sprintIndex + 1}</th>`;
  }
  html += "</tr></thead><tbody>";
  state.visualGameConfig.product_names.forEach((productName, productIndex) => {
    html += `<tr><th>${escapeHtml(productName)}</th>`;
    for (let sprintIndex = 0; sprintIndex < sprintCount; sprintIndex += 1) {
      html += `
        <td>
          <div class="matrix-cell">
            <label class="field">
              <span>Value</span>
              <input id="ringValue_${productIndex}_${sprintIndex}" type="number" value="${state.visualGameConfig.board_ring_values[productIndex][sprintIndex]}" />
            </label>
            <label class="field">
              <span>Features</span>
              <input id="featureValue_${productIndex}_${sprintIndex}" type="number" min="1" value="${state.visualGameConfig.board_features[productIndex][sprintIndex]}" />
            </label>
          </div>
        </td>
      `;
    }
    html += "</tr>";
  });
  html += "</tbody></table>";
  host.innerHTML = html;
}

function renderVisualDiceRules() {
  const host = $("diceRulesList");
  host.innerHTML = "";
  state.visualGameConfig.dice_rules.forEach((rule, index) => {
    const row = document.createElement("div");
    row.className = "list-row";
    row.innerHTML = `
      <div class="list-row-head">
        <strong>Rule ${index + 1}</strong>
        <button class="button secondary" type="button" data-remove-dice="${index}">Remove</button>
      </div>
      <div class="list-row-grid">
        <label class="field"><span>Min Features</span><input id="diceMin_${index}" type="number" min="1" value="${rule.min_features}" /></label>
        <label class="field"><span>Max Features</span><input id="diceMax_${index}" type="number" min="1" value="${rule.max_features ?? ""}" placeholder="blank = no max" /></label>
        <label class="field"><span>Dice Count</span><input id="diceCount_${index}" type="number" min="1" value="${rule.dice_count}" /></label>
        <label class="field"><span>Dice Sides</span><input id="diceSides_${index}" type="number" min="2" value="${rule.dice_sides}" /></label>
      </div>
    `;
    host.appendChild(row);
  });
}

function renderVisualRefinementRules() {
  const config = state.visualGameConfig;
  $("refinementActiveInput").checked = Boolean(config.refinement.active);
  $("refinementModelInput").value = config.refinement.model_name;
  $("refinementDieSidesInput").value = config.refinement.die_sides;
  const host = $("refinementRulesList");
  host.innerHTML = "";
  config.refinement.product_rules.forEach((rule, index) => {
    const row = document.createElement("div");
    row.className = "list-row";
    row.innerHTML = `
      <div class="list-row-head">
        <strong>Product Rule ${index + 1}</strong>
      </div>
      <div class="list-row-grid">
        <label class="field"><span>Product Key</span><input id="refinementKey_${index}" type="text" value="${escapeHtml(rule.product_key)}" /></label>
        <label class="field"><span>Increase Rolls</span><input id="refinementIncrease_${index}" type="text" value="${rule.increase_rolls.join(", ")}" /></label>
        <label class="field"><span>Decrease Rolls</span><input id="refinementDecrease_${index}" type="text" value="${rule.decrease_rolls.join(", ")}" /></label>
      </div>
    `;
    host.appendChild(row);
  });
}

function renderVisualIncidentCards() {
  const config = state.visualGameConfig;
  $("incidentActiveInput").checked = Boolean(config.incident.active);
  $("playerSpecificIncidentsInput").checked = Boolean(config.incident.allow_player_specific_incidents);
  $("incidentDrawProbabilityInput").value = config.incident.draw_probability;
  $("incidentSeverityMultiplierInput").value = config.incident.severity_multiplier;
  const host = $("incidentCardsList");
  host.innerHTML = "";
  config.incident.cards.forEach((card, index) => {
    const row = document.createElement("div");
    row.className = "list-row";
    row.innerHTML = `
      <div class="list-row-head">
        <strong>Incident Card ${index + 1}</strong>
        <button class="button secondary" type="button" data-remove-incident="${index}">Remove</button>
      </div>
      <div class="grid three">
        <label class="field hidden"><span>Card ID</span><input id="incidentId_${index}" type="number" value="${card.card_id}" /></label>
        <label class="field"><span>Name</span><input id="incidentName_${index}" type="text" value="${escapeHtml(card.name)}" /></label>
        <label class="field"><span>Effect Type</span><input id="incidentEffect_${index}" type="text" value="${escapeHtml(card.effect_type)}" /></label>
        <label class="field span-2"><span>Description</span><textarea id="incidentDescription_${index}" rows="2">${escapeHtml(card.description)}</textarea></label>
        <label class="field"><span>Target Products</span><input id="incidentTargets_${index}" type="text" value="${escapeHtml(card.target_products.join(", "))}" placeholder="comma-separated product keys" /></label>
        <label class="field"><span>Delta Money</span><input id="incidentDelta_${index}" type="number" value="${card.delta_money}" /></label>
        <label class="field"><span>Target Sprint</span><input id="incidentSprint_${index}" type="number" min="1" value="${card.target_sprint ?? ""}" /></label>
        <label class="field"><span>Set Exact Value</span><input id="incidentExactValue_${index}" type="number" value="${card.set_value_money ?? ""}" /></label>
        <label class="field"><span>Weight</span><input id="incidentWeight_${index}" type="number" min="0.1" step="0.1" value="${card.weight}" /></label>
        <label class="field checkbox-inline">
          <input id="incidentFutureOnly_${index}" type="checkbox" ${card.future_only ? "checked" : ""} />
          <span>Future Only</span>
        </label>
      </div>
    `;
    host.appendChild(row);
  });
}

function syncGameJsonEditorFromVisual() {
  try {
    const canonical = readVisualEditorIntoState();
    $("gameConfigEditor").value = formatJson(canonical);
    $("summaryProducts").textContent = String(canonical.product_names.length);
    $("summarySprints").textContent = String(canonical.board_ring_values[0]?.length || 0);
    $("summaryActions").textContent = String(canonical.product_names.length + 1);
    $("summaryIncidentCards").textContent = String(canonical.incident.cards.length);
    if (!$("gameConfigFileNameInput").value.trim()) {
      $("gameConfigFileNameInput").value = String(canonical.config_name || "game_config")
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, "_")
        .replace(/^_+|_+$/g, "");
    }
  } catch (_error) {
    // Keep the current editor text untouched until the input becomes valid again.
  }
}

function renderVisualEditor() {
  ensureVisualShapeConsistency();
  renderVisualMetadata();
  renderVisualProductNames();
  renderVisualBoardMatrix();
  renderVisualDiceRules();
  renderVisualRefinementRules();
  renderVisualIncidentCards();
  syncGameJsonEditorFromVisual();
}

function renderGameConfigs() {
  $("gameConfigCount").textContent = `${state.gameConfigs.length}`;
  const container = $("gameConfigsList");
  container.innerHTML = "";
  state.gameConfigs.forEach((config) => {
    const card = document.createElement("article");
    card.className = "list-card";
    card.innerHTML = `
      <h4>${escapeHtml(config.config_name || config.label)}</h4>
      <div class="card-meta">
        <span class="tag">${escapeHtml(config.source)}</span>
        <span class="tag">${escapeHtml(String(config.products_count))} products</span>
        <span class="tag">${escapeHtml(String(config.sprints_per_product))} sprints</span>
      </div>
    `;
    container.appendChild(card);
  });
}

function renderTrainingConfigs() {
  $("trainingConfigCount").textContent = `${state.trainingConfigs.length}`;
  const container = $("trainingConfigsList");
  container.innerHTML = "";
  state.trainingConfigs.forEach((config) => {
    const card = document.createElement("article");
    card.className = "list-card";
    card.innerHTML = `
      <h4>${escapeHtml(config.label)}</h4>
      <div class="card-meta">
        <span class="tag">${escapeHtml(config.source)} profile</span>
        <span class="tag">${escapeHtml(String(config.episodes))} episodes</span>
        <span class="tag">lr ${escapeHtml(String(config.learning_rate))}</span>
        <span class="tag">gamma ${escapeHtml(String(config.gamma))}</span>
      </div>
    `;
    container.appendChild(card);
  });
}

function renderGameConfigValidation() {
  const container = $("gameConfigValidationCard");
  if (!state.gameConfigValidation) {
    container.className = "empty-state";
    container.textContent = "Validate the current game config to see structural errors.";
    return;
  }
  if (!state.gameConfigValidation.valid) {
    container.className = "list-card";
    container.innerHTML = `<h4>Validation Error</h4><p>${escapeHtml(state.gameConfigValidation.error || "Unknown validation error.")}</p>`;
    return;
  }
  container.className = "list-card";
  container.innerHTML = `
    <h4>Validation OK</h4>
    <div class="card-meta">
      <span class="tag good">valid</span>
      <span class="tag">${escapeHtml(String(state.gameConfigValidation.products_count))} products</span>
      <span class="tag">${escapeHtml(String(state.gameConfigValidation.sprints_per_product))} sprints</span>
      <span class="tag">${escapeHtml(String(state.gameConfigValidation.actions_count))} actions</span>
    </div>
  `;
}

function renderTrainingConfigValidation() {
  const container = $("trainingConfigValidationCard");
  if (!state.trainingConfigValidation) {
    container.className = "empty-state";
    container.textContent = "Validate the current training config to see structural errors.";
    return;
  }
  if (!state.trainingConfigValidation.valid) {
    container.className = "list-card";
    container.innerHTML = `<h4>Validation Error</h4><p>${escapeHtml(state.trainingConfigValidation.error || "Unknown validation error.")}</p>`;
    return;
  }
  container.className = "list-card";
  container.innerHTML = `
    <h4>Validation OK</h4>
    <div class="card-meta">
      <span class="tag good">valid</span>
      <span class="tag">${escapeHtml(String(state.trainingConfigValidation.episodes))} episodes</span>
      <span class="tag">lr ${escapeHtml(String(state.trainingConfigValidation.learning_rate))}</span>
      <span class="tag">gamma ${escapeHtml(String(state.trainingConfigValidation.gamma))}</span>
      <span class="tag">batch ${escapeHtml(String(state.trainingConfigValidation.batch_size))}</span>
    </div>
  `;
}

function renderRuns() {
  $("runCount").textContent = `${state.runs.length}`;
  buildOptions("robustnessRunSelect", state.runs);
  const container = $("runsList");
  container.innerHTML = "";
  state.runs.forEach((run) => {
    const card = document.createElement("article");
    card.className = "list-card";
    card.innerHTML = `
      <h4>${run.label}</h4>
      <p>${run.run_notes || "No notes"}</p>
      <div class="card-meta">
        <span class="tag">${run.created_at || "unknown date"}</span>
        <span class="tag">${run.average_reward_per_episode ?? "-" } avg reward</span>
        <span class="tag">${run.bankruptcy_rate ?? "-" } bankruptcy</span>
      </div>
      <div class="inline-actions">
        <button class="button secondary view-run-button" data-run-id="${run.id}" type="button">View Run</button>
        <button class="button secondary open-inspect-run-button" data-run-id="${run.id}" type="button">Open Inspect</button>
        ${run.best_checkpoint_path ? `<button class="button secondary use-run-best-button" data-run-id="${run.id}" type="button">Use Best Brain</button>` : ""}
      </div>
    `;
    container.appendChild(card);
  });

  container.querySelectorAll(".view-run-button").forEach((button) => {
    button.addEventListener("click", async () => {
      try {
        await fetchRunDetail(button.dataset.runId, true);
      } catch (error) {
        showMessage(error.message, "error");
      }
    });
  });

  container.querySelectorAll(".open-inspect-run-button").forEach((button) => {
    button.addEventListener("click", async () => {
      try {
        await openInspectForRun(button.dataset.runId, true);
      } catch (error) {
        showMessage(error.message, "error");
      }
    });
  });

  container.querySelectorAll(".use-run-best-button").forEach((button) => {
    button.addEventListener("click", () => {
      const run = state.runs.find((item) => item.id === button.dataset.runId);
      const checkpoint = checkpointByPath(run?.best_checkpoint_path);
      if (!checkpoint) {
        showMessage("Best checkpoint for this run is not available in the current catalog.", "error");
        return;
      }
      state.activeCheckpointId = checkpoint.id;
      $("activeCheckpointSelect").value = checkpoint.id;
      updateSummaryPills();
      clearCompatibilityResult();
      clearEvaluationResults();
      renderContextCard();
      renderTrainingSelectionSummary();
      renderCheckpointDetail();
      showMessage(`Active brain set to the best checkpoint from ${run.label}.`);
    });
  });
}

function renderJobs() {
  const container = $("jobsList");
  container.innerHTML = "";
  const visibleJobs = state.jobs.filter((job) => ["queued", "running", "completed", "failed", "stopped"].includes(job.status));

  if (
    state.activeProgressJobId &&
    !state.jobs.some((job) => job.id === state.activeProgressJobId)
  ) {
    state.activeProgressJobId = null;
    state.trainingProgress = null;
  }

  if (!visibleJobs.length) {
    container.innerHTML = `<div class="empty-state">No jobs yet.</div>`;
    if (!state.activeProgressJobId) {
      renderTrainingProgress();
    }
    return;
  }

  if (!state.activeProgressJobId) {
    // Don't auto-select a job to track when the user is actively inspecting a specific
    // one. It would silently replace the inspected run's progress with the running job's.
    const userInspectingJob = state.activePage === "inspect" && state.activeJobDetailId != null;
    if (!userInspectingJob) {
      const preferredJob = state.jobs.find((job) => ["running", "queued"].includes(job.status) && ["train", "fine_tune"].includes(job.job_type))
        || state.jobs.find((job) => ["train", "fine_tune"].includes(job.job_type));
      if (preferredJob) {
        state.activeProgressJobId = preferredJob.id;
      }
    }
  }

  visibleJobs.forEach((job) => {
    const card = document.createElement("article");
    card.className = "list-card";
    const queuedTrainingJobs = visibleJobs
      .filter((item) => ["train", "fine_tune"].includes(item.job_type) && item.status === "queued")
      .slice()
      .reverse();
    const queueIndex = queuedTrainingJobs.findIndex((item) => item.id === job.id);
    const queueTag = queueIndex >= 0 ? `<span class="tag">queue #${queueIndex + 1}</span>` : "";
    const statusTone = job.status === "completed"
      ? "good"
      : job.status === "failed" || job.status === "stopped"
        ? "bad"
        : job.status === "running"
          ? "warn"
          : "";
    const stopButton = ["queued", "running"].includes(job.status)
      ? `<button class="button secondary stop-job-button" data-job-id="${job.id}" type="button">Stop</button>`
      : "";
    const dismissButton = ["completed", "failed", "stopped"].includes(job.status)
      ? `<button class="button secondary dismiss-job-button" data-job-id="${job.id}" type="button">Dismiss</button>`
      : "";
    const inspectButton = `<button class="button secondary open-inspect-job-button" data-job-id="${job.id}" type="button">Open Inspect</button>`;
    card.innerHTML = `
      <h4>Job #${job.id} - ${escapeHtml(job.job_type)}</h4>
      <div class="card-meta">
        <span class="tag ${statusTone}">${job.status}</span>
        ${queueTag}
        <span class="tag">${job.created_at}</span>
        <span class="tag">${runLabelFromPath(job.run_dir)}</span>
      </div>
      <div class="inline-actions">
        ${inspectButton}
        ${stopButton}
        ${dismissButton}
      </div>
    `;
    container.appendChild(card);
  });

  container.querySelectorAll(".stop-job-button").forEach((button) => {
    button.addEventListener("click", async () => {
      try {
        await apiRequest(`/jobs/${button.dataset.jobId}/stop`, { method: "POST" });
        showMessage(`Stopped job ${button.dataset.jobId}.`);
        await refreshJobs();
      } catch (error) {
        showMessage(error.message, "error");
      }
    });
  });

  container.querySelectorAll(".open-inspect-job-button").forEach((button) => {
    button.addEventListener("click", async () => {
      try {
        await openInspectForJob(Number(button.dataset.jobId), true);
      } catch (error) {
        showMessage(error.message, "error");
      }
    });
  });

  container.querySelectorAll(".dismiss-job-button").forEach((button) => {
    button.addEventListener("click", async () => {
      const jobId = Number(button.dataset.jobId);
      try {
        await apiRequest(`/jobs/${jobId}`, { method: "DELETE" });
        if (state.activeJobDetailId === jobId) {
          state.activeJobDetailId = null;
          state.jobDetail = null;
          state.jobLog = null;
          renderJobDetail();
          renderJobLog();
        }
        showMessage(`Dismissed job ${jobId}.`);
        await refreshJobs();
      } catch (error) {
        showMessage(error.message, "error");
      }
    });
  });

  if (state.activeProgressJobId) {
    fetchTrainingProgress(state.activeProgressJobId, false).catch(() => {});
  } else {
    renderTrainingProgress();
  }
}

function renderRunDetail() {
  const label = $("runDetailLabel");
  const container = $("runDetailCard");
  if (!state.runDetail) {
    label.textContent = "No run selected";
    container.className = "empty-state";
    container.textContent = "Select a run to inspect its metadata, brains, and evaluation actions.";
    return;
  }
  label.textContent = state.runDetail.label;
  const metadata = state.runDetail.metadata || {};
  const metrics = state.runDetail.metrics || {};
  const checkpoints = state.runDetail.checkpoints || [];
  const bestCheckpoint = checkpointByPath(metadata.best_checkpoint_path || state.runDetail.metadata?.best_checkpoint_path || state.runs.find((item) => item.id === state.runDetail.id)?.best_checkpoint_path);
  const rating = state.runRating;
  const gradeColors = { S: "#7c3aed", A: "#16a34a", B: "#0284c7", C: "#ca8a04", D: "#ea580c", F: "#dc2626" };
  const ratingHtml = (() => {
    if (!rating) return "";
    if (rating.grade === "N/A") return `<span class="tag">rating unavailable</span>`;
    const color = gradeColors[rating.grade] || "#6b7280";
    const sn = rating.snapshot || {};
    const brPct = sn.bankruptcy_rate != null ? `${(sn.bankruptcy_rate * 100).toFixed(0)}%` : "?";
    const reward = sn.average_reward != null ? Math.round(sn.average_reward).toLocaleString() : "?";
    return `
      <div class="rating-card">
        <div class="rating-grade" style="background:${color};">${escapeHtml(rating.grade)}</div>
        <div>
          <strong>${escapeHtml(String(rating.score))}/100</strong>
          <div class="checkpoint-subtitle">bankruptcies ${escapeHtml(brPct)} | avg reward ${escapeHtml(reward)}</div>
        </div>
      </div>`;
  })();

  container.className = "list-card";
  container.innerHTML = `
    <h4>${escapeHtml(state.runDetail.label)}</h4>
    <div class="card-meta">
      <span class="tag">${escapeHtml(metadata.created_at || "-")}</span>
      <span class="tag">${escapeHtml(metadata.resume_mode || "new")}</span>
    </div>
    <div class="card-meta">
      <span class="tag">avg reward ${escapeHtml(String(metrics.average_reward_per_episode ?? "-"))}</span>
      <span class="tag">bankruptcy ${escapeHtml(String(metrics.bankruptcy_rate ?? "-"))}</span>
      <span class="tag">${checkpoints.length} checkpoints</span>
    </div>
    ${ratingHtml}
    ${metadata.run_notes ? `<p>${escapeHtml(metadata.run_notes)}</p>` : ""}
    <div class="inline-actions">
      ${bestCheckpoint ? `<button class="button secondary use-run-detail-best-button" type="button">Use Best Brain</button>` : ""}
      <button class="button secondary open-run-inspect-button" type="button">Open Inspect</button>
      <button class="button secondary queue-run-robustness-button" type="button">Queue Robustness</button>
    </div>
  `;

  container.querySelector(".use-run-detail-best-button")?.addEventListener("click", () => {
    state.activeCheckpointId = bestCheckpoint.id;
    $("activeCheckpointSelect").value = bestCheckpoint.id;
    updateSummaryPills();
    clearCompatibilityResult();
    clearEvaluationResults();
    renderContextCard();
    renderTrainingSelectionSummary();
    renderCheckpointDetail();
    showMessage(`Active brain set to ${bestCheckpoint.label}.`);
  });

  container.querySelector(".open-run-inspect-button")?.addEventListener("click", async () => {
    try {
      await openInspectForRun(state.runDetail.id, true);
    } catch (error) {
      showMessage(error.message, "error");
    }
  });

  container.querySelector(".queue-run-robustness-button")?.addEventListener("click", async () => {
    try {
      const job = await apiRequest("/jobs/evaluate", {
        method: "POST",
        body: JSON.stringify({
          job_type: "robustness",
          run_dir: state.runDetail.path,
        }),
      });
      showMessage(`Queued robustness job #${job.id}.`);
      await refreshJobs();
    } catch (error) {
      showMessage(error.message, "error");
    }
  });
}

function renderJobDetail() {
  const label = $("jobDetailLabel");
  const container = $("jobDetailCard");
  if (!state.jobDetail) {
    label.textContent = "No job selected";
    container.className = "empty-state";
    container.textContent = "Select a job to inspect its details.";
    return;
  }
  label.textContent = `Job #${state.jobDetail.id}`;
  const payload = state.jobDetail.payload || {};
  container.className = "list-card";
  const runId = runLabelFromPath(state.jobDetail.run_dir);
  // Resolve the checkpoint this job was resumed from (if any).
  const resumeFrom = payload.resume_from || "";
  const resumeCheckpoint = resumeFrom
    ? state.checkpoints.find((c) => c.path === resumeFrom || c.id === resumeFrom) || null
    : null;
  const resumeLabel = resumeCheckpoint
    ? checkpointUiLabel(resumeCheckpoint)
    : resumeFrom
      ? resumeFrom.replace(/\\/g, "/").split("/").slice(-3).join("/")
      : "";
  container.innerHTML = `
    <h4>Job #${state.jobDetail.id} - ${escapeHtml(state.jobDetail.job_type)}</h4>
    <div class="card-meta">
      <span class="tag">${escapeHtml(state.jobDetail.status)}</span>
      <span class="tag">${escapeHtml(state.jobDetail.created_at || "-")}</span>
      <span class="tag">${escapeHtml(runLabelFromPath(state.jobDetail.run_dir))}</span>
    </div>
    <div class="card-meta">
      ${payload.resume_mode ? `<span class="tag">resume ${escapeHtml(payload.resume_mode)}</span>` : "<span class='tag'>new run</span>"}
      ${payload.episodes ? `<span class="tag">${escapeHtml(String(payload.episodes))} episodes</span>` : ""}
      ${payload.evaluation_episodes ? `<span class="tag">${escapeHtml(String(payload.evaluation_episodes))} eval eps</span>` : ""}
      ${payload.autopilot_after_completion ? "<span class='tag'>autopilot</span>" : ""}
    </div>
    ${resumeLabel ? `<div class="checkpoint-subtitle path-wrap">From: ${escapeHtml(resumeLabel)}</div>` : ""}
    ${state.jobDetail.error_message ? `<p>${escapeHtml(state.jobDetail.error_message)}</p>` : ""}
    <div class="inline-actions">
      ${state.jobDetail.run_dir ? `<button class="button secondary open-job-run-button" data-run-id="${escapeHtml(runId)}" type="button">Open Run</button>` : ""}
      <button class="button secondary refresh-job-log-button" data-job-id="${state.jobDetail.id}" type="button">Refresh Log</button>
    </div>
  `;

  container.querySelector(".open-job-run-button")?.addEventListener("click", async (event) => {
    try {
      await fetchRunDetail(event.target.dataset.runId, true);
      setPage("evaluate");
    } catch (error) {
      showMessage(error.message, "error");
    }
  });

  container.querySelector(".refresh-job-log-button")?.addEventListener("click", async (event) => {
    try {
      state.jobLog = await apiRequest(`/jobs/${Number(event.target.dataset.jobId)}/log`);
      renderJobLog();
      showMessage("Refreshed job log.");
    } catch (error) {
      showMessage(error.message, "error");
    }
  });
}

function renderJobLog() {
  const container = $("jobLogCard");
  if (!state.jobLog) {
    container.className = "empty-state";
    container.textContent = "Select a job to inspect the latest log lines.";
    return;
  }
  container.className = "log-card";
  container.textContent = (state.jobLog.lines || []).join("\n") || "(no log yet)";
}

function renderTrainingSelectionSummary() {
  const container = $("trainingSelectionSummary");
  const gameConfig = selectedGameConfig();
  const trainingConfig = selectedTrainingConfig();
  const checkpoint = selectedCheckpoint();
  const mode = currentTrainingMode();
  const resumeText =
    mode === "train"
      ? "New training ignores the active checkpoint."
        : checkpoint
          ? `Resume source: ${checkpointUiLabel(checkpoint)}`
        : "Select an active checkpoint for resume or fine-tune.";

  container.innerHTML = `
    <h4>Current Selection</h4>
    <div class="card-meta">
      <span class="tag">Game: ${escapeHtml(gameConfig?.label || "-")}</span>
      <span class="tag">Training: ${escapeHtml(trainingConfig?.label || "-")}</span>
      <span class="tag">Mode: ${escapeHtml(mode === "train" ? "new training" : mode === "resume" ? "strict resume" : "fine-tune")}</span>
    </div>
    <p>${escapeHtml(resumeText)}</p>
  `;
}

function renderTrainingPreflight() {
  const container = $("trainingPreflightCard");
  const mode = currentTrainingMode();
  const checkpoint = selectedCheckpoint();
  const gameConfig = selectedGameConfig();
  if (mode === "train") {
    state.trainingPreflight = null;
    container.className = "list-card";
    container.innerHTML = `
      <h4>Launch Check</h4>
      <p>New training will start from random weights.</p>
      <div class="card-meta">
        <span class="tag good">safe to launch</span>
        <span class="tag">game ${escapeHtml(gameConfig?.label || "-")}</span>
      </div>
    `;
    return;
  }

  if (!gameConfig || !checkpoint) {
    container.className = "empty-state";
    container.textContent = "Select both a game config and a checkpoint to validate resume or fine-tune.";
    return;
  }

  if (!state.trainingPreflight) {
    container.className = "empty-state";
    container.textContent = "Checking compatibility for the selected mode...";
    return;
  }

  const strictOkay = String(state.trainingPreflight.strict_resume_status || "").includes("compatible");
  const fineTuneOkay = String(state.trainingPreflight.fine_tune_status || "").includes("compatible");
  const activeOkay = mode === "resume" ? strictOkay : fineTuneOkay;
  const launchTone = activeOkay ? "good" : "bad";
  const activeLabel = mode === "resume" ? "strict resume" : "fine-tune";
  const guidance = activeOkay
    ? `The current ${activeLabel} pair looks usable.`
    : `The current ${activeLabel} pair is not safe to launch.`;

  container.className = "list-card";
  container.innerHTML = `
    <h4>Launch Check</h4>
    <p>${escapeHtml(guidance)}</p>
    <div class="card-meta">
      <span class="tag ${launchTone}">${escapeHtml(activeLabel)} ${activeOkay ? "ready" : "blocked"}</span>
      <span class="tag ${checkpointCompatibilityTone(state.trainingPreflight.strict_resume_status)}">strict ${escapeHtml(state.trainingPreflight.strict_resume_status)}</span>
      <span class="tag ${checkpointCompatibilityTone(state.trainingPreflight.fine_tune_status)}">fine-tune ${escapeHtml(state.trainingPreflight.fine_tune_status)}</span>
    </div>
    <div class="card-meta">
      <span class="tag">brain ${escapeHtml(checkpointUiLabel(checkpoint))}</span>
    </div>
  `;
}

async function fetchAutopilotData(runId) {
  if (!runId) return;
  try {
    const [settings, historyResult, stopStatus] = await Promise.all([
      apiRequest("/autopilot/settings"),
      apiRequest(`/autopilot/history/${encodeURIComponent(runId)}`),
      apiRequest("/autopilot/status"),
    ]);
    state.autopilotSettings = settings;
    state.autopilotHistory = historyResult.items || [];
    state.autopilotStopRequested = stopStatus.stop_requested || false;
  } catch (_error) {
    // Leave previous state intact on error.
  }
  renderAutopilotPanel();
}

function renderAutopilotPanel() {
  const settings = state.autopilotSettings;
  const history = state.autopilotHistory;
  const runDetail = state.runDetail;

  // --- Controls card ---
  const controlCard = $("autopilotControlCard");
  const statusLabel = $("autopilotStatusLabel");

  if (!settings) {
    controlCard.className = "empty-state";
    controlCard.textContent = "Open a training run to manage autopilot.";
    statusLabel.textContent = "-";
  } else {
    const logicOn = settings.logic_enabled;
    const aiOn = settings.ai_enabled;
    const stopPending = state.autopilotStopRequested;
    const lastDecision = history.length ? history[history.length - 1] : null;
    const aiUsed = history.filter((d) => d.advisor === "ai").length;

    statusLabel.textContent = logicOn ? (aiOn ? "logic + AI" : "logic only") : "disabled";

    controlCard.className = "list-card";
    controlCard.innerHTML = `
      <div class="autopilot-toggles">
        <div class="autopilot-toggle-row">
          <span>Logic Autopilot</span>
          <button class="button ${logicOn ? "primary" : "secondary"} autopilot-toggle-btn" data-key="logic_enabled" data-value="${!logicOn}" type="button">
            ${logicOn ? "Enabled" : "Disabled"}
          </button>
        </div>
        <div class="autopilot-toggle-row">
          <span>AI Advisor <em style="font-size:11px;color:var(--muted)">(on plateau)</em></span>
          <button class="button ${aiOn ? "primary" : "secondary"} autopilot-toggle-btn" data-key="ai_enabled" data-value="${!aiOn}" type="button">
            ${aiOn ? "Enabled" : "Disabled"}
          </button>
        </div>
      </div>
      <div class="card-meta">
        <span class="tag ${stopPending ? "bad" : ""}">Stop after cycle: ${stopPending ? "pending" : "off"}</span>
        <span class="tag">AI used: ${aiUsed} / 3</span>
        ${lastDecision ? `<span class="tag">Last: ${actionTag(lastDecision.action)} by ${escapeHtml(lastDecision.advisor)}</span>` : ""}
      </div>
      <div class="inline-actions">
        <button class="button primary" id="startAutopilotLoopButton" type="button">Start Autopilot Loop</button>
        ${stopPending
          ? `<button class="button secondary" id="clearStopButton" type="button">Resume Auto-Chain</button>`
          : `<button class="button secondary" id="requestStopButton" type="button">Stop After Cycle</button>`
        }
      </div>
    `;

    const startLoopBtn = controlCard.querySelector("#startAutopilotLoopButton");
    if (startLoopBtn) {
      startLoopBtn.addEventListener("click", async () => {
        const runId = state.activeRunId;
        if (!runId) { showMessage("No run selected.", "error"); return; }
        try {
          startLoopBtn.disabled = true;
          startLoopBtn.textContent = "Running...";
          const result = await apiRequest(`/autopilot/run/${runId}`, { method: "POST", body: JSON.stringify({}) });
          showMessage(`Autopilot loop started: ${result.action} - ${result.reason}`);
          await fetchAutopilotData(runId);
        } catch (error) {
          showMessage(error.message, "error");
          startLoopBtn.disabled = false;
          startLoopBtn.textContent = "Start Autopilot Loop";
        }
      });
    }

    const stopBtn = controlCard.querySelector("#requestStopButton");
    if (stopBtn) {
      stopBtn.addEventListener("click", async () => {
        try {
          await apiRequest("/autopilot/stop-after-cycle", { method: "POST" });
          showMessage("Autopilot will stop after the current training block.");
          state.autopilotStopRequested = true;
          renderAutopilotPanel();
          renderAutopilotTrainingPanel();
        } catch (error) {
          showMessage(error.message, "error");
        }
      });
    }

    const resumeBtn = controlCard.querySelector("#clearStopButton");
    if (resumeBtn) {
      resumeBtn.addEventListener("click", async () => {
        try {
          await apiRequest("/autopilot/stop-after-cycle", { method: "DELETE" });
          showMessage("Stop request cleared. Autopilot will resume after next block.");
          state.autopilotStopRequested = false;
          renderAutopilotPanel();
          renderAutopilotTrainingPanel();
        } catch (error) {
          showMessage(error.message, "error");
        }
      });
    }

    controlCard.querySelectorAll(".autopilot-toggle-btn").forEach((btn) => {
      btn.addEventListener("click", async () => {
        try {
          state.autopilotSettings = await apiRequest("/autopilot/settings", {
            method: "POST",
            body: JSON.stringify({ [btn.dataset.key]: btn.dataset.value === "true" }),
          });
          renderAutopilotPanel();
          renderAutopilotTrainingPanel();
        } catch (error) {
          showMessage(error.message, "error");
        }
      });
    });
  }

  // --- Run settings card ---
  const settingsCard = $("autopilotRunSettingsCard");
  const tc = runDetail?.training_config;
  if (!tc) {
    settingsCard.className = "empty-state";
    settingsCard.textContent = "Open a training run to see active settings.";
  } else {
    settingsCard.className = "list-card";
    settingsCard.innerHTML = `
      <div class="metric-grid">
        <div class="metric-card"><span>Learning Rate</span><strong>${formatNumber(tc.learning_rate, 6)}</strong></div>
        <div class="metric-card"><span>Epsilon Decay Ep.</span><strong>${tc.epsilon_decay_end?.toLocaleString() ?? "-"}</strong></div>
        <div class="metric-card"><span>Episodes</span><strong>${tc.episodes?.toLocaleString() ?? "-"}</strong></div>
        <div class="metric-card"><span>Gamma</span><strong>${formatNumber(tc.gamma, 4)}</strong></div>
        <div class="metric-card"><span>Batch Size</span><strong>${tc.batch_size ?? "-"}</strong></div>
        <div class="metric-card"><span>Epsilon Min</span><strong>${formatNumber(tc.epsilon_min, 4)}</strong></div>
      </div>
    `;
  }

  // --- Decision history ---
  const decisionsCard = $("autopilotDecisionsCard");
  const countLabel = $("autopilotDecisionCount");
  countLabel.textContent = `${history.length} decision${history.length !== 1 ? "s" : ""}`;

  if (!history.length) {
    decisionsCard.className = "empty-state";
    decisionsCard.textContent = "No autopilot decisions recorded for this run.";
    return;
  }

  decisionsCard.className = "decision-list";
  decisionsCard.innerHTML = [...history].reverse().map((d) => {
    const ts = d.decided_at ? new Date(d.decided_at).toLocaleString() : "-";
    const m = d.metrics || {};
    return `
      <div class="decision-row">
        <div class="decision-row-head">
          ${actionTag(d.action)}
          <span class="tag ${d.advisor === "ai" ? "ai" : ""}">${escapeHtml(d.advisor || "logic")}</span>
          <span class="tag">${escapeHtml(ts)}</span>
          ${d.job_enqueued ? `<span class="tag good">job #${d.job_id} queued</span>` : ""}
        </div>
        <p class="decision-reason">${escapeHtml(d.reason || "")}</p>
        <div class="card-meta">
          ${m.latest_epsilon != null ? `<span class="tag">eps ${formatNumber(m.latest_epsilon, 3)}</span>` : ""}
          ${m.latest_reward != null ? `<span class="tag">reward ${formatNumber(m.latest_reward)}</span>` : ""}
          ${m.bankruptcy_rate != null ? `<span class="tag">bankruptcy ${formatNumber(m.bankruptcy_rate, 3)}</span>` : ""}
          ${m.invalid_action_rate != null ? `<span class="tag">invalid ${formatNumber(m.invalid_action_rate, 3)}</span>` : ""}
          ${m.reward_improvement_ratio != null ? `<span class="tag">improvement ${formatNumber(m.reward_improvement_ratio * 100, 1)}%</span>` : ""}
        </div>
        ${d.next_payload ? `<div class="card-meta">
          <span class="tag info">lr ${formatNumber(d.next_payload.learning_rate, 6)}</span>
          <span class="tag info">eps decay ${d.next_payload.epsilon_decay_end?.toLocaleString()}</span>
          <span class="tag info">${d.next_payload.episodes?.toLocaleString()} ep</span>
        </div>` : ""}
      </div>
    `;
  }).join("");
}

function actionTag(action) {
  const tones = {
    continue: "good",
    lower_lr: "warn",
    extend_epsilon_decay: "info",
    fine_tune: "ai",
    stop: "bad",
  };
  return `<span class="tag ${tones[action] || ""}">${escapeHtml(action || "-")}</span>`;
}

function renderAutopilotTrainingPanel() {
  const card = $("autopilotTrainingCard");
  const label = $("autopilotTrainingStatusLabel");
  const settings = state.autopilotSettings;

  if (!settings) {
    card.className = "empty-state";
    card.textContent = "Connect to the backend to manage autopilot settings.";
    label.textContent = "-";
    return;
  }

  const logicOn = settings.logic_enabled;
  const aiOn = settings.ai_enabled;
  const stopPending = state.autopilotStopRequested;
  label.textContent = logicOn ? (aiOn ? "logic + AI" : "logic only") : "disabled";

  card.className = "list-card";
  card.innerHTML = `
    <div class="autopilot-toggles">
      <div class="autopilot-toggle-row">
        <span>Logic Autopilot</span>
        <button class="button ${logicOn ? "primary" : "secondary"} ap-toggle-btn" data-key="logic_enabled" data-value="${!logicOn}" type="button">
          ${logicOn ? "Enabled" : "Disabled"}
        </button>
      </div>
      <div class="autopilot-toggle-row">
        <span>AI Advisor <em style="font-size:11px;color:var(--muted)">(on plateau)</em></span>
        <button class="button ${aiOn ? "primary" : "secondary"} ap-toggle-btn" data-key="ai_enabled" data-value="${!aiOn}" type="button">
          ${aiOn ? "Enabled" : "Disabled"}
        </button>
      </div>
    </div>
    <div class="card-meta">
      <span class="tag ${stopPending ? "bad" : ""}">Stop after cycle: ${stopPending ? "pending" : "off"}</span>
    </div>
    <div class="inline-actions">
      <button class="button primary" id="startAutopilotTrainingBtn" type="button"
        ${state.trainingProgress?.status === "completed" ? "" : "disabled"}>
        Start Autopilot Loop
      </button>
      ${stopPending
        ? `<button class="button secondary" id="clearStopTrainingBtn" type="button">Resume Auto-Chain</button>`
        : `<button class="button secondary" id="requestStopTrainingBtn" type="button">Stop After Cycle</button>`
      }
    </div>
  `;

  card.querySelectorAll(".ap-toggle-btn").forEach((btn) => {
    btn.addEventListener("click", async () => {
      try {
        state.autopilotSettings = await apiRequest("/autopilot/settings", {
          method: "POST",
          body: JSON.stringify({ [btn.dataset.key]: btn.dataset.value === "true" }),
        });
        renderAutopilotTrainingPanel();
        renderAutopilotPanel();
      } catch (error) {
        showMessage(error.message, "error");
      }
    });
  });

  const startLoopTrainingBtn = card.querySelector("#startAutopilotTrainingBtn");
  if (startLoopTrainingBtn) {
    startLoopTrainingBtn.addEventListener("click", async () => {
      const progress = state.trainingProgress;
      const runId = progress?.run_dir ? runLabelFromPath(progress.run_dir) : state.activeProgressRunId;
      if (!runId) { showMessage("No completed run to analyze.", "error"); return; }
      try {
        startLoopTrainingBtn.disabled = true;
        startLoopTrainingBtn.textContent = "Running...";
        const result = await apiRequest(`/autopilot/run/${runId}`, { method: "POST", body: JSON.stringify({}) });
        showMessage(`Autopilot loop started: ${result.action} - ${result.reason}`);
        await fetchAutopilotData(runId);
      } catch (error) {
        showMessage(error.message, "error");
        renderAutopilotTrainingPanel();
      }
    });
  }

  const stopBtn = card.querySelector("#requestStopTrainingBtn");
  if (stopBtn) {
    stopBtn.addEventListener("click", async () => {
      try {
        await apiRequest("/autopilot/stop-after-cycle", { method: "POST" });
        state.autopilotStopRequested = true;
        renderAutopilotTrainingPanel();
        renderAutopilotPanel();
      } catch (error) { showMessage(error.message, "error"); }
    });
  }

  const resumeBtn = card.querySelector("#clearStopTrainingBtn");
  if (resumeBtn) {
    resumeBtn.addEventListener("click", async () => {
      try {
        await apiRequest("/autopilot/stop-after-cycle", { method: "DELETE" });
        state.autopilotStopRequested = false;
        renderAutopilotTrainingPanel();
        renderAutopilotPanel();
      } catch (error) { showMessage(error.message, "error"); }
    });
  }
}

function renderCampaignPanel() {
  const card = $("campaignCard");
  const label = $("campaignStatusLabel");
  const stopButton = $("stopCampaignButton");
  const escalateButton = $("escalateCampaignButton");
  if (!card || !label || !stopButton || !escalateButton) return;

  const active = state.campaigns.find((campaign) => campaign.status === "running");
  const completed = state.campaigns.find((campaign) => campaign.status === "completed");
  const display = active || completed || null;

  if (!display) {
    state.activeCampaignId = null;
    label.textContent = "-";
    card.className = "empty-state";
    card.textContent = "No active campaign.";
    stopButton.style.display = "none";
    escalateButton.style.display = "none";
    return;
  }

  state.activeCampaignId = display.campaign_id;
  const done = Number(display.variations_completed || 0);
  const total = Number(display.max_variations || 0);
  const pct = total > 0 ? Math.min(100, Math.round((done / total) * 100)) : 0;
  const history = display.variation_history || [];

  label.textContent = display.status;
  card.className = "list-card";
  card.innerHTML = `
    <h4>${escapeHtml(display.campaign_id)}</h4>
    <div class="card-meta">
      <span class="tag">${escapeHtml(display.status)}</span>
      <span class="tag">${done} / ${total} variations</span>
      <span class="tag">${pct}%</span>
      ${display.escalate_mode ? `<span class="tag warn">escalated</span>` : ""}
    </div>
    <div class="card-meta">
      <span class="tag">base ${escapeHtml(display.base_run_id || "-")}</span>
      <span class="tag">current ${escapeHtml(display.current_run_id || "-")}</span>
    </div>
    ${history.length
      ? `<div class="decision-list" style="margin-top:0.5rem;">${history.map((item) => `
          <div class="decision-row">
            <div class="decision-row-head">
              <span class="tag">v${escapeHtml(String(item.index ?? "-"))}</span>
              ${item.escalate ? `<span class="tag warn">escalate</span>` : ""}
              <span class="tag">${escapeHtml(item.to_run || "-")}</span>
            </div>
            <p class="decision-reason">${escapeHtml(item.reason || "")}</p>
          </div>
        `).join("")}</div>`
      : `<p class="muted">Waiting for the first plateau stop.</p>`
    }
  `;
  stopButton.style.display = display.status === "running" ? "" : "none";
  escalateButton.style.display = display.status === "completed" ? "" : "none";
}

async function refreshCampaigns() {
  try {
    state.campaigns = await apiRequest("/campaigns");
  } catch (_error) {
    state.campaigns = [];
  }
  renderCampaignPanel();
}

async function refreshAutopilotSettings() {
  try {
    const [settings, stopStatus] = await Promise.all([
      apiRequest("/autopilot/settings"),
      apiRequest("/autopilot/status"),
    ]);
    state.autopilotSettings = settings;
    state.autopilotStopRequested = stopStatus.stop_requested || false;
  } catch (_error) {
    // Non-fatal. Leave previous state.
  }
  renderAutopilotTrainingPanel();
  renderAutopilotPanel();
}

function renderTrainingProgress() {
  const progressLabel = $("trainingProgressJobLabel");
  const container = $("trainingProgressCard");
  const progress = state.trainingProgress;
  const job = selectedProgressJob();
  const run = selectedProgressRun();

  if (state.activeProgressRunId && !run && !job) {
    state.activeProgressRunId = null;
    state.trainingProgress = null;
  }

  if (!progress) {
    progressLabel.textContent = "No job selected";
    container.className = "empty-state";
    container.textContent = "Open a training job or finished run to see progress.";
    renderLineChart("trainingRewardChart", [], "rolling_average_reward", "#4cb782", "");
    renderLineChart("trainingEvalChart", [], "average_reward", "#7fb7ff", "");
    return;
  }

  const percent = Math.max(0, Math.min(100, Math.round((progress.progress_ratio || 0) * 100)));
  const latest = progress.latest_training_row || {};
  const latestEval = progress.latest_evaluation_row || {};
  const progressStatus = job?.status || progress.status || "completed";
  progressLabel.textContent = job
    ? `Job #${job.id} | ${job.job_type} | ${job.status}`
    : run
      ? `${run.id} | ${progressStatus}`
      : `${runLabelFromPath(progress.run_dir)} | ${progressStatus}`;
  const runPath = job?.run_dir || run?.path || progress.run_dir || "";
  const runName = runLabelFromPath(runPath);
  const completedEpisodes = Number.isFinite(progress.completed_episodes)
    ? progress.completed_episodes
    : (progress.start_episode ? Math.max(0, (progress.latest_episode || 0) - progress.start_episode + 1) : (progress.latest_episode || 0));

  container.className = "progress-stack";
  container.innerHTML = `
    <div class="list-card">
      <h4>${escapeHtml(runName || "Training run")}</h4>
      <p>${progress.total_episodes ? `${completedEpisodes} / ${progress.total_episodes} episodes this run` : `${progress.latest_episode} episodes logged`}</p>
      <div class="progress-track">
        <div class="progress-fill" style="width: ${percent}%"></div>
      </div>
      <div class="card-meta">
        <span class="tag">${percent}%</span>
        <span class="tag">epsilon ${formatNumber(latest.epsilon, 4)}</span>
        <span class="tag">status ${escapeHtml(progressStatus)}</span>
      </div>
    </div>
    <div class="metric-grid">
      <div class="metric-card"><span>Rolling Reward</span><strong>${formatNumber(latest.rolling_average_reward)}</strong></div>
      <div class="metric-card"><span>Episode Reward</span><strong>${formatNumber(latest.episode_reward)}</strong></div>
      <div class="metric-card"><span>Recent Loss</span><strong>${formatNumber(latest.mean_recent_loss)}</strong></div>
      <div class="metric-card"><span>Average Ending Money</span><strong>${formatNumber(latest.average_ending_money)}</strong></div>
      <div class="metric-card"><span>Eval Reward</span><strong>${formatNumber(latestEval.average_reward)}</strong></div>
      <div class="metric-card"><span>Eval Bankruptcy Rate</span><strong>${formatNumber(latestEval.bankruptcy_rate)}</strong></div>
    </div>
  `;

  renderLineChart(
    "trainingRewardChart",
    progress.training_series || [],
    "rolling_average_reward",
    "#4cb782",
    "rolling average reward over logged training points"
  );
  renderLineChart(
    "trainingEvalChart",
    progress.evaluation_series || [],
    "average_reward",
    "#7fb7ff",
    "evaluation reward over periodic evaluation points"
  );
}

function renderCompatibility() {
  const container = $("compatibilityCard");
  if (!state.compatibility) {
    container.className = "empty-state";
    container.textContent = "Select a blueprint and brain, then run compatibility.";
    return;
  }

  container.className = "list-card";
  container.innerHTML = `
    <h4>Compatibility Result</h4>
    <p>${escapeHtml(state.compatibility.message)}</p>
    <div class="card-meta">
      <span class="tag ${checkpointCompatibilityTone(state.compatibility.strict_resume_status)}">Strict: ${escapeHtml(state.compatibility.strict_resume_status)}</span>
      <span class="tag ${checkpointCompatibilityTone(state.compatibility.fine_tune_status)}">Fine-Tune: ${escapeHtml(state.compatibility.fine_tune_status)}</span>
    </div>
  `;
}

function renderCheckpointDetail() {
  const container = $("checkpointDetailCard");
  const checkpoint = state.checkpoints.find((item) => item.id === state.activeCheckpointId);
  if (!checkpoint) {
    container.className = "empty-state";
    container.textContent = "Select a brain from the library or workspace selector to inspect it.";
    return;
  }

  container.className = "list-card";
  container.innerHTML = `
    <h4>${escapeHtml(checkpoint.label)}</h4>
    <p>${escapeHtml(checkpointUiLabel(checkpoint))}</p>
    <div class="card-meta">
      <span class="tag">${escapeHtml(checkpoint.checkpoint_type)}</span>
      ${checkpoint.episode != null ? `<span class="tag">ep ${Number(checkpoint.episode).toLocaleString()}</span>` : ""}
      <span class="tag ${checkpointCompatibilityTone(checkpoint.compatibility_status)}">${escapeHtml(checkpoint.compatibility_status)}</span>
    </div>
    <div class="card-meta">
      <span class="tag">state ${checkpoint.state_dim || "-"}</span>
      <span class="tag">actions ${checkpoint.num_actions || "-"}</span>
    </div>
    <div class="inline-actions">
      <button class="button secondary download-brain-button" type="button">Download Brain</button>
      ${checkpoint.source_type === "run" && checkpoint.source_run ? `<button class="button secondary open-brain-inspect-button" data-run-id="${escapeHtml(checkpoint.source_run)}" type="button">Open Inspect</button>` : ""}
    </div>
  `;

  container.querySelector(".download-brain-button")?.addEventListener("click", () => {
    try {
      downloadCheckpointFile(checkpoint);
      showMessage(`Downloading ${checkpoint.label}.`);
    } catch (error) {
      showMessage(error.message, "error");
    }
  });

  container.querySelector(".open-brain-inspect-button")?.addEventListener("click", async (event) => {
    try {
      await openInspectForRun(event.target.dataset.runId, true);
    } catch (error) {
      showMessage(error.message, "error");
    }
  });
}

function defaultSeatName(type, index) {
  if (type === "human") return "Player";
  if (type === "model-expert") return `AI Expert ${index}`;
  if (type === "model-balanced") return `AI Balanced ${index}`;
  if (type === "model-beginner") return `AI Beginner ${index}`;
  if (type === "heuristic") return `Heuristic AI ${index}`;
  return `Random AI ${index}`;
}

function playSeatPayload(draft, index) {
  const displayName = String(draft.display_name || "").trim() || defaultSeatName(draft.type, index + 1);
  if (draft.type === "human") return { type: "human", display_name: displayName };
  if (draft.type === "heuristic") return { type: "heuristic", display_name: displayName };
  if (draft.type === "random") return { type: "random", display_name: displayName };
  if (draft.type?.startsWith("model-")) {
    return {
      type: "model",
      checkpoint_id: state.activeCheckpointId,
      profile_name: draft.type.replace("model-", ""),
      display_name: displayName,
    };
  }
  return null;
}

function renderPlaySeatEditor() {
  const host = $("playSeatEditor");
  if (!host) return;
  host.innerHTML = state.playSeatDrafts.map((seat, index) => `
    <article class="list-row play-seat-row" data-seat-index="${index}">
      <div class="list-row-head">
        <strong>Seat ${index + 1}</strong>
        <button class="button secondary remove-play-seat-button" type="button" data-seat-index="${index}" ${state.playSeatDrafts.length <= 1 ? "disabled" : ""}>Remove</button>
      </div>
      <div class="grid two">
        <label class="field">
          <span>Seat Type</span>
          <select class="play-seat-type" data-seat-index="${index}">
            <option value="human" ${seat.type === "human" ? "selected" : ""}>Human</option>
            <option value="model-expert" ${seat.type === "model-expert" ? "selected" : ""}>Active Brain Expert</option>
            <option value="model-balanced" ${seat.type === "model-balanced" ? "selected" : ""}>Active Brain Balanced</option>
            <option value="model-beginner" ${seat.type === "model-beginner" ? "selected" : ""}>Active Brain Beginner</option>
            <option value="heuristic" ${seat.type === "heuristic" ? "selected" : ""}>Heuristic AI</option>
            <option value="random" ${seat.type === "random" ? "selected" : ""}>Random AI</option>
          </select>
        </label>
        <label class="field">
          <span>Name</span>
          <input class="play-seat-name" data-seat-index="${index}" type="text" value="${escapeHtml(seat.display_name || "")}" />
        </label>
      </div>
    </article>
  `).join("");

  host.querySelectorAll(".play-seat-type").forEach((select) => {
    select.addEventListener("change", (event) => {
      const index = Number(event.target.dataset.seatIndex);
      state.playSeatDrafts[index].type = event.target.value;
      state.playSeatDrafts[index].display_name = defaultSeatName(event.target.value, index + 1);
      renderPlaySeatEditor();
    });
  });
  host.querySelectorAll(".play-seat-name").forEach((input) => {
    input.addEventListener("input", (event) => {
      const index = Number(event.target.dataset.seatIndex);
      state.playSeatDrafts[index].display_name = event.target.value;
    });
  });
  host.querySelectorAll(".remove-play-seat-button").forEach((button) => {
    button.addEventListener("click", () => {
      if (state.playSeatDrafts.length <= 1) return;
      state.playSeatDrafts.splice(Number(button.dataset.seatIndex), 1);
      renderPlaySeatEditor();
    });
  });
}

function renderPlayBoard() {
  const host = $("playBoardCard");
  const label = $("playBoardStatus");
  if (!host || !label) return;
  const board = state.playSession?.board;
  if (!board) {
    label.textContent = "No session";
    host.className = "empty-state";
    host.textContent = "Start a shared session to see the board.";
    return;
  }

  const products = board.products || [];
  const cellWidth = 112;
  const rowHeight = 58;
  const labelWidth = 120;
  const width = labelWidth + cellWidth * Math.max(1, products[0]?.cells?.length || 1) + 24;
  const height = 36 + rowHeight * products.length + 12;
  const seatColors = ["#557287", "#4f8d68", "#c58b2f", "#cc5f5f"];
  const seatsById = Object.fromEntries((state.playSession.seats || []).map((seat, index) => [
    seat.id,
    { seat, index, color: seatColors[index % seatColors.length] },
  ]));
  const rows = products.map((product, rowIndex) => {
    const y = 32 + rowIndex * rowHeight;
    const cells = (product.cells || []).map((cell, cellIndex) => {
      const x = labelWidth + cellIndex * cellWidth;
      const classes = ["play-board-cell"];
      if (cell.completed) classes.push("is-complete");
      if (cell.active) classes.push("is-active");
      const badges = [
        cell.incident_delta || cell.incident_override !== null ? "I" : "",
        cell.refinement_delta ? "R" : "",
      ].filter(Boolean);
      const seatMarkers = (cell.active_seats || [])
        .map((seatId, markerIndex) => {
          const meta = seatsById[seatId];
          if (!meta) return "";
          const markerX = x + cellWidth - 24 - markerIndex * 18;
          return `<circle cx="${markerX}" cy="${y + 32}" r="7" fill="${meta.color}"><title>Seat ${meta.index + 1}: ${escapeHtml(meta.seat.controller.display_name)}</title></circle>`;
        })
        .join("");
      return `
        <g>
          <rect class="${classes.join(" ")}" x="${x}" y="${y}" width="${cellWidth - 10}" height="44" rx="6"></rect>
          <text class="play-board-sprint" x="${x + 10}" y="${y + 17}">S${cell.sprint}</text>
          <text class="play-board-meta" x="${x + 10}" y="${y + 34}">${formatNumber(cell.sprint_value, 0)} / f${cell.features_required}</text>
          ${badges.map((badge, badgeIndex) => `<text class="play-board-badge" x="${x + cellWidth - 28 - badgeIndex * 16}" y="${y + 17}">${badge}</text>`).join("")}
          ${seatMarkers}
        </g>
      `;
    }).join("");
    return `
      <g>
        <text class="play-board-product" x="10" y="${y + 25}">${escapeHtml(product.name)}</text>
        ${cells}
      </g>
    `;
  }).join("");

  label.textContent = board.incident?.active ? `Incident: ${board.incident.name}` : "Shared board";
  host.className = "play-board-wrap";
  host.innerHTML = `
    <svg class="play-board-svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="Shared play board">
      <text class="play-board-header" x="${labelWidth}" y="18">Shared products and sprints</text>
      ${rows}
    </svg>
  `;
}

function renderPlayTopbar() {
  const sessionCode = $("playSessionCode");
  const roundCode = $("playRoundCode");
  const incidentBanner = $("playIncidentBanner");
  if (!sessionCode || !roundCode || !incidentBanner) return;

  const session = state.playSession;
  sessionCode.textContent = session?.id || "No session";
  roundCode.textContent = session ? String(session.round_number) : "-";

  const incident = session?.board?.incident;
  if (incident?.active) {
    incidentBanner.classList.add("is-active");
    incidentBanner.textContent = `Incident: ${incident.name || "Unknown"}`;
  } else {
    incidentBanner.classList.remove("is-active");
    const latest = session?.round_incidents?.slice(-1)[0];
    incidentBanner.textContent = latest ? `Last incident: ${latest.name}` : "No active incident";
  }
}

function renderPlayStandings() {
  const host = $("playStandingsCard");
  if (!host) return;
  const rows = state.playSession?.standings || [];
  if (!rows.length) {
    host.className = "empty-state";
    host.textContent = "Standings will appear after a session starts.";
    return;
  }
  const seatsById = Object.fromEntries((state.playSession?.seats || []).map((seat) => [seat.id, seat]));
  host.className = "play-standings-list";
  host.innerHTML = rows.map((row) => {
    const seat = seatsById[row.seat_id];
    const product = productNameById(seat?.state?.current_product);
    const bank = Number(row.ending_money || 0);
    return `
      <article class="play-score-card">
        <div class="play-score-top">
          <span class="play-score-name">${escapeHtml(row.controller)}</span>
          <strong class="play-bank ${bank < 0 ? "is-negative" : ""}">${formatCurrency(bank)}</strong>
        </div>
        <div class="play-score-meta">
          <span class="tag">${escapeHtml(row.type)}</span>
          <span class="tag">${escapeHtml(product)} S${escapeHtml(String(seat?.state?.current_sprint ?? "-"))}</span>
          <span class="tag ${seat?.state?.loan_active ? "bad" : "good"}">${seat?.state?.loan_active ? "loan" : "cash"}</span>
          <span class="tag ${row.done ? "bad" : "good"}">${row.done ? "done" : "active"}</span>
        </div>
      </article>
    `;
  }).join("");
}

function renderPlayTurnLog() {
  const host = $("playTurnLogCard");
  if (!host) return;
  const rows = state.playSession?.turn_log || [];
  if (!rows.length) {
    host.className = "empty-state";
    host.textContent = "Turns will appear after the first round.";
    return;
  }
  host.className = "play-turn-log-list";
  host.innerHTML = rows.slice(-24).reverse().map((row) => `
    <article class="play-log-row">
      <strong>R${escapeHtml(String(row.round))} - ${escapeHtml(row.controller)} - ${escapeHtml(row.action)}</strong>
      <div class="play-log-meta">
        <span class="tag ${row.outcome === "Success" ? "good" : row.outcome === "Invalid" ? "bad" : ""}">${escapeHtml(row.outcome)}</span>
        <span class="tag">bank ${formatCurrency(row.bank)}</span>
        <span class="tag ${Number(row.reward) >= 0 ? "good" : "bad"}">reward ${formatCurrency(row.reward)}</span>
      </div>
      <p>${escapeHtml(row.product || "-")} sprint ${escapeHtml(String(row.sprint ?? "-"))} · ${escapeHtml(row.refinement || "none")}</p>
    </article>
  `).join("");
}

function productNameById(productId) {
  const product = state.playSession?.board?.products?.find((item) => Number(item.product_id) === Number(productId));
  return product?.name || `Product ${productId || "-"}`;
}

function formatCurrency(value) {
  const amount = Number(value || 0);
  return `${amount < 0 ? "-" : ""}€${Math.abs(Math.round(amount)).toLocaleString()}`;
}

function renderPlayActionButtons(humanSeat) {
  const host = $("playActionButtonGrid");
  const select = $("playHumanActionSelect");
  if (!host || !select) return;

  if (!humanSeat) {
    host.innerHTML = "";
    return;
  }

  const validActions = new Map((humanSeat.valid_actions || []).map((action) => [Number(action.action_id), action]));
  const products = state.playSession?.board?.products || [];
  const allActions = [
    { action_id: 0, label: "Continue", hint: "Play current sprint" },
    ...products.map((product) => ({
      action_id: Number(product.product_id),
      label: `Switch to ${product.name}`,
      hint: `Start S${humanSeat.state?.target_next_sprints?.[Number(product.product_id) - 1] || 1}`,
    })),
  ];
  const selected = Number(select.value || (humanSeat.valid_actions || [])[0]?.action_id || 0);

  host.innerHTML = allActions.map((action) => {
    const valid = validActions.has(Number(action.action_id));
    const label = validActions.get(Number(action.action_id))?.label || action.label;
    return `
      <button class="play-action-button ${selected === Number(action.action_id) ? "is-selected" : ""}" type="button"
        data-action-id="${action.action_id}" ${valid ? "" : "disabled"}>
        <strong>${escapeHtml(label)}</strong>
        <span>${valid ? escapeHtml(action.hint || "Available") : "Locked"}</span>
      </button>
    `;
  }).join("");

  host.querySelectorAll(".play-action-button").forEach((button) => {
    button.addEventListener("click", () => {
      select.value = button.dataset.actionId;
      renderPlayActionButtons(humanSeat);
      renderPlayDicePreview(humanSeat, Number(button.dataset.actionId));
    });
  });
}

function latestPlayTurn() {
  const rows = state.playSession?.turn_log || [];
  return rows.length ? rows[rows.length - 1] : null;
}

function showPlayDiceOverlay() {
  $("playDiceOverlay")?.classList.remove("hidden");
}

function hidePlayDiceOverlay() {
  $("playDiceOverlay")?.classList.add("hidden");
}

function diceNotationFromTurnDice(dice) {
  const firstScrum = dice?.daily_scrums?.[0];
  const diceCount = Number(firstScrum?.dice_count || 0);
  const diceSides = Number(firstScrum?.dice_sides || 0);
  const scrumCount = Number(dice?.daily_scrums?.length || 0);
  if (!diceCount || !diceSides || !scrumCount) return "1d6";
  return `${diceCount * scrumCount}d${diceSides}`;
}

async function ensurePlayDiceBox() {
  if (state.playDiceBoxReady && state.playDiceBox) return state.playDiceBox;
  if (state.playDiceBoxInitPromise) return state.playDiceBoxInitPromise;

  state.playDiceBoxInitPromise = (async () => {
    const host = $("playDiceBox");
    if (!host) return null;
    const module = await import(DICE_BOX_MODULE_URL);
    const DiceBox = module.default;
    const box = new DiceBox("#playDiceBox", {
      assetPath: DICE_BOX_ASSET_PATH,
      theme: "default",
      scale: 6,
      gravity: 1,
      mass: 1,
      friction: 0.8,
      restitution: 0.25,
    });
    await box.init();
    state.playDiceBox = box;
    state.playDiceBoxReady = true;
    return box;
  })().catch((error) => {
    state.playDiceBoxReady = false;
    state.playDiceBox = null;
    state.playDiceBoxInitPromise = null;
    console.warn("3D dice failed to initialize; using fallback dice.", error);
    return null;
  });

  return state.playDiceBoxInitPromise;
}

function renderFallbackDice(dice) {
  const host = $("playDiceBox");
  if (!host) return;
  const firstScrum = dice?.daily_scrums?.[0];
  const sides = Number(firstScrum?.dice_sides || 6);
  const rolls = (dice?.daily_scrums || [])
    .flatMap((scrum) => scrum.rolls || [])
    .slice(0, 12);
  host.innerHTML = `
    <div class="fallback-dice-stage">
      ${rolls.map((roll, index) => `
        <div class="fallback-die" style="animation-delay:${index * 35}ms">
          ${escapeHtml(String(Math.max(1, Math.min(Number(roll || 1), sides))))}
        </div>
      `).join("")}
    </div>
  `;
}

async function rollPlayDice(dice) {
  showPlayDiceOverlay();
  const slot = $("playDiceCard")?.querySelector(".dice-animation-slot");
  slot?.classList.add("is-rolling");
  const boxHost = $("playDiceBox");
  if (boxHost) {
    boxHost.innerHTML = `<div class="fallback-dice-stage"><div class="fallback-die">?</div></div>`;
  }

  const diceBox = await ensurePlayDiceBox();
  if (diceBox) {
    try {
      await diceBox.roll(diceNotationFromTurnDice(dice));
      slot?.classList.remove("is-rolling");
      return;
    } catch (error) {
      console.warn("3D dice roll failed; using fallback dice.", error);
    }
  }

  renderFallbackDice(dice);
  window.setTimeout(() => slot?.classList.remove("is-rolling"), 650);
}

function renderPlayDicePreview(humanSeat, actionId) {
  const host = $("playDiceCard");
  if (!host || !humanSeat) return;
  const action = (humanSeat.valid_actions || []).find((item) => Number(item.action_id) === Number(actionId));
  host.innerHTML = `
    <div class="dice-animation-slot">Dice animation slot</div>
    <div class="dice-summary">
      <div class="dice-stat"><span>Queued</span><strong>${escapeHtml(action?.label || "Action")}</strong></div>
      <div class="dice-stat"><span>Product</span><strong>${escapeHtml(productNameById(actionId || humanSeat.state?.current_product))}</strong></div>
      <div class="dice-stat"><span>Target</span><strong>60</strong></div>
    </div>
    <p>Submit action to roll the five Daily Scrums.</p>
  `;
}

function renderPlayDiceZone() {
  const host = $("playDiceCard");
  if (!host) return;
  const row = latestPlayTurn();
  const dice = row?.dice;
  if (!dice?.daily_scrums?.length) {
    host.innerHTML = `
      <div class="dice-animation-slot">Dice animation slot</div>
      <p>Select an action to reveal the daily scrum math.</p>
    `;
    return;
  }

  const variance = Number(dice.variance || 0);
  const payout = Number(dice.payout || 0);
  host.innerHTML = `
    <div class="dice-animation-slot">3D dice stage</div>
    <div class="dice-summary">
      <div class="dice-stat"><span>Rolling</span><strong>${escapeHtml(dice.dice_label)}</strong></div>
      <div class="dice-stat"><span>Total / Target</span><strong>${dice.total_rolled} / ${dice.target_total}</strong></div>
      <div class="dice-stat"><span>Variance</span><strong class="${variance <= 0 ? "play-number-good" : "play-number-bad"}">${variance}</strong></div>
    </div>
    <div class="dice-roll-list">
      ${dice.daily_scrums.map((scrum) => `
        <div class="dice-roll-row">
          <strong>D${escapeHtml(String(scrum.scrum_number))}</strong>
          <span class="dice-roll-values">[${(scrum.rolls || []).map((roll) => escapeHtml(String(roll))).join(", ")}]</span>
          <span class="dice-roll-total">${escapeHtml(String(scrum.roll_total))}</span>
        </div>
      `).join("")}
    </div>
    <div class="dice-summary">
      <div class="dice-stat"><span>Penalty</span><strong class="play-number-bad">-${formatCurrency(dice.planning_penalty)}</strong></div>
      <div class="dice-stat"><span>Payout</span><strong class="${payout >= 0 ? "play-number-good" : "play-number-bad"}">${formatCurrency(payout)}</strong></div>
      <div class="dice-stat"><span>Result</span><strong>${escapeHtml(row.outcome || "-")}</strong></div>
    </div>
  `;
}

function renderPlaySession() {
  const card = $("playSessionCard");
  const humanWrap = $("playHumanActionWrap");
  const setupForm = $("playSessionForm");
  const addSeatButton = $("addPlaySeatButton");
  const actionPanel = document.querySelector(".play-action-panel");
  renderPlayTopbar();
  if (!state.playSession) {
    actionPanel?.classList.remove("is-playing");
    actionPanel?.classList.add("is-setup");
    setupForm?.classList.remove("hidden");
    if (addSeatButton) addSeatButton.classList.remove("hidden");
    card.className = "play-session-summary hidden";
    card.textContent = "";
    humanWrap.className = "play-action-wrap hidden";
    renderPlayBoard();
    renderPlayStandings();
    renderPlayTurnLog();
    renderPlayDiceZone();
    return;
  }

  actionPanel?.classList.remove("is-setup");
  actionPanel?.classList.add("is-playing");
  setupForm?.classList.add("hidden");
  if (addSeatButton) addSeatButton.classList.add("hidden");
  card.className = "play-session-summary play-session-summary-card";
  card.innerHTML = `
    <div class="play-score-meta">
      <span class="tag">${state.playSession.done ? "complete" : "in progress"}</span>
      <span class="tag">${state.playSession.seats.length} seats</span>
      <span class="tag">seed ${state.playSession.base_seed}</span>
    </div>
  `;

  const humanSeat = state.playSession.seats.find((seat) => seat.controller.type === "human" && !seat.done);
  renderPlayBoard();
  renderPlayStandings();
  renderPlayTurnLog();
  renderPlayDiceZone();
  if (!humanSeat) {
    humanWrap.className = "play-action-wrap hidden";
    renderPlayActionButtons(null);
    return;
  }

  humanWrap.className = "play-action-wrap";
  buildOptions("playHumanActionSelect", humanSeat.valid_actions || [], "action_id", "label", "No actions");
  renderPlayActionButtons(humanSeat);
}

function parseSeedList(value) {
  return String(value || "")
    .split(",")
    .map((item) => Number(item.trim()))
    .filter((item) => Number.isFinite(item));
}

function renderDirectEvaluation() {
  const host = $("directEvaluationResult");
  const metricsHost = $("directEvaluationMetrics");
  if (!state.directEvaluation) {
    host.className = "empty-state";
    host.textContent = "Run a direct evaluation for the active brain and blueprint.";
    if (metricsHost) metricsHost.innerHTML = "";
    renderBarChart("directEvaluationChart", [], "episode_reward", "seed", "#6b8aa3", "#cc5f5f", "");
    renderBarChart("directEvaluationMoneyChart", [], "ending_money", "seed", "#6b8aa3", "#cc5f5f", "");
    renderBarChart("directEvaluationTurnsChart", [], "turns_played", "seed", "#5a9e7a", "#5a9e7a", "");
    renderTable("directEvaluationTable", [], []);
    return;
  }
  const summary = state.directEvaluation.summary;
  const winRate = ((1 - summary.bankruptcies / summary.episodes) * 100).toFixed(1);
  host.className = "list-card";
  host.innerHTML = `
    <h4>${escapeHtml(state.directEvaluation.checkpoint.label)}</h4>
    <p>Greedy evaluation across ${summary.episodes} episode${summary.episodes !== 1 ? "s" : ""}.</p>
  `;
  if (metricsHost) {
    metricsHost.innerHTML = `
      <div class="metric-card">
        <span>Mean Reward</span>
        <strong>${Number(summary.mean_reward).toFixed(2)}</strong>
      </div>
      <div class="metric-card">
        <span>Mean Bank</span>
        <strong>${Number(summary.mean_ending_money).toFixed(2)}</strong>
      </div>
      <div class="metric-card">
        <span>Win Rate</span>
        <strong>${winRate}%</strong>
      </div>
      <div class="metric-card">
        <span>Bankruptcies</span>
        <strong>${summary.bankruptcies} / ${summary.episodes}</strong>
      </div>
      <div class="metric-card">
        <span>Avg Turns</span>
        <strong>${Number(summary.mean_turns_played).toFixed(1)}</strong>
      </div>
      <div class="metric-card">
        <span>Avg Loans Taken</span>
        <strong>${Number(summary.mean_loans_taken).toFixed(2)}</strong>
      </div>
      <div class="metric-card">
        <span>Reward Range</span>
        <strong>${Number(summary.min_reward).toFixed(1)} - ${Number(summary.max_reward).toFixed(1)}</strong>
      </div>
      <div class="metric-card">
        <span>Invalid Actions</span>
        <strong>${summary.invalid_actions}</strong>
      </div>
    `;
  }
  renderBarChart(
    "directEvaluationChart",
    state.directEvaluation.results || [],
    "episode_reward",
    "seed",
    "#6b8aa3",
    "#cc5f5f",
    "episode reward by seed"
  );
  renderBarChart(
    "directEvaluationMoneyChart",
    state.directEvaluation.results || [],
    "ending_money",
    "seed",
    "#6b8aa3",
    "#cc5f5f",
    "ending money by seed"
  );
  renderBarChart(
    "directEvaluationTurnsChart",
    state.directEvaluation.results || [],
    "turns_played",
    "seed",
    "#5a9e7a",
    "#5a9e7a",
    "turns played by seed"
  );
  renderTable(
    "directEvaluationTable",
    [
      { key: "seed", label: "Seed" },
      { key: "episode_reward", label: "Reward" },
      { key: "ending_money", label: "Bank" },
      { key: "turns_played", label: "Turns" },
      { key: "loan_turns", label: "Loan Turns" },
      { key: "loans_taken", label: "Loans" },
      { key: "invalid_action_count", label: "Invalid" },
      { key: "terminal_reason", label: "End" },
    ],
    state.directEvaluation.results || []
  );
}

function renderCheckpointComparison() {
  const host = $("checkpointCompareResult");
  const metricsHost = $("comparisonMetrics");
  const compareOptions = sidebarCheckpointOptions()
    .filter((item) => item.id !== state.activeCheckpointId);
  buildOptions("compareCheckpointSelect", compareOptions, "id", "ui_label", "No comparison brains");
  if (!state.comparisonEvaluation) {
    host.className = "empty-state";
    host.textContent = "Compare the selected right-side brain against the active brain from the top bar.";
    if (metricsHost) metricsHost.innerHTML = "";
    renderBarChart("comparisonChart", [], "reward_delta", "seed", "#6b8aa3", "#cc5f5f", "");
    renderBarChart("comparisonMoneyChart", [], "bank_delta", "seed", "#6b8aa3", "#cc5f5f", "");
    renderTable("comparisonTable", [], []);
    return;
  }
  const left = state.comparisonEvaluation.left;
  const right = state.comparisonEvaluation.right;
  const comparisonRows = (left.results || []).map((leftRow, index) => {
    const rightRow = right.results?.[index] || {};
    return {
      seed: leftRow.seed,
      left_reward: leftRow.episode_reward,
      right_reward: rightRow.episode_reward,
      reward_delta: Number(leftRow.episode_reward || 0) - Number(rightRow.episode_reward || 0),
      left_bank: leftRow.ending_money,
      right_bank: rightRow.ending_money,
      bank_delta: Number(leftRow.ending_money || 0) - Number(rightRow.ending_money || 0),
      left_turns: leftRow.turns_played,
      right_turns: rightRow.turns_played,
      left_terminal: leftRow.terminal_reason || "completed",
      right_terminal: rightRow.terminal_reason || "completed",
    };
  });
  host.className = "list-card";
  host.innerHTML = `
    <h4>${escapeHtml(left.checkpoint.label)} <span style="color:var(--muted)">vs</span> ${escapeHtml(right.checkpoint.label)}</h4>
    <p>Left is the active brain. Positive delta means left wins.</p>
  `;
  if (metricsHost) {
    const leftWinRate = ((1 - left.summary.bankruptcies / left.summary.episodes) * 100).toFixed(1);
    const rightWinRate = ((1 - right.summary.bankruptcies / right.summary.episodes) * 100).toFixed(1);
    metricsHost.innerHTML = `
      <div class="metric-card">
        <span>Delta Reward</span>
        <strong>${Number(state.comparisonEvaluation.delta_mean_reward).toFixed(2)}</strong>
      </div>
      <div class="metric-card">
        <span>Delta Bank</span>
        <strong>${Number(state.comparisonEvaluation.delta_mean_ending_money).toFixed(2)}</strong>
      </div>
      <div class="metric-card">
        <span>Left Reward</span>
        <strong>${Number(left.summary.mean_reward).toFixed(2)}</strong>
      </div>
      <div class="metric-card">
        <span>Right Reward</span>
        <strong>${Number(right.summary.mean_reward).toFixed(2)}</strong>
      </div>
      <div class="metric-card">
        <span>Left Win Rate</span>
        <strong>${leftWinRate}%</strong>
      </div>
      <div class="metric-card">
        <span>Right Win Rate</span>
        <strong>${rightWinRate}%</strong>
      </div>
      <div class="metric-card">
        <span>Left Bank</span>
        <strong>${Number(left.summary.mean_ending_money).toFixed(2)}</strong>
      </div>
      <div class="metric-card">
        <span>Right Bank</span>
        <strong>${Number(right.summary.mean_ending_money).toFixed(2)}</strong>
      </div>
    `;
  }
  renderBarChart(
    "comparisonChart",
    comparisonRows,
    "reward_delta",
    "seed",
    "#6b8aa3",
    "#cc5f5f",
    "left minus right reward by seed"
  );
  renderBarChart(
    "comparisonMoneyChart",
    comparisonRows,
    "bank_delta",
    "seed",
    "#6b8aa3",
    "#cc5f5f",
    "left minus right bank by seed"
  );
  renderTable(
    "comparisonTable",
    [
      { key: "seed", label: "Seed" },
      { key: "left_reward", label: "L Reward" },
      { key: "right_reward", label: "R Reward" },
      { key: "reward_delta", label: "Delta Reward" },
      { key: "left_bank", label: "L Bank" },
      { key: "right_bank", label: "R Bank" },
      { key: "bank_delta", label: "Delta Bank" },
      { key: "left_turns", label: "L Turns" },
      { key: "right_turns", label: "R Turns" },
      { key: "left_terminal", label: "L End" },
      { key: "right_terminal", label: "R End" },
    ],
    comparisonRows
  );
}

function syncSelectors() {
  buildOptions("activeGameConfigSelect", state.gameConfigs);
  buildOptions("activeTrainingConfigSelect", state.trainingConfigs);
  const checkpointOptions = sidebarCheckpointOptions();
  buildOptions("activeCheckpointSelect", checkpointOptions, "id", "ui_label");

  if (!state.activeGameConfigId && state.gameConfigs.length) {
    state.activeGameConfigId = state.gameConfigs[0].id;
  }
  if (!state.activeTrainingConfigId && state.trainingConfigs.length) {
    state.activeTrainingConfigId = state.trainingConfigs[0].id;
  }
  if (!checkpointOptions.some((item) => item.id === state.activeCheckpointId)) {
    state.activeCheckpointId = checkpointOptions[0]?.id || "";
  }

  $("activeGameConfigSelect").value = state.activeGameConfigId || "";
  $("activeTrainingConfigSelect").value = state.activeTrainingConfigId || "";
  $("activeCheckpointSelect").value = state.activeCheckpointId || "";
  $("activeCheckpointIncludeAllToggle").checked = state.includeCheckpointSelections;
}

async function loadActiveGameConfigIntoEditor() {
  if (!state.activeGameConfigId) {
    showMessage("Select a game config first.", "error");
    return;
  }
  const payload = await apiRequest(`/configs/game/${encodeURIComponent(state.activeGameConfigId)}`);
  state.activeGameConfigPayload = payload;
  state.visualGameConfig = clone(payload.config);
  $("gameConfigEditor").value = formatJson(payload.config);
  $("gameConfigFileNameInput").value = payload.id === "default_game_config" ? "my_game_config" : payload.id;
  renderVisualEditor();
}

async function loadActiveTrainingConfigIntoEditor() {
  if (!state.activeTrainingConfigId) {
    showMessage("Select a training config first.", "error");
    return;
  }
  const payload = await apiRequest(`/configs/training/${encodeURIComponent(state.activeTrainingConfigId)}`);
  state.activeTrainingConfigPayload = payload;
  $("trainingConfigEditor").value = formatJson(payload.config);
  $("trainingConfigFileNameInput").value = payload.id === "default_training_config" ? "my_training_config" : payload.id;
}

async function saveGameConfig(overwrite = false) {
  let config;
  try {
    config = readVisualEditorIntoState();
    $("gameConfigEditor").value = formatJson(config);
  } catch (_error) {
    config = parseJsonEditor("gameConfigEditor");
  }
  const body = {
    config,
    file_name: $("gameConfigFileNameInput").value.trim(),
  };
  if (overwrite) {
    body.id = state.activeGameConfigId;
  }
  const payload = await apiRequest("/configs/game", {
    method: "POST",
    body: JSON.stringify(body),
  });
  showMessage(`Saved game config ${payload.label}.`);
  await refreshAll();
  state.activeGameConfigId = payload.id;
  await loadActiveGameConfigIntoEditor();
  await validateGameConfigDraft(false).catch(() => {});
}

async function saveTrainingConfig(overwrite = false) {
  const config = parseJsonEditor("trainingConfigEditor");
  const body = {
    config,
    file_name: $("trainingConfigFileNameInput").value.trim(),
  };
  if (overwrite) {
    body.id = state.activeTrainingConfigId;
  }
  const payload = await apiRequest("/configs/training", {
    method: "POST",
    body: JSON.stringify(body),
  });
  showMessage(`Saved training config ${payload.label}.`);
  await refreshAll();
  state.activeTrainingConfigId = payload.id;
  await loadActiveTrainingConfigIntoEditor();
  await validateTrainingConfigDraft(false).catch(() => {});
}

async function refreshJobs() {
  const payload = await apiRequest("/jobs");
  state.jobs = payload.items || [];
  renderJobs();
  updateSummaryPills();
  renderTrainingSelectionSummary();
  await refreshCampaigns().catch(() => {});
  if (state.activeJobDetailId) {
    await fetchJobDetail(state.activeJobDetailId, false).catch(() => {});
  }
}

async function fetchTrainingProgress(jobId, announce = false) {
  state.activeProgressJobId = Number(jobId);
  state.activeProgressRunId = null;
  state.trainingProgress = await apiRequest(`/jobs/${state.activeProgressJobId}/progress`);
  renderTrainingProgress();
  if (announce) {
    showMessage(`Loaded progress for job ${jobId}.`);
  }
}

async function fetchRunProgress(runId, announce = false) {
  state.activeProgressRunId = runId;
  state.activeProgressJobId = null;
  state.trainingProgress = await apiRequest(`/runs/${encodeURIComponent(runId)}/progress`);
  renderTrainingProgress();
  if (announce) {
    showMessage(`Loaded progress for run ${runId}.`);
  }
}

async function refreshTrainingPreflight() {
  const mode = currentTrainingMode();
  const checkpoint = selectedCheckpoint();
  const gameConfig = selectedGameConfig();
  if (mode === "train") {
    state.trainingPreflight = null;
    renderTrainingPreflight();
    return;
  }
  if (!checkpoint || !gameConfig) {
    state.trainingPreflight = null;
    renderTrainingPreflight();
    return;
  }
  try {
    state.trainingPreflight = await apiRequest(
      `/checkpoints/${encodeURIComponent(checkpoint.id)}/compatibility?game_config_id=${encodeURIComponent(gameConfig.id)}`
    );
  } catch (_error) {
    state.trainingPreflight = null;
  }
  renderTrainingPreflight();
}

async function validateGameConfigDraft(showSuccess = false) {
  try {
    const config = readVisualEditorIntoState();
    $("gameConfigEditor").value = formatJson(config);
    state.gameConfigValidation = await apiRequest("/configs/game/validate", {
      method: "POST",
      body: JSON.stringify({ config }),
    });
    renderGameConfigValidation();
    if (showSuccess) {
      showMessage("Game config validation passed.");
    }
    return state.gameConfigValidation;
  } catch (error) {
    state.gameConfigValidation = { valid: false, error: error.message };
    renderGameConfigValidation();
    if (showSuccess) {
      showMessage(error.message, "error");
    }
    throw error;
  }
}

async function validateTrainingConfigDraft(showSuccess = false) {
  try {
    const config = parseJsonEditor("trainingConfigEditor");
    state.trainingConfigValidation = await apiRequest("/configs/training/validate", {
      method: "POST",
      body: JSON.stringify({ config }),
    });
    renderTrainingConfigValidation();
    if (showSuccess) {
      showMessage("Training config validation passed.");
    }
    return state.trainingConfigValidation;
  } catch (error) {
    state.trainingConfigValidation = { valid: false, error: error.message };
    renderTrainingConfigValidation();
    if (showSuccess) {
      showMessage(error.message, "error");
    }
    throw error;
  }
}

async function refreshPlaySession() {
  if (!state.playSession?.id) {
    renderPlaySession();
    return;
  }
  state.playSession = await apiRequest(`/play/session/${encodeURIComponent(state.playSession.id)}`);
  renderPlaySession();
}

async function fetchRunDetail(runId, announce = false) {
  state.activeRunId = runId;
  state.runRating = null;
  state.runDetail = await apiRequest(`/runs/${encodeURIComponent(runId)}`);
  renderRunDetail();
  // Fetch rating in the background. It does not block the inspect panel.
  apiRequest(`/runs/${encodeURIComponent(runId)}/rating`).then((rating) => {
    state.runRating = rating;
    renderRunDetail();
  }).catch(() => {});
  if (announce) {
    showMessage(`Loaded run ${runId}.`);
  }
}

async function fetchJobDetail(jobId, announce = false) {
  state.activeJobDetailId = Number(jobId);
  state.jobDetail = await apiRequest(`/jobs/${state.activeJobDetailId}`);
  renderJobDetail();
  try {
    state.jobLog = await apiRequest(`/jobs/${state.activeJobDetailId}/log`);
  } catch (_error) {
    state.jobLog = null;
  }
  renderJobLog();
  if (announce) {
    showMessage(`Loaded job ${jobId}.`);
  }
}

async function openInspectForRun(runId, announce = true) {
  await fetchRunDetail(runId, false);
  await fetchAutopilotData(runId).catch(() => {});
  const linkedJob = jobForRunId(runId);
  if (linkedJob) {
    await fetchJobDetail(linkedJob.id, false).catch(() => {});
    if (["train", "fine_tune"].includes(linkedJob.job_type)) {
      await fetchTrainingProgress(linkedJob.id, false).catch(() => {});
    }
  } else {
    state.activeJobDetailId = null;
    state.jobDetail = null;
    state.jobLog = null;
    renderJobDetail();
    renderJobLog();
    await fetchRunProgress(runId, false).catch(() => {
      state.activeProgressRunId = null;
      state.activeProgressJobId = null;
      state.trainingProgress = null;
      renderTrainingProgress();
    });
  }
  setPage("inspect");
  if (announce) {
    showMessage(`Opened inspect view for ${runId}.`);
  }
}

async function openInspectForJob(jobId, announce = true) {
  await fetchJobDetail(jobId, false);
  const job = state.jobDetail;
  if (job?.run_dir) {
    await fetchRunDetail(runLabelFromPath(job.run_dir), false).catch(() => {});
  }
  if (job?.run_dir) {
    await fetchAutopilotData(runLabelFromPath(job.run_dir)).catch(() => {});
  }
  if (job && ["train", "fine_tune"].includes(job.job_type)) {
    await fetchTrainingProgress(job.id, false).catch(() => {});
  } else {
    state.activeProgressRunId = null;
    state.activeProgressJobId = null;
    state.trainingProgress = null;
    renderTrainingProgress();
  }
  setPage("inspect");
  if (announce) {
    showMessage(`Opened inspect view for job ${jobId}.`);
  }
}

async function refreshCheckpoints() {
  // Checkpoints load .pth files via torch and can be slow on first call.
  // Fetched separately so a slow response never blocks the connect flow.
  try {
    const checkpoints = await apiRequest("/checkpoints", {}, 120000);
    state.checkpoints = checkpoints.items || [];
    syncSelectors();
    renderCheckpointDetail();
    renderTrainingSelectionSummary();
    renderTrainingPreflight();
    renderCompatibility();
  } catch (_err) {
    // Non-fatal. UI still works without checkpoints loaded.
  }
}

async function refreshAll() {
  clearMessage();
  const [health, gameConfigs, trainingConfigs, runs, jobs] = await Promise.all([
    apiRequest("/health"),
    apiRequest("/configs/game"),
    apiRequest("/configs/training"),
    apiRequest("/runs"),
    apiRequest("/jobs"),
  ]);

  state.health = health;
  state.gameConfigs = gameConfigs.items || [];
  state.trainingConfigs = trainingConfigs.items || [];
  state.runs = runs.items || [];
  state.jobs = jobs.items || [];

  // Load checkpoints in the background. Slow on first call (torch init + .pth loads).
  refreshCheckpoints();

  syncSelectors();
  renderGameConfigs();
  renderTrainingConfigs();
  renderRuns();
  renderRunDetail();
  renderCheckpointDetail();
  renderJobs();
  renderJobDetail();
  renderJobLog();
  renderTrainingSelectionSummary();
  renderTrainingProgress();
  renderTrainingPreflight();
  renderGameConfigValidation();
  renderTrainingConfigValidation();
  renderCompatibility();
  renderPlaySession();
  renderDirectEvaluation();
  renderCheckpointComparison();
  renderCampaignPanel();
  updateStatusCard();
  updateSummaryPills();
  renderContextCard();
  await refreshTrainingPreflight();
  await refreshAutopilotSettings().catch(() => {});
  await refreshCampaigns().catch(() => {});

  if (!$("gameConfigEditor").value && state.gameConfigs.length) {
    await loadActiveGameConfigIntoEditor();
  }
  if (!$("trainingConfigEditor").value && state.trainingConfigs.length) {
    await loadActiveTrainingConfigIntoEditor();
  }
}

async function runCompatibility() {
  if (!state.activeCheckpointId || !state.activeGameConfigId) {
    showMessage("Select both a game config and a checkpoint first.", "error");
    return;
  }
  const checkpointId = encodeURIComponent(state.activeCheckpointId);
  const gameConfigId = encodeURIComponent(state.activeGameConfigId);
  state.compatibility = await apiRequest(`/checkpoints/${checkpointId}/compatibility?game_config_id=${gameConfigId}`);
  renderCompatibility();
}

async function queueTrainingJob(event) {
  event.preventDefault();
  const mode = currentTrainingMode();
  const activeCheckpoint = selectedCheckpoint();
  if (mode !== "train") {
    if (!activeCheckpoint) {
      showMessage("Select an active checkpoint before resume or fine-tune.", "error");
      return;
    }
    await refreshTrainingPreflight();
    if (mode === "resume" && !String(state.trainingPreflight?.strict_resume_status || "").includes("compatible")) {
      showMessage("Strict resume is blocked for the selected checkpoint and game config.", "error");
      return;
    }
    if (mode === "fine_tune" && !String(state.trainingPreflight?.fine_tune_status || "").includes("compatible")) {
      showMessage("Fine-tune is blocked for the selected checkpoint and game config.", "error");
      return;
    }
  }

  const payload = {
    game_config_path: selectedGameConfig()?.path || "",
    training_config_path: selectedTrainingConfig()?.path || "",
    episodes: Number($("trainEpisodesInput").value),
    evaluation_episodes: Number($("trainEvalEpisodesInput").value),
    agent_type: $("trainAgentTypeSelect").value,
    run_name: $("trainRunNameInput").value.trim(),
    run_notes: $("trainNotesInput").value.trim(),
  };

  if (mode === "resume" || mode === "fine_tune") {
    payload.resume_from = activeCheckpoint?.path || "";
    payload.resume_mode = mode === "resume" ? "strict" : "fine-tune";
    payload.resume_episodes_mode = "incremental";
  }

  // If Logic Autopilot is enabled, mark this job so autopilot triggers after it completes.
  if (state.autopilotSettings?.logic_enabled) {
    payload.autopilot_after_completion = true;
  }

  const job = await apiRequest("/jobs/train", {
    method: "POST",
    body: JSON.stringify(payload),
  });
  if ($("campaignEnabledToggle")?.checked) {
    const runId = runLabelFromPath(job.run_dir);
    const maxVariations = Number($("campaignMaxVariationsInput").value) || 5;
    if (runId && runId !== "-") {
      await apiRequest("/campaigns", {
        method: "POST",
        body: JSON.stringify({ run_id: runId, max_variations: maxVariations }),
      });
      await refreshCampaigns();
    }
  }
  state.activeProgressJobId = job.id;
  state.trainingProgress = null;
  showMessage(`Queued ${job.job_type} job #${job.id}.`);
  await refreshJobs();
}

async function queueRobustnessJob(event) {
  event.preventDefault();
  const runId = $("robustnessRunSelect").value;
  const selectedRun = state.runs.find((item) => item.id === runId);
  if (!selectedRun) {
    showMessage("Select a run for robustness evaluation.", "error");
    return;
  }

  const job = await apiRequest("/jobs/evaluate", {
    method: "POST",
    body: JSON.stringify({
      job_type: "robustness",
      run_dir: selectedRun.path,
    }),
  });
  showMessage(`Queued robustness job #${job.id}.`);
  await refreshJobs();
}

async function queueRulesetSearchJob(event) {
  event.preventDefault();
  if (!state.activeGameConfigId) {
    showMessage("Select a game config first.", "error");
    return;
  }
  const payload = {
    game_config_path: selectedGameConfig()?.path || "",
    initial_games: Number($("searchInitialGamesInput").value) || 10,
    max_rulesets: Number($("searchMaxRulesetsInput").value) || 50,
    retrain_episodes: Number($("searchRetrainEpisodesInput").value) || 500,
    promising_target: Number($("searchPromisingTargetInput").value) || 0.7,
    leniency: Number($("searchLeniencyInput").value) || 0.05,
  };

  try {
    const job = await apiRequest("/jobs/ruleset-search", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    state.activeProgressJobId = job.id;
    showMessage(`Queued ruleset_search job #${job.id}.`);
    await refreshJobs();
  } catch (err) {
    showMessage(err.message, "error");
  }
}

async function createPlaySession(event) {
  event.preventDefault();
  if (!state.activeGameConfigId) {
    showMessage("Select a game config first.", "error");
    return;
  }
  const seats = state.playSeatDrafts
    .map((draft, index) => playSeatPayload(draft, index))
    .filter(Boolean);
  if (!seats.length) {
    showMessage("Add at least one seat.", "error");
    return;
  }
  if (seats.length > 4) {
    showMessage("Shared play supports at most 4 seats.", "error");
    return;
  }
  if (seats.filter((seat) => seat.type === "human").length > 1) {
    showMessage("Shared play supports at most one human seat.", "error");
    return;
  }
  if (seats.some((seat) => seat.type === "model") && !state.activeCheckpointId) {
    showMessage("Select an active brain before adding Active Brain seats.", "error");
    return;
  }
  state.playSession = await apiRequest("/play/session", {
    method: "POST",
    body: JSON.stringify({
      mode: "shared",
      game_config_id: state.activeGameConfigId,
      base_seed: Number($("playSeedInput").value),
      seats,
    }),
  }, 120000);
  renderPlaySession();
  showMessage("Play session started.");
}

async function advancePlayRound(humanAction = null) {
  if (!state.playSession?.id) {
    showMessage("Start a play session first.", "error");
    return;
  }
  showPlayDiceOverlay();
  $("playDiceCard")?.querySelector(".dice-animation-slot")?.classList.add("is-rolling");
  const humanSeat = state.playSession.seats?.find((seat) => seat.controller.type === "human" && !seat.done);
  const payload = humanAction === null
    ? {}
    : { human_actions: { [humanSeat?.id || "seat_1"]: Number(humanAction) } };
  state.playSession = await apiRequest(`/play/session/${encodeURIComponent(state.playSession.id)}/action`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
  renderPlaySession();
  const latestDice = latestPlayTurn()?.dice;
  if (latestDice) {
    rollPlayDice(latestDice).catch(() => {});
  }
}

async function runDirectEvaluation(event) {
  event.preventDefault();
  if (!state.activeCheckpointId || !state.activeGameConfigId) {
    showMessage("Select an active checkpoint and game config first.", "error");
    return;
  }
  state.directEvaluation = await apiRequest("/testing/evaluate", {
    method: "POST",
    body: JSON.stringify({
      checkpoint_id: state.activeCheckpointId,
      game_config_id: state.activeGameConfigId,
      seeds: parseSeedList($("testingSeedsInput").value),
    }),
  }, 120000);
  renderDirectEvaluation();
}

async function runCheckpointComparison(event) {
  event.preventDefault();
  const rightCheckpointId = $("compareCheckpointSelect").value;
  if (!state.activeCheckpointId || !rightCheckpointId || !state.activeGameConfigId) {
    showMessage("Select an active brain, a comparison brain, and a blueprint first.", "error");
    return;
  }
  state.comparisonEvaluation = await apiRequest("/testing/compare", {
    method: "POST",
    body: JSON.stringify({
      left_checkpoint_id: state.activeCheckpointId,
      right_checkpoint_id: rightCheckpointId,
      game_config_id: state.activeGameConfigId,
      seeds: parseSeedList($("compareSeedsInput").value),
    }),
  }, 120000);
  renderCheckpointComparison();
}

function exportDirectEvaluationJson() {
  if (!state.directEvaluation) {
    showMessage("Run a direct evaluation first.", "error");
    return;
  }
  downloadJsonFile("direct_evaluation.json", state.directEvaluation);
  showMessage("Exported direct evaluation JSON.");
}

function exportDirectEvaluationCsv() {
  if (!state.directEvaluation) {
    showMessage("Run a direct evaluation first.", "error");
    return;
  }
  const rows = state.directEvaluation.results || [];
  downloadCsvFile(
    "direct_evaluation.csv",
    ["seed", "episode_reward", "ending_money", "loan_turns", "loans_taken", "invalid_action_count", "terminal_reason"],
    rows.map((row) => [
      row.seed,
      row.episode_reward,
      row.ending_money,
      row.loan_turns,
      row.loans_taken,
      row.invalid_action_count,
      row.terminal_reason || "completed",
    ])
  );
  showMessage("Exported direct evaluation CSV.");
}

function exportComparisonJson() {
  if (!state.comparisonEvaluation) {
    showMessage("Run a checkpoint comparison first.", "error");
    return;
  }
  downloadJsonFile("checkpoint_comparison.json", state.comparisonEvaluation);
  showMessage("Exported checkpoint comparison JSON.");
}

function exportComparisonCsv() {
  if (!state.comparisonEvaluation) {
    showMessage("Run a checkpoint comparison first.", "error");
    return;
  }
  const leftRows = state.comparisonEvaluation.left?.results || [];
  const rightRows = state.comparisonEvaluation.right?.results || [];
  downloadCsvFile(
    "checkpoint_comparison.csv",
    ["seed", "left_reward", "right_reward", "reward_delta", "left_bank", "right_bank", "bank_delta", "left_terminal", "right_terminal"],
    leftRows.map((leftRow, index) => {
      const rightRow = rightRows[index] || {};
      return [
        leftRow.seed,
        leftRow.episode_reward,
        rightRow.episode_reward,
        Number(leftRow.episode_reward || 0) - Number(rightRow.episode_reward || 0),
        leftRow.ending_money,
        rightRow.ending_money,
        Number(leftRow.ending_money || 0) - Number(rightRow.ending_money || 0),
        leftRow.terminal_reason || "completed",
        rightRow.terminal_reason || "completed",
      ];
    })
  );
  showMessage("Exported checkpoint comparison CSV.");
}

const AUTO_CONNECT_URLS = [
  "http://127.0.0.1:8000",
];

function _showConnectedUi() {
  const manual = $("backendManualConnect");
  const actions = $("backendConnectedActions");
  if (manual) manual.style.display = "none";
  if (actions) actions.style.display = "";
}

function _showManualConnectUi() {
  const manual = $("backendManualConnect");
  const actions = $("backendConnectedActions");
  if (manual) manual.style.display = "";
  if (actions) actions.style.display = "none";
}

async function _tryConnect(url) {
  state.apiBaseUrl = url.replace(/\/$/, "");
  await refreshAll();
}

async function autoConnect() {
  for (const url of AUTO_CONNECT_URLS) {
    try {
      await _tryConnect(url);
      _showConnectedUi();
      showMessage(`Connected to ${url}`);
      return;
    } catch (_err) {
      // try next
    }
  }
  // All candidates failed. Show manual fields so user can enter a custom URL.
  _showManualConnectUi();
  state.health = null;
  updateStatusCard();
  showMessage("Could not auto-connect. Enter the backend URL manually.", "error");
}

function attachEvents() {
  document.querySelectorAll(".nav-button").forEach((button) => {
    button.addEventListener("click", () => setPage(button.dataset.page));
  });

  $("connectButton").addEventListener("click", async () => {
    const inputUrl = $("apiBaseUrlInput").value.trim().replace(/\/$/, "");
    state.apiBaseUrl = inputUrl;
    try {
      await refreshAll();
      _showConnectedUi();
      showMessage(`Connected to ${inputUrl}`);
    } catch (error) {
      state.health = null;
      updateStatusCard();
      showMessage(error.message, "error");
    }
  });

  $("refreshAllButton").addEventListener("click", async () => {
    try {
      await refreshAll();
      showMessage("Refreshed backend data.");
    } catch (error) {
      showMessage(error.message, "error");
    }
  });

  $("switchBackendButton").addEventListener("click", () => {
    const input = $("apiBaseUrlInput");
    if (input) input.value = state.apiBaseUrl;
    _showManualConnectUi();
  });

  $("refreshJobsButton").addEventListener("click", async () => {
    try {
      await refreshJobs();
      showMessage("Refreshed jobs.");
    } catch (error) {
      showMessage(error.message, "error");
    }
  });

  $("runCompatibilityButton").addEventListener("click", async () => {
    try {
      await runCompatibility();
      showMessage("Compatibility check completed.");
    } catch (error) {
      showMessage(error.message, "error");
    }
  });

  $("activeGameConfigSelect").addEventListener("change", (event) => {
    state.activeGameConfigId = event.target.value;
    updateSummaryPills();
    renderTrainingSelectionSummary();
    clearCompatibilityResult();
    clearEvaluationResults();
    refreshTrainingPreflight().catch(() => {});
    loadActiveGameConfigIntoEditor().catch(() => {});
  });

  $("activeTrainingConfigSelect").addEventListener("change", (event) => {
    state.activeTrainingConfigId = event.target.value;
    renderTrainingSelectionSummary();
    loadActiveTrainingConfigIntoEditor().catch(() => {});
  });

  $("activeCheckpointSelect").addEventListener("change", (event) => {
    state.activeCheckpointId = event.target.value;
    updateSummaryPills();
    renderTrainingSelectionSummary();
    clearCompatibilityResult();
    clearEvaluationResults();
    refreshTrainingPreflight().catch(() => {});
    renderCheckpointDetail();
  });

  $("activeCheckpointIncludeAllToggle").addEventListener("change", (event) => {
    state.includeCheckpointSelections = Boolean(event.target.checked);
    syncSelectors();
    updateSummaryPills();
    renderTrainingSelectionSummary();
    clearCompatibilityResult();
    clearEvaluationResults();
    refreshTrainingPreflight().catch(() => {});
    renderCheckpointDetail();
  });

  $("trainModeSelect").addEventListener("change", () => {
    renderTrainingSelectionSummary();
    refreshTrainingPreflight().catch(() => {});
  });

  $("campaignEnabledToggle").addEventListener("change", (event) => {
    $("campaignMaxVarField").style.display = event.target.checked ? "" : "none";
  });

  $("trainJobForm").addEventListener("submit", async (event) => {
    try {
      await queueTrainingJob(event);
    } catch (error) {
      showMessage(error.message, "error");
    }
  });

  $("robustnessJobForm").addEventListener("submit", async (event) => {
    try {
      await queueRobustnessJob(event);
    } catch (error) {
      showMessage(error.message, "error");
    }
  });

  $("queueRulesetSearchButton")?.addEventListener("click", async (event) => {
    try {
      await queueRulesetSearchJob(event);
    } catch (error) {
      showMessage(error.message, "error");
    }
  });

  $("stopCampaignButton").addEventListener("click", async () => {
    if (!state.activeCampaignId) return;
    try {
      await apiRequest(`/campaigns/${encodeURIComponent(state.activeCampaignId)}/stop`, { method: "POST" });
      await refreshCampaigns();
      showMessage("Campaign stopped.");
    } catch (error) {
      showMessage(error.message, "error");
    }
  });

  $("escalateCampaignButton").addEventListener("click", async () => {
    if (!state.activeCampaignId) return;
    try {
      await apiRequest(`/campaigns/${encodeURIComponent(state.activeCampaignId)}/escalate`, { method: "POST" });
      await refreshCampaigns();
      showMessage("Campaign escalation queued.");
    } catch (error) {
      showMessage(error.message, "error");
    }
  });

  $("directEvaluationForm").addEventListener("submit", async (event) => {
    try {
      await runDirectEvaluation(event);
      showMessage("Direct evaluation completed.");
    } catch (error) {
      showMessage(error.message, "error");
    }
  });

  $("checkpointCompareForm").addEventListener("submit", async (event) => {
    try {
      await runCheckpointComparison(event);
      showMessage("Checkpoint comparison completed.");
    } catch (error) {
      showMessage(error.message, "error");
    }
  });

  $("exportDirectEvaluationJsonButton").addEventListener("click", exportDirectEvaluationJson);
  $("exportDirectEvaluationCsvButton").addEventListener("click", exportDirectEvaluationCsv);
  $("exportComparisonJsonButton").addEventListener("click", exportComparisonJson);
  $("exportComparisonCsvButton").addEventListener("click", exportComparisonCsv);

  $("playSessionForm").addEventListener("submit", async (event) => {
    try {
      await createPlaySession(event);
    } catch (error) {
      showMessage(error.message, "error");
    }
  });

  $("addPlaySeatButton").addEventListener("click", () => {
    if (state.playSeatDrafts.length >= 4) {
      showMessage("Shared play supports at most 4 seats.", "error");
      return;
    }
    const nextIndex = state.playSeatDrafts.length + 1;
    state.playSeatDrafts.push({
      id: `draft_${Date.now()}`,
      type: "random",
      display_name: `Random AI ${nextIndex}`,
    });
    renderPlaySeatEditor();
  });

  $("refreshPlayButton").addEventListener("click", async () => {
    try {
      await refreshPlaySession();
      showMessage("Refreshed play session.");
    } catch (error) {
      showMessage(error.message, "error");
    }
  });

  $("advancePlayRoundButton").addEventListener("click", async () => {
    try {
      await advancePlayRound();
      showMessage("Advanced play round.");
    } catch (error) {
      showMessage(error.message, "error");
    }
  });

  $("closePlayDiceOverlayButton").addEventListener("click", hidePlayDiceOverlay);

  $("submitHumanActionButton").addEventListener("click", async () => {
    try {
      await advancePlayRound($("playHumanActionSelect").value);
      showMessage("Submitted human action.");
    } catch (error) {
      showMessage(error.message, "error");
    }
  });

  $("loadActiveGameConfigButton").addEventListener("click", async () => {
    try {
      await loadActiveGameConfigIntoEditor();
      await validateGameConfigDraft(false).catch(() => {});
      showMessage("Loaded active game config into editor.");
    } catch (error) {
      showMessage(error.message, "error");
    }
  });

  $("loadActiveTrainingConfigButton").addEventListener("click", async () => {
    try {
      await loadActiveTrainingConfigIntoEditor();
      await validateTrainingConfigDraft(false).catch(() => {});
      showMessage("Loaded active training config into editor.");
    } catch (error) {
      showMessage(error.message, "error");
    }
  });

  $("validateGameConfigButton").addEventListener("click", async () => {
    try {
      await validateGameConfigDraft(true);
    } catch (_error) {
      // message already shown
    }
  });

  $("validateTrainingConfigButton").addEventListener("click", async () => {
    try {
      await validateTrainingConfigDraft(true);
    } catch (_error) {
      // message already shown
    }
  });

  $("cloneGameConfigButton").addEventListener("click", async () => {
    try {
      await loadActiveGameConfigIntoEditor();
      $("gameConfigFileNameInput").value = `${state.activeGameConfigId}_copy`;
      showMessage("Game config cloned into the editor. Save As New to create the copy.");
    } catch (error) {
      showMessage(error.message, "error");
    }
  });

  $("cloneTrainingConfigButton").addEventListener("click", async () => {
    try {
      await loadActiveTrainingConfigIntoEditor();
      $("trainingConfigFileNameInput").value = `${state.activeTrainingConfigId}_copy`;
      showMessage("Training config cloned into the editor. Save As New to create the copy.");
    } catch (error) {
      showMessage(error.message, "error");
    }
  });

  $("deleteGameConfigButton").addEventListener("click", async () => {
    if (!state.activeGameConfigId || state.activeGameConfigId === "default_game_config") {
      showMessage("Only custom game configs can be deleted.", "error");
      return;
    }
    if (!window.confirm(`Delete game config ${state.activeGameConfigId}?`)) return;
    try {
      await apiRequest(`/configs/game/${encodeURIComponent(state.activeGameConfigId)}`, { method: "DELETE" });
      state.activeGameConfigId = "default_game_config";
      await refreshAll();
      await loadActiveGameConfigIntoEditor();
      showMessage("Game config deleted.");
    } catch (error) {
      showMessage(error.message, "error");
    }
  });

  $("deleteTrainingConfigButton").addEventListener("click", async () => {
    if (!state.activeTrainingConfigId || state.activeTrainingConfigId === "default_training_config") {
      showMessage("Only custom training configs can be deleted.", "error");
      return;
    }
    if (!window.confirm(`Delete training config ${state.activeTrainingConfigId}?`)) return;
    try {
      await apiRequest(`/configs/training/${encodeURIComponent(state.activeTrainingConfigId)}`, { method: "DELETE" });
      state.activeTrainingConfigId = "default_training_config";
      await refreshAll();
      await loadActiveTrainingConfigIntoEditor();
      showMessage("Training config deleted.");
    } catch (error) {
      showMessage(error.message, "error");
    }
  });

  $("exportGameConfigButton").addEventListener("click", async () => {
    try {
      const payload = await apiRequest(`/configs/game/${encodeURIComponent(state.activeGameConfigId)}`);
      downloadJsonFile(`${payload.id || "game_config"}.json`, payload.config);
      showMessage("Exported game config JSON.");
    } catch (error) {
      showMessage(error.message, "error");
    }
  });

  $("exportTrainingConfigButton").addEventListener("click", async () => {
    try {
      const payload = await apiRequest(`/configs/training/${encodeURIComponent(state.activeTrainingConfigId)}`);
      downloadJsonFile(`${payload.id || "training_config"}.json`, payload.config);
      showMessage("Exported training config JSON.");
    } catch (error) {
      showMessage(error.message, "error");
    }
  });

  $("importGameConfigButton").addEventListener("click", () => $("importGameConfigInput").click());
  $("importTrainingConfigButton").addEventListener("click", () => $("importTrainingConfigInput").click());

  $("importGameConfigInput").addEventListener("change", async (event) => {
    const file = event.target.files?.[0];
    if (!file) return;
    try {
      const text = await file.text();
      const payload = JSON.parse(text);
      state.visualGameConfig = clone(payload);
      $("gameConfigEditor").value = formatJson(payload);
      $("gameConfigFileNameInput").value = file.name.replace(/\.json$/i, "");
      renderVisualEditor();
      await validateGameConfigDraft(false).catch(() => {});
      showMessage(`Imported ${file.name} into the game config editor.`);
    } catch (error) {
      showMessage(error.message, "error");
    } finally {
      event.target.value = "";
    }
  });

  $("importTrainingConfigInput").addEventListener("change", async (event) => {
    const file = event.target.files?.[0];
    if (!file) return;
    try {
      const text = await file.text();
      const payload = JSON.parse(text);
      $("trainingConfigEditor").value = formatJson(payload);
      $("trainingConfigFileNameInput").value = file.name.replace(/\.json$/i, "");
      await validateTrainingConfigDraft(false).catch(() => {});
      showMessage(`Imported ${file.name} into the training config editor.`);
    } catch (error) {
      showMessage(error.message, "error");
    } finally {
      event.target.value = "";
    }
  });

  $("saveGameConfigButton").addEventListener("click", async () => {
    try {
      await saveGameConfig(false);
    } catch (error) {
      showMessage(error.message, "error");
    }
  });

  $("overwriteGameConfigButton").addEventListener("click", async () => {
    try {
      await saveGameConfig(true);
    } catch (error) {
      showMessage(error.message, "error");
    }
  });

  $("saveTrainingConfigButton").addEventListener("click", async () => {
    try {
      await saveTrainingConfig(false);
    } catch (error) {
      showMessage(error.message, "error");
    }
  });

  $("overwriteTrainingConfigButton").addEventListener("click", async () => {
    try {
      await saveTrainingConfig(true);
    } catch (error) {
      showMessage(error.message, "error");
    }
  });

  document.addEventListener("input", (event) => {
    if (
      event.target.matches(
        "#configNameInput, #schemaVersionInput, #configDescriptionInput, #playersCountInput, #productsCountInput, #sprintsPerProductInput, #maxTurnsInput, #startingMoneyInput, #ringValueInput, #costContinueInput, #costSwitchMidInput, #costSwitchAfterInput, #mandatoryLoanInput, #loanInterestInput, #penaltyNegativeInput, #penaltyPositiveInput, #dailyScrumsInput, #dailyScrumTargetInput, #productNamesGrid input, #boardMatrixContainer input, #diceRulesList input, #refinementRulesList input, #incidentCardsList input, #incidentCardsList textarea, #refinementModelInput, #refinementDieSidesInput, #incidentDrawProbabilityInput, #incidentSeverityMultiplierInput"
      )
    ) {
      if (event.target.id === "productsCountInput" || event.target.id === "sprintsPerProductInput") {
        syncVisualShapeFromInputs();
        renderVisualEditor();
        return;
      }
      syncGameJsonEditorFromVisual();
    }
  });

  document.addEventListener("change", (event) => {
    if (
      event.target.matches(
        "#refinementActiveInput, #incidentActiveInput, #playerSpecificIncidentsInput, #incidentFutureOnly_0, #incidentCardsList input[type='checkbox']"
      )
    ) {
      syncGameJsonEditorFromVisual();
    }
  });

  document.addEventListener("click", (event) => {
    const removeDiceIndex = event.target.getAttribute("data-remove-dice");
    if (removeDiceIndex !== null) {
      state.visualGameConfig.dice_rules.splice(Number(removeDiceIndex), 1);
      renderVisualEditor();
    }

    const removeIncidentIndex = event.target.getAttribute("data-remove-incident");
    if (removeIncidentIndex !== null) {
      state.visualGameConfig.incident.cards.splice(Number(removeIncidentIndex), 1);
      renderVisualEditor();
    }
  });

  $("addDiceRuleButton").addEventListener("click", () => {
    ensureVisualGameConfig();
    state.visualGameConfig.dice_rules.push({
      min_features: 1,
      max_features: null,
      dice_count: 1,
      dice_sides: 6,
    });
    renderVisualEditor();
  });

  $("resetRefinementRulesButton").addEventListener("click", () => {
    readVisualEditorIntoState();
    rebuildVisualRefinementRules();
    renderVisualEditor();
  });

  $("addIncidentCardButton").addEventListener("click", () => {
    ensureVisualGameConfig();
    const sprintCount = state.visualGameConfig.board_ring_values[0]?.length || 1;
    state.visualGameConfig.incident.cards.push({
      card_id: Date.now(),
      name: "Custom Incident",
      description: "",
      effect_type: "adjust_future_products",
      target_products: [],
      delta_money: 0,
      target_sprint: Math.min(1, sprintCount),
      set_value_money: null,
      future_only: true,
      weight: 1.0,
    });
    renderVisualEditor();
  });

  $("visualEditorLoadButton").addEventListener("click", async () => {
    try {
      await loadActiveGameConfigIntoEditor();
      showMessage("Loaded active game config into visual editor.");
    } catch (error) {
      showMessage(error.message, "error");
    }
  });

  $("visualEditorResetButton").addEventListener("click", () => {
    state.visualGameConfig = clone(DEFAULT_GAME_CONFIG);
    renderVisualEditor();
    showMessage("Reset visual editor to defaults.");
  });
}

// ── Login form handler ───────────────────────────────────────────────────────
document.getElementById("loginForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const errorEl = document.getElementById("loginError");
  const submitBtn = document.getElementById("loginSubmitButton");
  const username = document.getElementById("loginUsername").value.trim();
  const password = document.getElementById("loginPassword").value;

  errorEl.textContent = "";
  submitBtn.disabled = true;
  submitBtn.textContent = "Signing in…";

  try {
    const resp = await fetch(`${state.apiBaseUrl}/auth/login`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, password }),
    });

    if (!resp.ok) {
      const data = await resp.json().catch(() => ({}));
      errorEl.textContent = data.detail || "Login failed. Check your credentials.";
      return;
    }

    const data = await resp.json();
    setToken(data.access_token);
    hideLoginScreen();
    startProgressPolling();
    autoConnect();
  } catch {
    errorEl.textContent = "Could not reach the server. Try again.";
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = "Sign in";
  }
});
// ─────────────────────────────────────────────────────────────────────────────

attachEvents();
setPage("rules");
state.visualGameConfig = clone(DEFAULT_GAME_CONFIG);
renderVisualEditor();
renderTrainingSelectionSummary();
renderTrainingProgress();
renderCampaignPanel();
renderPlaySeatEditor();

// Show login screen or boot the app depending on whether a token exists
if (getToken()) {
  hideLoginScreen();
  startProgressPolling();
  autoConnect();
} else {
  showLoginScreen();
}
