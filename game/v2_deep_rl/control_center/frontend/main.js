// ── Entry point — imports all modules and wires events ────────────────────────
import { getToken, setToken, getRole, setRole, isGuest, apiRequest } from './api/client.js';
import { showLoginScreen, hideLoginScreen, logout } from './components/auth.js';
import { state } from './state/store.js';
import { DEFAULT_GAME_CONFIG } from './constants/defaults.js';
import { $, clone, showMessage, clearMessage, buildOptions, runLabelFromPath, downloadJsonFile, jobForRunId } from './utils/helpers.js';
import { formatJson, sidebarCheckpointOptions as getSidebarCheckpointOptions } from './utils/formatting.js';
import { setPage, getSavedPage, updateStatusCard, updateSummaryPills, renderContextCard } from './components/navigation.js';
import { autoConnect, _showConnectedUi, _showManualConnectUi, startProgressPolling } from './components/connection.js';
import { renderVisualEditor, ensureVisualGameConfig, syncVisualShapeFromInputs, syncGameJsonEditorFromVisual, rebuildVisualRefinementRules, readVisualEditorIntoState } from './components/configs/visualEditor.js';
import { renderGameConfigs, renderGameConfigValidation, loadActiveGameConfigIntoEditor, validateGameConfigDraft, saveGameConfig } from './components/configs/gameConfig.js';
import { renderTrainingConfigs, renderTrainingConfigValidation, loadActiveTrainingConfigIntoEditor, validateTrainingConfigDraft, saveTrainingConfig } from './components/configs/trainingConfig.js';
import { renderJobs, renderJobDetail, renderJobLog, renderTrainingSelectionSummary, renderTrainingPreflight, refreshJobs, fetchJobDetail, refreshTrainingPreflight, queueTrainingJob, queueRobustnessJob } from './components/training/jobs.js';
import { renderRuns, renderTrainingProgress, fetchTrainingProgress, fetchRunProgress, fetchRunDetail } from './components/training/progress.js';
import { fetchAutopilotData, renderAutopilotPanel, renderAutopilotTrainingPanel, refreshAutopilotSettings } from './components/training/autopilot.js';
import { renderCampaignPanel, refreshCampaigns } from './components/training/campaigns.js';
import { renderCompatibility, renderCheckpointDetail, refreshCheckpoints, runCompatibility, clearCompatibilityResult } from './components/evaluation/checkpoints.js';
import { renderDirectEvaluation, runDirectEvaluation, exportDirectEvaluationJson, exportDirectEvaluationCsv } from './components/evaluation/directEval.js';
import { renderCheckpointComparison, runCheckpointComparison, exportComparisonJson, exportComparisonCsv } from './components/evaluation/comparison.js';
import { renderPlaySession, renderPlayBoard, renderPlaySeatEditor } from './components/play/board.js';
import { createPlaySession, advancePlayRound, refreshPlaySession, latestPlayTurn } from './components/play/session.js';
import { hidePlayDiceOverlay, rollPlayDice } from './components/play/dice.js';

// ── Theme switcher ────────────────────────────────────────────────────────────

const _THEME_KEY = "cc_theme";
const _THEMES = ["dark", "light"];

function setTheme(name) {
  if (!_THEMES.includes(name)) name = "dark";
  document.body.dataset.theme = name;
  localStorage.setItem(_THEME_KEY, name);
  document.querySelectorAll("[data-theme-btn]").forEach(btn => {
    btn.classList.toggle("is-active", btn.dataset.themeBtn === name);
  });
}

function initTheme() {
  const saved = localStorage.getItem(_THEME_KEY) || "dark";
  setTheme(saved);
}

// ── Orchestration helpers (live here to avoid circular deps) ─────────────────

function syncSelectors() {
  buildOptions("activeGameConfigSelect", state.gameConfigs);
  buildOptions("activeTrainingConfigSelect", state.trainingConfigs);
  const checkpointOptions = getSidebarCheckpointOptions();
  buildOptions("activeCheckpointSelect", checkpointOptions, "id", "ui_label");
  buildOptions("modelASelect", checkpointOptions, "id", "ui_label");
  buildOptions("modelBSelect", checkpointOptions, "id", "ui_label");

  const modelASelect = $("modelASelect");
  const modelBSelect = $("modelBSelect");
  if (modelASelect && modelASelect.options.length && !modelASelect.querySelector("option[value='']")) {
    modelASelect.insertBefore(new Option("Select Model A", "", false, false), modelASelect.firstChild);
  }
  if (modelBSelect && modelBSelect.options.length && !modelBSelect.querySelector("option[value='']")) {
    modelBSelect.insertBefore(new Option("Select Model B", "", false, false), modelBSelect.firstChild);
  }

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

async function _refreshCheckpoints() {
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

async function _refreshAll() {
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

  // Load checkpoints in the background (slow on first call).
  _refreshCheckpoints();

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

  // Re-apply guest restrictions after every render cycle so dynamically
  // re-rendered panels don't accidentally re-enable write controls.
  applyGuestRestrictions();
}

// renderRunDetail lives here because it calls openInspectForRun (circular if in progress.js)
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
  const bestCheckpoint = state.checkpoints.find((c) =>
    c.path === (metadata.best_checkpoint_path || state.runs.find((item) => item.id === state.runDetail.id)?.best_checkpoint_path)
  );
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
        <div class="rating-grade" style="background:${color};">${rating.grade}</div>
        <div>
          <strong>${String(rating.score)}/100</strong>
          <div class="checkpoint-subtitle">bankruptcies ${brPct} | avg reward ${reward}</div>
        </div>
      </div>`;
  })();

  container.className = "list-card";
  container.innerHTML = `
    <h4>${state.runDetail.label}</h4>
    <div class="card-meta">
      <span class="tag">${metadata.created_at || "-"}</span>
      <span class="tag">${metadata.resume_mode || "new"}</span>
    </div>
    <div class="card-meta">
      <span class="tag">avg reward ${String(metrics.average_reward_per_episode ?? "-")}</span>
      <span class="tag">bankruptcy ${String(metrics.bankruptcy_rate ?? "-")}</span>
      <span class="tag">${checkpoints.length} checkpoints</span>
    </div>
    ${ratingHtml}
    ${metadata.run_notes ? `<p>${metadata.run_notes}</p>` : ""}
    <div class="inline-actions">
      ${bestCheckpoint ? `<button class="button secondary use-run-detail-best-button" type="button" ${isGuest() ? 'disabled title="Guests cannot perform this action"' : ""}>Use Best Brain</button>` : ""}
      <button class="button secondary open-run-inspect-button" type="button">Open Inspect</button>
      <button class="button secondary queue-run-robustness-button" type="button" ${isGuest() ? 'disabled title="Guests cannot perform this action"' : ""}>Queue Robustness</button>
    </div>
  `;

  container.querySelector(".use-run-detail-best-button")?.addEventListener("click", () => {
    state.activeCheckpointId = bestCheckpoint.id;
    $("activeCheckpointSelect").value = bestCheckpoint.id;
    updateSummaryPills();
    clearCompatibilityResult();
    _clearEvaluationResults();
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
    if (isGuest()) return;
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

function _clearEvaluationResults() {
  state.directEvaluation = null;
  state.comparisonEvaluation = null;
  renderDirectEvaluation();
  renderCheckpointComparison();
}

// ── Guest mode — hide or disable write actions for read-only users ───────────

// Buttons that should be HIDDEN entirely for guests
// (destructive or additive-only — no read context).
const _GUEST_HIDE_IDS = [
  "addDiceRuleButton",
  "addIncidentCardButton",
  "resetRefinementRulesButton",
  "deleteGameConfigButton",
  "deleteTrainingConfigButton",
  "cloneGameConfigButton",
  "cloneTrainingConfigButton",
  "importGameConfigButton",
  "importTrainingConfigButton",
  "visualEditorResetButton",
];

// Buttons / forms that should remain VISIBLE but clearly disabled
// (guests can see there is an action here, just not allowed to use it).
const _GUEST_DISABLE_IDS = [
  // Design tab — config saves
  "saveGameConfigButton",
  "overwriteGameConfigButton",
  "saveTrainingConfigButton",
  "overwriteTrainingConfigButton",
  // Train tab — job queue & campaign controls
  "trainJobForm",
  "robustnessJobForm",
  "stopCampaignButton",
  "escalateCampaignButton",
  // Evaluate tab — run jobs
  "directEvaluationForm",
  "checkpointCompareForm",
];

function applyGuestRestrictions() {
  if (!isGuest()) return;

  // Drive all CSS-based restrictions through a single body class.
  document.body.classList.add("guest-mode");

  // Show a "Guest View" badge in the header so it's clear to users that they're in a restricted mode
  if (!document.getElementById("guestModeBadge")) {
    const badge = document.createElement("span");
    badge.id = "guestModeBadge";
    badge.textContent = "Guest View";
    badge.style.cssText = [
      "display:inline-flex",
      "align-items:center",
      "padding:4px 10px",
      "border-radius:999px",
      "background:#f59e0b",
      "color:#fff",
      "margin-left:0.5rem",
      "font-size:0.75rem",
      "font-weight:600",
      "letter-spacing:0.03em",
      "pointer-events:none",
      "white-space:nowrap",
      "flex-shrink:0",
    ].join(";");
    const header = document.querySelector("header") || document.body;
    header.appendChild(badge);
  }

  // Hide add/delete/destructive buttons completely.
  _GUEST_HIDE_IDS.forEach((id) => {
    const el = document.getElementById(id);
    if (el) el.style.display = "none";
  });

  // Disable (but keep visible) save/action buttons.
  // CSS (body.guest-mode .button:disabled) provides the visual treatment.
  _GUEST_DISABLE_IDS.forEach((id) => {
    const el = document.getElementById(id);
    if (!el) return;
    if (el.tagName === "FORM") {
      el.querySelectorAll('button[type="submit"], input[type="submit"]').forEach((btn) => {
        btn.disabled = true;
        btn.title = "Guests cannot perform this action";
      });
    } else {
      el.disabled = true;
      el.title = "Guests cannot perform this action";
    }
  });

  // Make all static Design-tab inputs explicitly readonly so keyboard users
  // can't edit them either. CSS pointer-events:none covers mouse interaction;
  // readonly / disabled covers keyboard access.
  const designPage = document.getElementById("page-rules");
  if (designPage) {
    designPage.querySelectorAll("input:not([type='checkbox']):not([type='file']), textarea").forEach((el) => {
      el.readOnly = true;
    });
    // selects can't be made readonly — disabled is the only option
    designPage.querySelectorAll("select").forEach((el) => {
      el.disabled = true;
    });
  }

  // Train tab — lock the job-parameter inputs (episodes, run name, notes).
  // The mode select is intentionally left alone so guests can preview options.
  ["trainEpisodesInput", "trainEvalEpisodesInput", "trainRunNameInput", "trainNotesInput"].forEach((id) => {
    const el = document.getElementById(id);
    if (el) {
      el.readOnly = true;
      el.style.pointerEvents = "none";
      el.style.cursor = "not-allowed";
      el.style.opacity = "0.6";
      el.style.background = "var(--panel-alt)";
    }
  });

  // NOTE: dynamically-rendered panels (autopilot toggles, job stop/dismiss,
  // run "Use Best Brain", etc.) are handled at render time inside their own
  // components — see autopilot.js, jobs.js, progress.js, and renderRunDetail.
}

// Undo all guest restrictions so a fresh login as admin gets a clean slate.
// Called at the top of the login handler and on logout.
function resetGuestRestrictions() {
  document.body.classList.remove("guest-mode");
  document.getElementById("guestModeBadge")?.remove();

  // Restore hidden buttons
  _GUEST_HIDE_IDS.forEach((id) => {
    const el = document.getElementById(id);
    if (el) el.style.display = "";
  });

  // Re-enable disabled buttons / form submits
  _GUEST_DISABLE_IDS.forEach((id) => {
    const el = document.getElementById(id);
    if (!el) return;
    if (el.tagName === "FORM") {
      el.querySelectorAll('button[type="submit"], input[type="submit"]').forEach((btn) => {
        btn.disabled = false;
        btn.title = "";
      });
    } else {
      el.disabled = false;
      el.title = "";
    }
  });

  // Restore Design-tab inputs
  const designPage = document.getElementById("page-rules");
  if (designPage) {
    designPage.querySelectorAll("input, textarea").forEach((el) => { el.readOnly = false; });
    designPage.querySelectorAll("select").forEach((el) => { el.disabled = false; });
  }

  // Restore Train-tab job inputs
  ["trainEpisodesInput", "trainEvalEpisodesInput", "trainRunNameInput", "trainNotesInput"].forEach((id) => {
    const el = document.getElementById(id);
    if (!el) return;
    el.readOnly = false;
    el.style.pointerEvents = "";
    el.style.cursor = "";
    el.style.opacity = "";
    el.style.background = "";
  });
}

async function openInspectForRun(runId, announce = true) {
  await fetchRunDetail(runId, false);
  renderRunDetail();
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
    renderRunDetail();
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

// ── Event listeners from progress.js / checkpoints.js (avoid circular deps) ──

document.addEventListener("openInspectForRun", async (event) => {
  try {
    await openInspectForRun(event.detail.runId, event.detail.announce);
  } catch (error) {
    showMessage(error.message, "error");
  }
});

document.addEventListener("openInspectForJob", async (event) => {
  try {
    await openInspectForJob(event.detail.jobId, event.detail.announce);
  } catch (error) {
    showMessage(error.message, "error");
  }
});

document.addEventListener("runDetailLoaded", () => {
  renderRunDetail();
});

document.addEventListener("checkpointsLoaded", () => {
  syncSelectors();
  renderCheckpointDetail();
  renderTrainingSelectionSummary();
  renderTrainingPreflight();
  renderCompatibility();
});

document.addEventListener("clearEvaluationResults", () => {
  renderCheckpointComparison();
});

document.addEventListener("activateCheckpoint", (event) => {
  const { checkpointId, runLabel } = event.detail;
  state.activeCheckpointId = checkpointId;
  $("activeCheckpointSelect").value = checkpointId;
  updateSummaryPills();
  clearCompatibilityResult();
  _clearEvaluationResults();
  renderContextCard();
  renderTrainingSelectionSummary();
  renderCheckpointDetail();
  showMessage(`Active brain set to the best checkpoint from ${runLabel}.`);
});

document.addEventListener("playSessionUpdated", () => {
  renderPlaySession();
});

document.addEventListener("showPlayDiceOverlay", () => {
  document.getElementById("playDiceOverlay")?.classList.remove("hidden");
});

document.addEventListener("rollPlayDiceForLatestTurn", () => {
  const latestDice = latestPlayTurn()?.dice;
  if (latestDice) {
    rollPlayDice(latestDice).catch(() => {});
  }
});

// ── Connection module override — call _refreshAll after connecting ────────────

document.addEventListener("backendConnected", async () => {
  try {
    await _refreshAll();
    startProgressPolling();
  } catch (error) {
    showMessage(error.message, "error");
  }
});

// ── Attach all DOM events ────────────────────────────────────────────────────

function attachEvents() {
  document.querySelectorAll(".nav-button").forEach((button) => {
    button.addEventListener("click", () => setPage(button.dataset.page));
  });

  $("connectButton").addEventListener("click", async () => {
    const inputUrl = $("apiBaseUrlInput").value.trim().replace(/\/$/, "");
    state.apiBaseUrl = inputUrl;
    try {
      await _refreshAll();
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
      await _refreshAll();
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

  $("logoutButton").addEventListener("click", () => {
    resetGuestRestrictions();
    logout();
  });

  document.querySelectorAll("[data-theme-btn]").forEach(btn => {
    btn.addEventListener("click", () => setTheme(btn.dataset.themeBtn));
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
    _clearEvaluationResults();
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
    _clearEvaluationResults();
    refreshTrainingPreflight().catch(() => {});
    renderCheckpointDetail();
  });

  $("activeCheckpointIncludeAllToggle").addEventListener("change", (event) => {
    state.includeCheckpointSelections = Boolean(event.target.checked);
    syncSelectors();
    updateSummaryPills();
    renderTrainingSelectionSummary();
    clearCompatibilityResult();
    _clearEvaluationResults();
    refreshTrainingPreflight().catch(() => {});
    renderCheckpointDetail();
  });

  $("modelASelect").addEventListener("change", () => {
    renderTrainingSelectionSummary();
    renderTrainingPreflight();
  });

  $("modelBSelect").addEventListener("change", () => {
    renderTrainingSelectionSummary();
    renderTrainingPreflight();
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
      await _refreshAll();
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
      await _refreshAll();
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
      await _refreshAll();
    } catch (error) {
      showMessage(error.message, "error");
    }
  });

  $("overwriteGameConfigButton").addEventListener("click", async () => {
    try {
      await saveGameConfig(true);
      await _refreshAll();
    } catch (error) {
      showMessage(error.message, "error");
    }
  });

  $("saveTrainingConfigButton").addEventListener("click", async () => {
    try {
      await saveTrainingConfig(false);
      await _refreshAll();
    } catch (error) {
      showMessage(error.message, "error");
    }
  });

  $("overwriteTrainingConfigButton").addEventListener("click", async () => {
    try {
      await saveTrainingConfig(true);
      await _refreshAll();
    } catch (error) {
      showMessage(error.message, "error");
    }
  });

  document.addEventListener("input", (event) => {
    if (isGuest()) return; // guests cannot modify the config
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
    if (isGuest()) return; // guests cannot modify the config
    if (
      event.target.matches(
        "#refinementActiveInput, #incidentActiveInput, #playerSpecificIncidentsInput, #incidentFutureOnly_0, #incidentCardsList input[type='checkbox']"
      )
    ) {
      syncGameJsonEditorFromVisual();
    }
  });

  document.addEventListener("click", (event) => {
    if (isGuest()) return; // guests cannot modify configs

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

// ── Login form handler ────────────────────────────────────────────────────────

document.getElementById("loginForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const errorEl = document.getElementById("loginError");
  const submitBtn = document.getElementById("loginSubmitButton");
  const username = document.getElementById("loginUsername").value.trim();
  const password = document.getElementById("loginPassword").value;

  errorEl.textContent = "";
  submitBtn.disabled = true;
  submitBtn.textContent = "Signing in…";

  // Wipe any previous session's guest restrictions before applying the new role.
  resetGuestRestrictions();

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
    setRole(data.role);
    state.userRole = data.role;
    hideLoginScreen();
    startProgressPolling();
    autoConnect().then(_refreshAll).catch(() => {});
  } catch {
    errorEl.textContent = "Could not reach the server. Try again.";
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = "Sign in";
  }
});

// ── Boot ──────────────────────────────────────────────────────────────────────

initTheme();
attachEvents();
setPage(getSavedPage());
state.visualGameConfig = clone(DEFAULT_GAME_CONFIG);
renderVisualEditor();
renderTrainingSelectionSummary();
renderTrainingProgress();
renderCampaignPanel();
renderPlaySeatEditor();

if (getToken()) {
  // Restore the role that was saved when the user last logged in.
  state.userRole = getRole();
  hideLoginScreen();
  startProgressPolling();
  autoConnect().then(_refreshAll).catch(() => {});
} else {
  showLoginScreen();
}
