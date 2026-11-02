/**
 * Universal AI/ML Studio — Client Dashboard Controller
 * Obsidian Telemetry Edition
 */

let state = {
  activeDatasetId: null,
  activeDatasetMeta: null,
  previewData: null,
  taskType: "classification",
  algorithms: { classification: [], regression: [] },
  models: [],
  selectedModel: null,
};

// ── Comparison constants (used by runModelComparison and helpers) ─────────────
let _lastComparisonData = null;
const _SKIP_METRICS = new Set(["confusion_matrix", "per_class", "class_labels", "mape_note",
                               "feature_importances", "feature_coef"]);
const _METRIC_LABELS = {
  accuracy: "Accuracy",          precision: "Precision (weighted)",
  recall: "Recall (weighted)",   f1: "F1 (weighted)",
  f1_macro: "F1 (macro)",        precision_macro: "Precision (macro)",
  recall_macro: "Recall (macro)", balanced_accuracy: "Balanced Accuracy",
  mcc: "MCC",                    roc_auc: "ROC-AUC",
  specificity: "Specificity",    sensitivity: "Sensitivity",
  r2: "R²",                      mae: "MAE",
  mse: "MSE",                    rmse: "RMSE",
  mape: "MAPE",
  cv_mean_accuracy: "CV Accuracy (mean)", cv_std_accuracy: "CV Accuracy (std)",
  cv_mean_r2: "CV R² (mean)",    cv_std_r2: "CV R² (std)",
  cv_mean_rmse: "CV RMSE (mean)",cv_std_rmse: "CV RMSE (std)",
};
const _LOWER_IS_BETTER = new Set(["mae","mse","rmse","mape","cv_mean_rmse","cv_std_rmse","cv_std_accuracy"]);

// ── Toast Notifications ─────────────────────────────────────────
function showToast(message, type = "info") {
  const container = document.getElementById("toast-container");
  if (!container) return;
  const toast = document.createElement("div");
  toast.className = `toast ${type}`;
  const icon = type === 'success' ? '✅' : type === 'error' ? '❌' : 'ℹ️';
  toast.innerHTML = `<span style="font-size:15px;">${icon}</span><span style="font-weight:500;">${message}</span>`;
  container.appendChild(toast);
  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transform = 'translateY(10px) scale(0.96)';
    toast.style.transition = 'all 0.25s cubic-bezier(0.4, 0, 0.2, 1)';
    setTimeout(() => toast.remove(), 250);
  }, 4000);
}

// ── Navigation ──────────────────────────────────────────────────
function showTab(tabName) {
  document.querySelectorAll(".nav-item").forEach(item => {
    item.classList.toggle("active", item.getAttribute("data-tab") === tabName);
  });
  document.querySelectorAll(".view-section").forEach(sec => {
    sec.classList.remove("active");
  });
  const activeSec = document.getElementById(`tab-${tabName}`);
  if (activeSec) activeSec.classList.add("active");

  const titles = {
    overview: "Platform Overview",
    datasets: "Datasets Management Hub",
    validation: "Automated Data Validation",
    eda: "Exploratory Data Analysis",
    preprocessing: "Preprocessing & Cleaning Studio",
    ml: "AutoML & Model Comparison",
    prediction: "Live Prediction Playground",
    reports: "Audit Reports & Artifacts",
    auditor: "Pipeline Auditor & Leakage Detection",
    repair: "Automated Pipeline Repair & Diagnostics",
    benchmark: "Auditor Benchmark Lab",
  };
  const heading = document.getElementById("page-heading");
  if (heading) heading.innerText = titles[tabName] || "Studio Dashboard";
}

document.querySelectorAll(".nav-item").forEach(item => {
  item.addEventListener("click", () => {
    const tab = item.getAttribute("data-tab");
    if (tab) showTab(tab);
  });
});

// ── Theme Management (Dark / Light Mode) ────────────────────────
function initTheme() {
  const saved = (typeof localStorage !== "undefined" && localStorage.getItem("studio_theme")) || "dark";
  applyTheme(saved);
}

function toggleTheme() {
  const current = document.documentElement.getAttribute("data-theme") || "dark";
  const target = current === "dark" ? "light" : "dark";
  applyTheme(target);
}

function applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  try {
    if (typeof localStorage !== "undefined") {
      localStorage.setItem("studio_theme", theme);
    }
  } catch (e) {}

  const labels = document.querySelectorAll(".theme-toggle-text, #theme-status-text");
  labels.forEach(el => {
    el.innerText = theme === "dark" ? "DARK" : "LIGHT";
  });

  // If Plotly charts exist on screen, update their grid colors
  if (typeof Plotly !== "undefined") {
    const isLight = theme === "light";
    const chartIds = [
      "viz-container-confusion", "viz-container-radar", "viz-container-cv",
      "viz-container-bar", "viz-container-feature", "viz-container-perclass",
      "viz-container-gauge", "viz-container-history", "compare-chart-bar",
      "compare-chart-heatmap", "compare-chart-scatter", "compare-chart-box",
      "compare-chart-radar"
    ];
    chartIds.forEach(id => {
      const el = document.getElementById(id);
      if (el && el.data) {
        Plotly.relayout(el, {
          "xaxis.gridcolor": isLight ? "rgba(0,0,0,0.06)" : "rgba(255,255,255,0.05)",
          "yaxis.gridcolor": isLight ? "rgba(0,0,0,0.06)" : "rgba(255,255,255,0.05)",
          "xaxis.tickfont.color": isLight ? "#475569" : "#c9d1d9",
          "yaxis.tickfont.color": isLight ? "#475569" : "#c9d1d9",
        }).catch(() => {});
      }
    });
  }
}

// Auto-run immediately to prevent theme flash
initTheme();

// ── Init on Page Load ───────────────────────────────────────────
document.addEventListener("DOMContentLoaded", async () => {
  initTheme();
  await loadAlgorithms();
  await refreshDatasetsList();
  await loadSampleCards();
  await refreshModelsList();
  await refreshReportsList();
});

// ── Algorithms ──────────────────────────────────────────────────
async function loadAlgorithms() {
  try {
    const res = await fetch("/api/ml/algorithms");
    if (res.ok) {
      state.algorithms = await res.json();
    }
  } catch (err) {
    console.error("Failed to load algorithms:", err);
  }
}

// ── Datasets ────────────────────────────────────────────────────
async function refreshDatasetsList() {
  try {
    const res = await fetch("/api/datasets");
    if (!res.ok) return;
    const datasets = await res.json();
    const selector = document.getElementById("dataset-selector");
    if (!selector) return;
    selector.innerHTML = '<option value="">-- Choose Dataset --</option>';

    datasets.forEach(ds => {
      const opt = document.createElement("option");
      opt.value = ds.id;
      opt.textContent = `${ds.name} (${ds.num_rows}r × ${ds.num_columns}c)`;
      selector.appendChild(opt);
    });

    if (datasets.length > 0 && !state.activeDatasetId) {
      handleDatasetSelect(datasets[0].id);
    }
  } catch (err) {
    console.error(err);
  }
}

async function loadSampleCards() {
  try {
    const res = await fetch("/api/datasets/samples/available");
    if (!res.ok) return;
    const samples = await res.json();
    const container = document.getElementById("samples-container");
    if (!container) return;
    container.innerHTML = "";

    const SAMPLE_ICONS = {
      iris: "🌸",
      breast_cancer: "🔬",
      wine: "🍷",
      diabetes: "🩺",
      titanic: "🚢",
      california_housing: "🏠",
      diabetes_regression: "📈"
    };

    samples.forEach(s => {
      const card = document.createElement("div");
      card.className = "sample-card";
      const icon = SAMPLE_ICONS[s.name] || "📊";
      card.innerHTML = `
        <div>
          <div class="sample-title">
            <span><span style="margin-right:6px; font-size:15px;">${icon}</span>${s.display_name}</span>
            <span class="badge ${s.task === 'classification' ? 'badge-info' : 'badge-algo'}">${s.task}</span>
          </div>
          <div class="sample-desc">${s.description}</div>
        </div>
        <button class="btn btn-secondary btn-sm" onclick="event.stopPropagation(); loadSample('${s.name}')">Load Dataset</button>
      `;
      card.onclick = () => loadSample(s.name);
      container.appendChild(card);
    });
  } catch (err) {
    console.error(err);
  }
}

async function loadSample(sampleName) {
  showToast(`Loading sample dataset '${sampleName}'...`, "info");
  try {
    const res = await fetch(`/api/datasets/sample/${sampleName}`, { method: "POST" });
    if (!res.ok) throw new Error(await res.text());
    const ds = await res.json();
    showToast(`Loaded ${ds.name} successfully!`, "success");
    await refreshDatasetsList();
    handleDatasetSelect(ds.id);
  } catch (err) {
    showToast(`Failed to load sample: ${err.message}`, "error");
  }
}

async function uploadFile(file) {
  if (!file) return;
  const formData = new FormData();
  formData.append("file", file);
  showToast(`Uploading ${file.name}...`, "info");

  try {
    const res = await fetch("/api/datasets/upload", {
      method: "POST",
      body: formData,
    });
    if (!res.ok) throw new Error(await res.text());
    const ds = await res.json();
    showToast(`Uploaded dataset '${ds.name}' successfully!`, "success");
    await refreshDatasetsList();
    handleDatasetSelect(ds.id);
  } catch (err) {
    showToast(`Upload failed: ${err.message}`, "error");
  }
}

async function handleDatasetSelect(datasetId) {
  if (!datasetId) return;
  state.activeDatasetId = datasetId;

  const selector = document.getElementById("dataset-selector");
  if (selector && selector.value !== datasetId) selector.value = datasetId;

  try {
    const res = await fetch(`/api/datasets/${datasetId}?preview_rows=10`);
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    state.activeDatasetMeta = data.metadata;
    state.previewData = data;

    // Update Topbar and Overview
    const elName = document.getElementById("active-dataset-name");
    if (elName) elName.innerText = data.metadata.name;
    const elKpiName = document.getElementById("kpi-dataset-name");
    if (elKpiName) elKpiName.innerText = data.metadata.name;
    const elDim = document.getElementById("kpi-dimensions");
    if (elDim) elDim.innerText = `${data.shape.rows} × ${data.shape.columns}`;
    const elStatus = document.getElementById("kpi-validation-status");
    if (elStatus) {
      const st = data.metadata.validation_status || "Pending";
      elStatus.innerText = st.replace(/_/g, " ");
      if (st.toLowerCase().includes("warn")) {
        elStatus.style.color = "var(--accent-amber-light)";
      } else if (st.toLowerCase().includes("valid") || st.toLowerCase().includes("pass")) {
        elStatus.style.color = "var(--accent-emerald-light)";
      } else if (st.toLowerCase().includes("fail") || st.toLowerCase().includes("error")) {
        elStatus.style.color = "var(--accent-rose-light)";
      } else {
        elStatus.style.color = "var(--text-main)";
      }
    }

    // Populate Target column selector
    const targetSel = document.getElementById("ml-target-col");
    if (targetSel) {
      targetSel.innerHTML = '<option value="">Select target column...</option>';
    }
    const edaTargetSel = document.getElementById("eda-target-col-select");
    if (edaTargetSel) {
      edaTargetSel.innerHTML = '<option value="">(Auto-detect target)</option>';
    }

    data.columns.forEach(col => {
      if (targetSel) {
        const opt = document.createElement("option");
        opt.value = col;
        opt.textContent = col;
        if (col.toLowerCase() === "target" || col.toLowerCase() === "survived") opt.selected = true;
        targetSel.appendChild(opt);
      }

      if (edaTargetSel) {
        const edaOpt = document.createElement("option");
        edaOpt.value = col;
        edaOpt.textContent = col;
        if (col.toLowerCase() === "target" || col.toLowerCase() === "survived") edaOpt.selected = true;
        edaTargetSel.appendChild(edaOpt);
      }
    });
    if (targetSel && targetSel.value) detectTaskType(targetSel.value);

    renderDatasetPreviewTable(data.columns, data.preview);
  } catch (err) {
    showToast(`Error loading dataset details: ${err.message}`, "error");
  }
}

function renderDatasetPreviewTable(columns, rows) {
  const table = document.getElementById("dataset-preview-table");
  if (!table) return;
  let thead = "<thead><tr>" + columns.map(c => `<th>${c}</th>`).join("") + "</tr></thead>";
  let tbody = "<tbody>";
  rows.forEach(r => {
    tbody += "<tr>" + columns.map(c => {
      const val = r[c];
      const isNull = val === null || val === undefined;
      const isNum = typeof val === 'number';
      return `<td>${isNull ? '<span style="color:var(--text-dim); font-family:var(--font-mono); font-size:11px;">null</span>' : isNum ? `<span style="font-family:var(--font-mono);">${val}</span>` : val}</td>`;
    }).join("") + "</tr>";
  });
  tbody += "</tbody>";
  table.innerHTML = thead + tbody;
}

function downloadCurrentDataset() {
  if (!state.activeDatasetId) return showToast("Select a dataset first", "info");
  window.open(`/api/datasets/${state.activeDatasetId}/download`, "_blank");
}

async function deleteCurrentDataset() {
  if (!state.activeDatasetId) return showToast("Select a dataset first", "info");
  if (!confirm("Are you sure you want to delete this dataset?")) return;
  try {
    const res = await fetch(`/api/datasets/${state.activeDatasetId}`, { method: "DELETE" });
    if (res.ok) {
      showToast("Dataset deleted", "success");
      state.activeDatasetId = null;
      await refreshDatasetsList();
    }
  } catch (err) {
    showToast(`Delete failed: ${err.message}`, "error");
  }
}

// ── Validation ──────────────────────────────────────────────────
async function runValidation() {
  if (!state.activeDatasetId) return showToast("Please select a dataset first", "info");
  showToast("Running 4-Tier Automated Validation...", "info");

  try {
    const targetSel = document.getElementById("ml-target-col");
    const targetCol = targetSel ? targetSel.value : "";
    const url = `/api/validation/run/${state.activeDatasetId}${targetCol ? `?target_column=${encodeURIComponent(targetCol)}` : ''}`;
    const res = await fetch(url, { method: "POST" });
    if (!res.ok) throw new Error(await res.text());
    const val = await res.json();

    const elValStatus = document.getElementById("val-status");
    if (elValStatus) {
      elValStatus.innerText = val.status.replace(/_/g, " ").toUpperCase();
      if (val.status.toLowerCase().includes("warn")) {
        elValStatus.style.color = "var(--accent-amber-light)";
      } else if (val.status.toLowerCase().includes("valid") || val.status.toLowerCase().includes("pass")) {
        elValStatus.style.color = "var(--accent-emerald-light)";
      } else {
        elValStatus.style.color = "var(--accent-rose-light)";
      }
    }
    const elValTotal = document.getElementById("val-total");
    if (elValTotal) elValTotal.innerText = val.total_checks;
    const elValPass = document.getElementById("val-passed");
    if (elValPass) elValPass.innerText = val.passed_checks;
    const elValWarn = document.getElementById("val-warnings");
    if (elValWarn) elValWarn.innerText = val.warning_count;
    const elValFail = document.getElementById("val-failed");
    if (elValFail) elValFail.innerText = val.failed_checks;
    const elKpiVal = document.getElementById("kpi-validation-status");
    if (elKpiVal) {
      elKpiVal.innerText = val.status.replace(/_/g, " ");
      if (val.status.toLowerCase().includes("warn")) {
        elKpiVal.style.color = "var(--accent-amber-light)";
      } else if (val.status.toLowerCase().includes("valid") || val.status.toLowerCase().includes("pass")) {
        elKpiVal.style.color = "var(--accent-emerald-light)";
      } else {
        elKpiVal.style.color = "var(--accent-rose-light)";
      }
    }

    // Render checks list
    const list = document.getElementById("validation-checks-list");
    if (list) {
      list.innerHTML = "";
      val.checks.forEach(c => {
        const card = document.createElement("div");
        card.className = "check-card";
        const badgeCls = c.status === "passed" ? "badge-valid" : c.status === "warning" ? "badge-warning" : "badge-invalid";
        card.innerHTML = `
          <div class="check-left">
            <span class="badge ${badgeCls}">
              <span style="width:6px; height:6px; border-radius:50%; background:currentColor; display:inline-block; margin-right:4px;"></span>
              ${c.status.toUpperCase()}
            </span>
            <div>
              <div class="check-msg"><strong>${c.check_name}</strong>: ${c.message}</div>
              <div class="check-sub">TIER: ${c.tier} ${c.suggestion ? `• SUGGESTION: ${c.suggestion}` : ''}</div>
            </div>
          </div>
        `;
        list.appendChild(card);
      });
    }

    showToast("Validation completed successfully!", "success");
  } catch (err) {
    showToast(`Validation error: ${err.message}`, "error");
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ── ENHANCED EDA ENGINE (11-Tab Controller & Plotly Interactive Visualizer) ──
// ═══════════════════════════════════════════════════════════════════════════════

state.edaData = null;
state.currentEDAInsights = [];

function switchEDASubtab(subtabId) {
  // Update nav buttons
  document.querySelectorAll(".eda-subtab-btn").forEach(btn => {
    btn.classList.remove("active");
  });
  const clickedBtn = Array.from(document.querySelectorAll(".eda-subtab-btn")).find(b => b.getAttribute("onclick")?.includes(subtabId));
  if (clickedBtn) clickedBtn.classList.add("active");

  // Update views
  document.querySelectorAll(".eda-section-view").forEach(sec => sec.classList.remove("active"));
  const targetSec = document.getElementById(`eda-sec-${subtabId}`);
  if (targetSec) targetSec.classList.add("active");

  // Re-trigger Plotly resize if needed
  if (window.Plotly) {
    window.dispatchEvent(new Event("resize"));
  }
}

async function runEDA() {
  return runEnhancedEDA();
}

async function runEnhancedEDA() {
  if (!state.activeDatasetId) return showToast("Select a dataset first", "info");
  const btn = document.getElementById("btn-run-eda");
  if (btn) {
    btn.disabled = true;
    btn.innerHTML = `<span class="loading-spinner"></span> Computing EDA...`;
  }
  showToast("Running automated EDA Engine with full statistical audits...", "info");

  try {
    const targetCol = document.getElementById("eda-target-col-select")?.value || "";
    const taskOverride = document.getElementById("eda-target-task-override")?.value || "";
    let url = `/api/eda/run/${state.activeDatasetId}?`;
    if (targetCol) url += `target_column=${encodeURIComponent(targetCol)}&`;
    if (taskOverride) url += `task_type=${encodeURIComponent(taskOverride)}`;

    const res = await fetch(url, { method: "POST" });
    if (!res.ok) throw new Error(await res.text());
    const payload = await res.json();
    state.edaData = payload.eda;
    renderEDAResults(state.edaData);
    showToast("EDA analysis and interactive visualisations ready!", "success");
  } catch (err) {
    showToast(`EDA calculation failed: ${err.message}`, "error");
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.innerHTML = "▶ Run Complete EDA";
    }
  }
}

function renderEDAResults(eda) {
  if (!eda) return;

  // 1. Overview
  renderOverviewSection(eda.overview || {});

  // 2. Quality
  renderQualitySection(eda.quality || {});

  // 3. Statistics
  renderStatisticsSection(eda.statistics || {});

  // 4. Univariate
  populateUnivariateControls(eda);

  // 5. Bivariate
  populateBivariateControls(eda);

  // 6. Multivariate
  renderMultivariateSection(eda);

  // 7. Outliers
  renderOutliersSection(eda);

  // 8. Target
  renderTargetSection(eda.target_analysis || {});

  // 9. Features
  renderFeaturesSection(eda.features || {});

  // 10. AI Insights
  renderInsightsSection(eda.insights || []);

  // Top overview summary text
  const box = document.getElementById("eda-overview-box");
  if (box && eda.overview) {
    const ov = eda.overview;
    box.innerHTML = `
      <span>📐 <strong>Dimensions:</strong> ${ov.num_rows?.toLocaleString() ?? 0} rows × ${ov.num_columns ?? 0} columns</span>
      <span>❤️ <strong>Health Score:</strong> <span style="color:#34d399; font-weight:700;">${ov.health_score ?? 100}/100</span> (${ov.health_grade ?? 'Good'})</span>
      <span>⚠️ <strong>Missing Data:</strong> ${ov.missing_pct ?? 0}%</span>
      <span>📑 <strong>Duplicate Rows:</strong> ${ov.duplicate_rows ?? 0}</span>
      <span>💾 <strong>Memory:</strong> ${ov.memory_mb ?? 0} MB</span>
    `;
  }
}

function renderOverviewSection(ov) {
  const healthVal = document.getElementById("eda-health-val");
  const healthBadge = document.getElementById("eda-health-badge");
  if (healthVal) healthVal.innerText = `${ov.health_score ?? '--'} / 100`;
  if (healthBadge) {
    const gr = ov.health_grade || "Good";
    healthBadge.innerText = `${gr.toUpperCase()} HEALTH`;
    healthBadge.className = `badge badge-${gr === 'Excellent' || gr === 'Good' ? 'valid' : 'invalid'}`;
  }

  const dimVal = document.getElementById("eda-dimensions-val");
  if (dimVal) dimVal.innerText = `${ov.num_rows?.toLocaleString() ?? 0} × ${ov.num_columns ?? 0}`;
  const cellsVal = document.getElementById("eda-cells-val");
  if (cellsVal) cellsVal.innerText = `${ov.total_cells?.toLocaleString() ?? 0} total cells`;

  const missVal = document.getElementById("eda-missing-val");
  if (missVal) missVal.innerText = `${ov.total_missing_cells?.toLocaleString() ?? 0} (${ov.missing_pct ?? 0}%)`;

  const memVal = document.getElementById("eda-memory-val");
  if (memVal) memVal.innerText = `${ov.memory_mb ?? 0} MB`;
  const dupVal = document.getElementById("eda-dup-rows-val");
  if (dupVal) dupVal.innerText = `${ov.duplicate_rows ?? 0} duplicate rows`;

  // Column types breakdown
  const typesBox = document.getElementById("eda-col-types-breakdown");
  if (typesBox) {
    typesBox.innerHTML = `
      <div style="display:flex; justify-content:space-between; align-items:center; padding:8px 12px; background:rgba(255,255,255,0.03); border-radius:var(--radius-sm);">
        <span>🔢 Numerical Features</span>
        <strong>${ov.num_numerical ?? (ov.numerical_columns?.length || 0)}</strong>
      </div>
      <div style="display:flex; justify-content:space-between; align-items:center; padding:8px 12px; background:rgba(255,255,255,0.03); border-radius:var(--radius-sm);">
        <span>🏷️ Categorical Features</span>
        <strong>${ov.num_categorical ?? (ov.categorical_columns?.length || 0)}</strong>
      </div>
      <div style="display:flex; justify-content:space-between; align-items:center; padding:8px 12px; background:rgba(255,255,255,0.03); border-radius:var(--radius-sm);">
        <span>🔘 Boolean Features</span>
        <strong>${ov.num_boolean ?? (ov.boolean_columns?.length || 0)}</strong>
      </div>
      <div style="display:flex; justify-content:space-between; align-items:center; padding:8px 12px; background:rgba(255,255,255,0.03); border-radius:var(--radius-sm);">
        <span>📅 Datetime Features</span>
        <strong>${ov.num_datetime ?? (ov.datetime_columns?.length || 0)}</strong>
      </div>
    `;
  }

  // Integrity summary
  const intBox = document.getElementById("eda-integrity-summary");
  if (intBox) {
    const dupCols = ov.duplicate_columns_count ?? 0;
    const constCols = ov.constant_columns_count ?? 0;
    intBox.innerHTML = `
      <div style="display:flex; justify-content:space-between; padding:8px 12px; background:rgba(255,255,255,0.03); border-radius:var(--radius-sm);">
        <span>Exact Duplicate Rows</span>
        <span class="badge ${ov.duplicate_rows > 0 ? 'badge-warning' : 'badge-valid'}">${ov.duplicate_rows ?? 0} (${ov.duplicate_rows_pct ?? 0}%)</span>
      </div>
      <div style="display:flex; justify-content:space-between; padding:8px 12px; background:rgba(255,255,255,0.03); border-radius:var(--radius-sm);">
        <span>Duplicate / Identical Columns</span>
        <span class="badge ${dupCols > 0 ? 'badge-warning' : 'badge-valid'}">${dupCols} detected</span>
      </div>
      <div style="display:flex; justify-content:space-between; padding:8px 12px; background:rgba(255,255,255,0.03); border-radius:var(--radius-sm);">
        <span>Zero-Variance Constant Columns</span>
        <span class="badge ${constCols > 0 ? 'badge-invalid' : 'badge-valid'}">${constCols} detected</span>
      </div>
    `;
  }
}

function renderQualitySection(qual) {
  const badge = document.getElementById("eda-quality-meter-badge");
  if (badge) {
    const sc = qual.quality_score ?? 100;
    badge.innerHTML = `<span class="badge ${sc >= 80 ? 'badge-success' : sc >= 60 ? 'badge-warning' : 'badge-critical'}">Quality Score: ${sc}/100 (${qual.critical_issues ?? 0} Critical, ${qual.warning_issues ?? 0} Warnings)</span>`;
  }

  // Issues table
  const tbody = document.querySelector("#eda-quality-issues-table tbody");
  if (tbody) {
    const issues = qual.issues || [];
    if (issues.length === 0) {
      tbody.innerHTML = '<tr><td colspan="4" style="text-align:center; color:#34d399; padding:20px;">🛡️ Clean dataset! No structural or quality defects detected.</td></tr>';
    } else {
      tbody.innerHTML = issues.map(i => {
        const sev = i.severity || "info";
        const bCls = sev === 'critical' ? 'badge-critical' : sev === 'warning' ? 'badge-warning' : 'badge-info';
        return `
          <tr>
            <td><span class="badge ${bCls}">${sev.toUpperCase()}</span></td>
            <td><strong>${i.column || 'Dataset'}</strong></td>
            <td><code>${i.type}</code></td>
            <td>${i.message}</td>
          </tr>
        `;
      }).join("");
    }
  }

  // Recommendations list
  const recBox = document.getElementById("eda-quality-recommendations-list");
  if (recBox) {
    const recs = qual.recommendations || [];
    if (recs.length === 0) {
      recBox.innerHTML = '<span style="color:#34d399; font-size:13px;">No data quality repairs needed.</span>';
    } else {
      recBox.innerHTML = recs.map(r => `
        <div style="display:flex; justify-content:space-between; align-items:flex-start; padding:10px 14px; background:var(--bg-card-hover); border:1px solid var(--border-subtle); border-radius:var(--radius-sm);">
          <div>
            <strong style="color:var(--text-bright); font-size:13px;">${r.column ? `Column [${r.column}]` : 'Dataset'}:</strong>
            <div style="color:var(--text-muted); font-size:12px; margin-top:2px;">${r.reason}</div>
          </div>
          <span class="badge ${r.priority === 'high' ? 'badge-critical' : 'badge-warning'}">${(r.priority || 'medium').toUpperCase()}</span>
        </div>
      `).join("");
    }
  }
}

function renderStatisticsSection(stats) {
  // Numerical stats
  const numStats = stats.numerical || {};
  const tbodyNum = document.querySelector("#eda-stats-table tbody");
  if (tbodyNum) {
    const rows = Object.entries(numStats).map(([col, s]) => `
      <tr>
        <td><strong>${col}</strong></td>
        <td>${s.count?.toLocaleString() ?? '-'}</td>
        <td>${s.mean !== null && s.mean !== undefined ? s.mean.toFixed(2) : '-'}</td>
        <td>${s.std !== null && s.std !== undefined ? s.std.toFixed(2) : '-'}</td>
        <td>${s.min !== null && s.min !== undefined ? s.min.toFixed(2) : '-'}</td>
        <td>${(s.p25 ?? s['25%']) !== null && (s.p25 ?? s['25%']) !== undefined ? (s.p25 ?? s['25%']).toFixed(2) : '-'}</td>
        <td>${s.median !== null && s.median !== undefined ? s.median.toFixed(2) : '-'}</td>
        <td>${(s.p75 ?? s['75%']) !== null && (s.p75 ?? s['75%']) !== undefined ? (s.p75 ?? s['75%']).toFixed(2) : '-'}</td>
        <td>${s.max !== null && s.max !== undefined ? s.max.toFixed(2) : '-'}</td>
        <td>${s.skewness !== null && s.skewness !== undefined ? s.skewness.toFixed(2) : '-'}</td>
      </tr>
    `).join("");
    tbodyNum.innerHTML = rows || '<tr><td colspan="10" style="text-align:center; color:var(--text-muted);">No numerical columns.</td></tr>';
  }

  // Categorical stats
  const catStats = stats.categorical || {};
  const tbodyCat = document.querySelector("#eda-cat-stats-table tbody");
  if (tbodyCat) {
    const rows = Object.entries(catStats).map(([col, s]) => `
      <tr>
        <td><strong>${col}</strong></td>
        <td>${s.count?.toLocaleString() ?? '-'}</td>
        <td>${s.nunique ?? '-'}</td>
        <td><code>${s.top_value ?? '-'}</code></td>
        <td>${s.top_freq?.toLocaleString() ?? '-'}</td>
        <td>${((s.top_freq / Math.max(s.count, 1)) * 100).toFixed(1)}%</td>
      </tr>
    `).join("");
    tbodyCat.innerHTML = rows || '<tr><td colspan="6" style="text-align:center; color:var(--text-muted);">No categorical columns.</td></tr>';
  }
}

function populateUnivariateControls(eda) {
  const sel = document.getElementById("eda-uni-col-select");
  if (!sel) return;

  const numCols = Object.keys(eda.statistics?.numerical || {});
  const catCols = Object.keys(eda.statistics?.categorical || {});
  const allCols = [...numCols, ...catCols];

  sel.innerHTML = allCols.map(c => `<option value="${c}">${c} (${numCols.includes(c) ? 'num' : 'cat'})</option>`).join("");
  if (allCols.length > 0) {
    renderUnivariateSelected();
  }
}

async function renderUnivariateSelected() {
  if (!state.activeDatasetId) return;
  const col = document.getElementById("eda-uni-col-select")?.value;
  const chartType = document.getElementById("eda-uni-chart-type")?.value || "hist_kde";
  if (!col) return;

  try {
    // 1. Plotly interactive chart
    const res = await fetch(`/api/eda/${state.activeDatasetId}/chart/${chartType}?column=${encodeURIComponent(col)}`);
    if (res.ok) {
      const data = await res.json();
      if (window.Plotly && data.figure) {
        Plotly.newPlot("eda-univariate-plot", data.figure.data, data.figure.layout, { responsive: true, displayModeBar: true });
      }
    }

    // 2. Column detail metrics
    const detailRes = await fetch(`/api/eda/${state.activeDatasetId}/univariate/detail?column=${encodeURIComponent(col)}`);
    if (detailRes.ok) {
      const det = await detailRes.json();
      const d = det.result?.data || {};
      const container = document.getElementById("eda-uni-stats-cards");
      if (container) {
        if (det.result?.type === "numerical") {
          container.innerHTML = `
            <div class="kpi-card"><div><div class="kpi-label">Mean</div><div class="kpi-val">${d.mean !== null ? d.mean.toFixed(2) : '-'}</div></div></div>
            <div class="kpi-card"><div><div class="kpi-label">Median</div><div class="kpi-val">${d.median !== null ? d.median.toFixed(2) : '-'}</div></div></div>
            <div class="kpi-card"><div><div class="kpi-label">Std Dev</div><div class="kpi-val">${d.std !== null ? d.std.toFixed(2) : '-'}</div></div></div>
            <div class="kpi-card"><div><div class="kpi-label">Skewness</div><div class="kpi-val">${d.skewness !== null ? d.skewness.toFixed(2) : '-'}</div><span class="badge ${d.is_normal ? 'badge-valid' : 'badge-warning'}">${d.is_normal ? 'Normal' : 'Skewed'}</span></div></div>
            <div class="kpi-card"><div><div class="kpi-label">IQR Bounds</div><div class="kpi-val">${d.iqr !== null ? d.iqr.toFixed(2) : '-'}</div></div></div>
            <div class="kpi-card"><div><div class="kpi-label">Outliers (IQR)</div><div class="kpi-val">${d.box_plot?.outlier_count ?? 0} (${d.box_plot?.outlier_pct ?? 0}%)</div></div></div>
          `;
        } else {
          container.innerHTML = `
            <div class="kpi-card"><div><div class="kpi-label">Unique Levels</div><div class="kpi-val">${d.nunique ?? 0}</div></div></div>
            <div class="kpi-card"><div><div class="kpi-label">Top Category</div><div class="kpi-val">${d.top_value ?? '-'}</div></div></div>
            <div class="kpi-card"><div><div class="kpi-label">Top Frequency</div><div class="kpi-val">${d.top_freq ?? 0} (${d.top_pct ?? 0}%)</div></div></div>
            <div class="kpi-card"><div><div class="kpi-label">Missing</div><div class="kpi-val">${d.missing ?? 0} (${d.missing_pct ?? 0}%)</div></div></div>
          `;
        }
      }
    }
  } catch (err) {
    console.error("Univariate render error:", err);
  }
}

function populateBivariateControls(eda) {
  const sel1 = document.getElementById("eda-biv-col1-select");
  const sel2 = document.getElementById("eda-biv-col2-select");
  if (!sel1 || !sel2) return;

  const numCols = Object.keys(eda.statistics?.numerical || {});
  const catCols = Object.keys(eda.statistics?.categorical || {});
  const allCols = [...numCols, ...catCols];

  sel1.innerHTML = allCols.map(c => `<option value="${c}">${c}</option>`).join("");
  sel2.innerHTML = allCols.map(c => `<option value="${c}">${c}</option>`).join("");

  if (allCols.length >= 2) {
    sel1.selectedIndex = 0;
    sel2.selectedIndex = 1;
    renderBivariateSelected();
  }

  // Top pairs table
  const tbody = document.querySelector("#eda-biv-top-pairs-table tbody");
  if (tbody) {
    const pairs = eda.bivariate?.top_pairwise_associations || [];
    tbody.innerHTML = pairs.slice(0, 10).map(p => `
      <tr>
        <td><strong>${p.col1}</strong></td>
        <td><strong>${p.col2}</strong></td>
        <td><code>${p.metric.toUpperCase()}</code></td>
        <td style="color:${Math.abs(p.score) > 0.7 ? '#34d399' : 'inherit'}">${p.score?.toFixed(3)}</td>
        <td><button class="btn btn-secondary btn-sm" onclick="setBivariatePair('${p.col1}', '${p.col2}')">View</button></td>
      </tr>
    `).join("");
  }
}

function setBivariatePair(c1, c2) {
  const sel1 = document.getElementById("eda-biv-col1-select");
  const sel2 = document.getElementById("eda-biv-col2-select");
  if (sel1 && sel2) {
    sel1.value = c1;
    sel2.value = c2;
    renderBivariateSelected();
    switchEDASubtab("bivariate");
  }
}

async function renderBivariateSelected() {
  if (!state.activeDatasetId) return;
  const col1 = document.getElementById("eda-biv-col1-select")?.value;
  const col2 = document.getElementById("eda-biv-col2-select")?.value;
  if (!col1 || !col2 || col1 === col2) return;

  try {
    const res = await fetch(`/api/eda/${state.activeDatasetId}/bivariate/pair?col1=${encodeURIComponent(col1)}&col2=${encodeURIComponent(col2)}`);
    if (!res.ok) return;
    const data = await res.json();
    const r = data.result || {};

    const infoBox = document.getElementById("eda-biv-relationship-info");
    let chartType = "scatter";

    if (r.relationship_type === "numerical_vs_numerical") {
      chartType = "scatter";
      if (infoBox) {
        infoBox.innerHTML = `
          <span class="badge badge-info">Numerical vs Numerical</span>
          Pearson r: <strong>${r.pearson?.correlation?.toFixed(3) ?? '-'}</strong> (p=${r.pearson?.p_value?.toFixed(4) ?? '-'}) |
          Spearman r: <strong>${r.spearman?.correlation?.toFixed(3) ?? '-'}</strong> |
          Trend: <code>${r.trendline?.formula ?? ''}</code>
        `;
      }
    } else if (r.relationship_type === "numerical_vs_categorical") {
      chartType = "grouped_box";
      if (infoBox) {
        infoBox.innerHTML = `
          <span class="badge badge-warning">Numerical vs Categorical</span>
          One-Way ANOVA F: <strong>${r.anova?.f_statistic?.toFixed(2) ?? '-'}</strong> (p=${r.anova?.p_value?.toFixed(4) ?? '-'}) |
          Effect Size (Eta²): <strong>${r.anova?.eta_squared?.toFixed(3) ?? '-'} (${r.anova?.effect_size ?? '-'})</strong>
        `;
      }
    } else {
      chartType = "stacked_bar";
      if (infoBox) {
        infoBox.innerHTML = `
          <span class="badge badge-algo">Categorical vs Categorical</span>
          Chi-Square: <strong>${r.chi_square?.statistic?.toFixed(2) ?? '-'}</strong> (p=${r.chi_square?.p_value?.toFixed(4) ?? '-'}) |
          Cramer's V: <strong>${r.chi_square?.cramers_v?.toFixed(3) ?? '-'} (${r.chi_square?.strength ?? '-'})</strong>
        `;
      }
    }

    // Load Plotly chart
    const chartRes = await fetch(`/api/eda/${state.activeDatasetId}/chart/${chartType}?col1=${encodeURIComponent(col1)}&col2=${encodeURIComponent(col2)}`);
    if (chartRes.ok) {
      const chartData = await chartRes.json();
      if (window.Plotly && chartData.figure) {
        Plotly.newPlot("eda-bivariate-plot", chartData.figure.data, chartData.figure.layout, { responsive: true, displayModeBar: true });
      }
    }
  } catch (err) {
    console.error("Bivariate render error:", err);
  }
}

async function renderMultivariateSection(eda) {
  if (!state.activeDatasetId) return;

  // Correlation heatmap
  try {
    const res = await fetch(`/api/eda/${state.activeDatasetId}/chart/corr_heatmap`);
    if (res.ok) {
      const d = await res.json();
      if (window.Plotly && d.figure) {
        Plotly.newPlot("eda-heatmap-plot", d.figure.data, d.figure.layout, { responsive: true, displayModeBar: true });
      }
    }
  } catch (e) {}

  // PCA plot
  try {
    const res = await fetch(`/api/eda/${state.activeDatasetId}/chart/pca`);
    if (res.ok) {
      const d = await res.json();
      if (window.Plotly && d.figure) {
        Plotly.newPlot("eda-pca-plot", d.figure.data, d.figure.layout, { responsive: true, displayModeBar: true });
      }
    }
  } catch (e) {}

  // VIF plot
  try {
    const res = await fetch(`/api/eda/${state.activeDatasetId}/chart/vif`);
    if (res.ok) {
      const d = await res.json();
      if (window.Plotly && d.figure) {
        Plotly.newPlot("eda-vif-plot", d.figure.data, d.figure.layout, { responsive: true, displayModeBar: true });
      }
    }
  } catch (e) {}

  // Full table
  const corr = eda.correlation?.matrix || {};
  const cols = Object.keys(corr);
  const corrTable = document.getElementById("eda-corr-table");
  if (corrTable && cols.length > 0) {
    const thead = "<thead><tr><th>Feature</th>" + cols.map(c => `<th>${c}</th>`).join("") + "</tr></thead>";
    const tbodyCorr = "<tbody>" + cols.map(rowCol => {
      return `<tr><td><strong style="color:var(--text-bright);">${rowCol}</strong></td>` + cols.map(colCol => {
        const val = corr[rowCol]?.[colCol] ?? 0;
        const color = val > 0.7 ? 'var(--accent-emerald-light)' : val < -0.5 ? 'var(--accent-rose-light)' : 'var(--text-main)';
        const bg = Math.abs(val) > 0.6 ? (val > 0 ? 'rgba(16, 185, 129, 0.10)' : 'rgba(244, 63, 94, 0.10)') : 'transparent';
        return `<td style="color:${color}; background:${bg}; font-family:var(--font-mono); font-weight:600;">${val.toFixed(2)}</td>`;
      }).join("") + "</tr>";
    }).join("") + "</tbody>";
    corrTable.innerHTML = thead + tbodyCorr;
  }
}

function renderOutliersSection(eda) {
  const out = eda.outliers || {};
  const methods = out.methods_comparison || [];
  const tbody = document.querySelector("#eda-outlier-methods-table tbody");
  if (tbody) {
    tbody.innerHTML = methods.map(m => `
      <tr>
        <td><strong>${m.method}</strong></td>
        <td>${m.outlier_count?.toLocaleString() ?? 0}</td>
        <td>${m.outlier_pct?.toFixed(2) ?? 0}%</td>
        <td>${m.affected_columns ?? 0}</td>
        <td><span style="font-size:12px; color:var(--text-muted);">${m.recommendation}</span></td>
      </tr>
    `).join("");
  }

  // Populate outlier inspection column selector
  const sel = document.getElementById("eda-outlier-col-select");
  if (sel) {
    const numCols = Object.keys(eda.statistics?.numerical || {});
    sel.innerHTML = numCols.map(c => `<option value="${c}">${c}</option>`).join("");
    if (numCols.length > 0) {
      renderOutlierScatter();
    }
  }
}

async function renderOutlierScatter() {
  if (!state.activeDatasetId) return;
  const col = document.getElementById("eda-outlier-col-select")?.value;
  if (!col) return;

  try {
    const res = await fetch(`/api/eda/${state.activeDatasetId}/chart/outlier_scatter?col1=${encodeURIComponent(col)}`);
    if (res.ok) {
      const d = await res.json();
      if (window.Plotly && d.figure) {
        Plotly.newPlot("eda-outlier-plot", d.figure.data, d.figure.layout, { responsive: true, displayModeBar: true });
      }
    }
  } catch (err) {
    console.error("Outlier chart error:", err);
  }
}

function renderTargetSection(target) {
  const box = document.getElementById("eda-target-summary-box");
  if (!target || !target.available) {
    if (box) {
      box.innerHTML = `
        <div style="color:var(--text-muted); font-size:13px; padding:16px; background:rgba(255,255,255,0.02); border-radius:var(--radius-sm);">
          No target column selected. Choose a target from the dropdown above or select from suggestions: <strong>${(target?.suggested_targets || []).join(', ')}</strong>
        </div>
      `;
    }
    return;
  }

  if (box) {
    box.innerHTML = `
      <div class="kpi-grid" style="margin-bottom:12px;">
        <div class="kpi-card"><div><div class="kpi-label">Target Name</div><div class="kpi-val">${target.target_column}</div></div></div>
        <div class="kpi-card"><div><div class="kpi-label">Inferred Task</div><div class="kpi-val" style="text-transform:capitalize;">${target.inferred_task}</div></div></div>
        <div class="kpi-card"><div><div class="kpi-label">Unique Values</div><div class="kpi-val">${target.nunique}</div></div></div>
        <div class="kpi-card"><div><div class="kpi-label">Imbalance / Skew</div><div class="kpi-val">${target.imbalance_status || (target.skewness ? `Skew: ${target.skewness.toFixed(2)}` : 'N/A')}</div></div></div>
      </div>
    `;
  }

  // Feature correlations table
  const tbody = document.querySelector("#eda-target-corrs-table tbody");
  if (tbody) {
    const corrs = target.feature_target_correlation || {};
    tbody.innerHTML = Object.entries(corrs).map(([feat, val]) => {
      const absVal = Math.abs(val || 0);
      const strength = absVal > 0.7 ? 'Strong Predictor' : absVal > 0.3 ? 'Moderate' : 'Weak';
      const color = absVal > 0.7 ? '#34d399' : absVal > 0.3 ? '#38bdf8' : 'var(--text-muted)';
      return `
        <tr>
          <td><strong>${feat}</strong></td>
          <td style="color:${color}; font-weight:600;">${val !== null ? val.toFixed(3) : '-'}</td>
          <td><span class="badge ${absVal > 0.7 ? 'badge-success' : 'badge-info'}">${strength}</span></td>
        </tr>
      `;
    }).join("");
  }
}

function renderFeaturesSection(feat) {
  // Correlated pairs
  const tbody = document.querySelector("#eda-high-corr-features-table tbody");
  if (tbody) {
    const pairs = feat.highly_correlated_pairs || [];
    if (pairs.length === 0) {
      tbody.innerHTML = '<tr><td colspan="5" style="text-align:center; color:#34d399; padding:16px;">✅ No redundant or collinear feature pairs (|r| >= 0.8) detected.</td></tr>';
    } else {
      tbody.innerHTML = pairs.map(p => `
        <tr>
          <td><strong>${p.feature1}</strong></td>
          <td><strong>${p.feature2}</strong></td>
          <td style="color:#f43f5e; font-weight:700;">${p.correlation?.toFixed(3)}</td>
          <td><span class="badge badge-critical">${p.severity.toUpperCase()}</span></td>
          <td style="font-size:12px; color:var(--text-muted);">Drop one feature before training linear models.</td>
        </tr>
      `).join("");
    }
  }

  // Anomalies list
  const anomBox = document.getElementById("eda-feature-anomalies-list");
  if (anomBox) {
    const lowVar = feat.low_variance_features || [];
    const leak = feat.potential_leakage_features || [];
    const cards = [];
    leak.forEach(l => {
      cards.push(`
        <div style="padding:10px 14px; background:rgba(244,63,94,0.1); border-left:3px solid #f43f5e; border-radius:4px;">
          <strong style="color:#f43f5e;">Target Leakage Risk: [${l.feature}]</strong>
          <div style="font-size:12px; color:var(--text-muted);">Correlation ${l.correlation_with_target?.toFixed(3)} with target. Verify data collection timeframe.</div>
        </div>
      `);
    });
    lowVar.forEach(v => {
      cards.push(`
        <div style="padding:10px 14px; background:rgba(255,255,255,0.03); border-radius:4px;">
          <strong style="color:#fbbf24;">Low Variance: [${v.feature}]</strong>
          <div style="font-size:12px; color:var(--text-muted);">Variance = ${v.variance?.toFixed(6)}. Near-constant information.</div>
        </div>
      `);
    });
    anomBox.innerHTML = cards.join("") || '<span style="color:#34d399; font-size:13px;">No variance or leakage risks found.</span>';
  }

  // Recommendations list
  const recBox = document.getElementById("eda-feature-recommendations-list");
  if (recBox) {
    const recs = feat.feature_selection_recommendations || [];
    recBox.innerHTML = recs.map(r => `
      <div style="display:flex; justify-content:space-between; align-items:flex-start; padding:10px 14px; background:rgba(255,255,255,0.03); border-radius:var(--radius-sm);">
        <span style="font-size:13px; color:var(--text-main);">${r.recommendation}</span>
        <span class="badge ${r.priority === 'critical' || r.priority === 'high' ? 'badge-critical' : 'badge-warning'}">${(r.priority || 'medium').toUpperCase()}</span>
      </div>
    `).join("") || '<span style="color:#34d399; font-size:13px;">Feature representation is balanced.</span>';
  }
}

function renderInsightsSection(insights) {
  state.currentEDAInsights = insights;
  filterEDAInsights("all");
}

function filterEDAInsights(severity) {
  const container = document.getElementById("eda-insights-container");
  if (!container) return;

  const filtered = severity === "all"
    ? state.currentEDAInsights
    : state.currentEDAInsights.filter(i => i.severity === severity);

  if (filtered.length === 0) {
    container.innerHTML = `<span style="color:var(--text-muted); font-size:13px;">No insights found for filter '${severity}'.</span>`;
    return;
  }

  container.innerHTML = filtered.map(ins => {
    const sev = ins.severity || "info";
    const bCls = sev === 'critical' ? 'badge-critical' : sev === 'warning' ? 'badge-warning' : sev === 'success' ? 'badge-success' : 'badge-info';
    return `
      <div class="eda-insight-card ${sev}">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
          <strong style="color:var(--text-bright); font-size:14px;">${ins.finding}</strong>
          <span class="badge ${bCls}">${sev.toUpperCase()}</span>
        </div>
        <div style="color:var(--text-muted); font-size:12px; margin-bottom:6px;"><strong>Evidence:</strong> ${ins.evidence}</div>
        <div style="color:var(--text-dim); font-size:12px; margin-bottom:6px;"><strong>Impact:</strong> ${ins.explanation}</div>
        <div style="color:#38bdf8; font-size:12px;"><strong>Action Plan:</strong> ${ins.recommendation}</div>
      </div>
    `;
  }).join("");
}

function downloadEDAReport(format) {
  if (!state.activeDatasetId) return showToast("Select a dataset first", "info");
  const targetCol = document.getElementById("eda-target-col-select")?.value || "";
  let url = `/api/eda/${state.activeDatasetId}/export?format=${format}`;
  if (targetCol) url += `&target_column=${encodeURIComponent(targetCol)}`;
  showToast(`Preparing ${format.toUpperCase()} report download...`, "info");
  window.open(url, "_blank");
}


// ── Preprocessing ───────────────────────────────────────────────
async function runAutoPreprocess() {
  if (!state.activeDatasetId) return showToast("Select a dataset first", "info");
  showToast("Running automated preprocessing pipeline...", "info");

  try {
    const targetSel = document.getElementById("ml-target-col");
    const targetCol = targetSel ? targetSel.value : null;
    const res = await fetch(`/api/preprocessing/auto/${state.activeDatasetId}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ target_column: targetCol || null }),
    });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    showToast(`Preprocessed! Created Version ${data.version} with ${data.operations_applied} transformations`, "success");
    await handleDatasetSelect(state.activeDatasetId);
    await refreshPreprocessingHistory();
  } catch (err) {
    showToast(`Preprocessing error: ${err.message}`, "error");
  }
}

async function applySingleOperation() {
  if (!state.activeDatasetId) return showToast("Select a dataset first", "info");
  const opType = document.getElementById("prep-op-type").value;
  const strategy = document.getElementById("prep-strategy").value;

  try {
    const res = await fetch(`/api/preprocessing/operation/${state.activeDatasetId}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ operation_type: opType, strategy }),
    });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    showToast(`Applied ${opType} (${strategy}) — created Version ${data.version}`, "success");
    await handleDatasetSelect(state.activeDatasetId);
    await refreshPreprocessingHistory();
  } catch (err) {
    showToast(`Operation failed: ${err.message}`, "error");
  }
}

async function refreshPreprocessingHistory() {
  if (!state.activeDatasetId) return;
  try {
    const res = await fetch(`/api/preprocessing/history/${state.activeDatasetId}`);
    if (!res.ok) return;
    const history = await res.json();
    const tbody = history.map(h => `
      <tr>
        <td><strong style="color:var(--text-bright);">${h.operation}</strong></td>
        <td><span class="badge badge-info">${h.category}</span></td>
        <td style="font-family:var(--font-mono); font-size:12px;">${(h.affected_columns || []).join(", ") || 'All'}</td>
        <td style="font-family:var(--font-mono);">${h.rows_before}</td>
        <td style="font-family:var(--font-mono);">${h.rows_after}</td>
        <td style="font-family:var(--font-mono); color:var(--text-dim);">${new Date(h.applied_at).toLocaleTimeString()}</td>
      </tr>
    `).join("");
    const tableEl = document.querySelector("#prep-history-table tbody");
    if (tableEl) tableEl.innerHTML = tbody;
  } catch (err) {
    console.error(err);
  }
}

// ── ML & Training ───────────────────────────────────────────────
async function detectTaskType(targetCol) {
  if (!state.activeDatasetId || !targetCol) return;
  try {
    const res = await fetch(`/api/ml/identify-task/${state.activeDatasetId}?target_column=${encodeURIComponent(targetCol)}`, { method: "POST" });
    if (!res.ok) return;
    const info = await res.json();
    state.taskType = info.task_type;

    const badge = document.getElementById("ml-task-badge");
    if (badge) {
      badge.innerText = `Task: ${info.task_type.toUpperCase()} (${info.nunique} unique values, ${info.confidence} confidence)`;
      badge.className = `badge ${info.task_type === 'classification' ? 'badge-info' : 'badge-algo'}`;
    }

    // Populate algorithm selector
    const algoSel = document.getElementById("ml-algorithm-select");
    if (algoSel) {
      algoSel.innerHTML = "";
      const algos = state.algorithms[info.task_type] || [];
      const ALGO_EMOJIS = {
        random_forest: "🌲",
        decision_tree: "🌴",
        gradient_boosting: "🚀",
        adaboost: "⚡",
        logistic_regression: "📈",
        linear_regression: "📈",
        ridge: "📐",
        lasso: "🎯",
        elastic_net: "🧬",
        svm: "🎯",
        svr: "🎯",
        knn: "📍",
        kneighbors: "📍",
        naive_bayes: "🧮",
        gaussian_nb: "🧮"
      };

      algos.forEach(a => {
        const opt = document.createElement("option");
        opt.value = a;
        const emo = ALGO_EMOJIS[a] || "⚙️";
        opt.textContent = `${emo} ${a.replace(/_/g, " ").toUpperCase()}`;
        algoSel.appendChild(opt);
      });
    }
  } catch (err) {
    console.error(err);
  }
}

async function trainSingleModel() {
  if (!state.activeDatasetId) return showToast("Select a dataset first", "info");
  const targetSel = document.getElementById("ml-target-col");
  const targetCol = targetSel ? targetSel.value : "";
  if (!targetCol) return showToast("Please select a target column", "info");

  const algorithm = document.getElementById("ml-algorithm-select").value;
  const testSize = parseFloat(document.getElementById("ml-test-size").value);
  const cvFolds = parseInt(document.getElementById("ml-cv-folds").value);

  showToast(`Training ${algorithm.toUpperCase()}...`, "info");

  try {
    const res = await fetch("/api/ml/train", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        dataset_id: state.activeDatasetId,
        target_column: targetCol,
        algorithm: algorithm,
        task_type: state.taskType,
        test_size: testSize,
        cv_folds: cvFolds,
      }),
    });

    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.detail || "Training failed");
    }
    const data = await res.json();
    showToast("Model trained successfully!", "success");

    // Pull full metrics from model object (has ALL metrics incl. confusion_matrix, per_class)
    const modelMeta = (data.models || [])[0] || {};
    const allMetrics = modelMeta.metrics || {};

    // Also pull scalar eval rows for CV fold breakdown
    const evals = data.evaluations || [];
    const cvRows = evals.filter(ev => ev.split === "cv");

    // ── Scalar metrics grid ──
    const scalarMetrics = Object.entries(allMetrics)
      .filter(([k, v]) => !_SKIP_METRICS.has(k) && typeof v === "number");

    const scalarHtml = scalarMetrics.length ? `
      <div style="display:grid; grid-template-columns:repeat(auto-fill,minmax(180px,1fr)); gap:10px; margin-bottom:18px;">
        ${scalarMetrics.map(([k, v]) => `
          <div style="background:rgba(56,189,248,0.06); border:1px solid rgba(56,189,248,0.12);
                      border-radius:8px; padding:12px 14px;">
            <div style="font-size:11px; color:var(--text-muted); margin-bottom:4px; text-transform:uppercase; letter-spacing:0.04em;">
              ${_METRIC_LABELS[k] || k}</div>
            <div style="font-size:20px; font-weight:700; font-family:var(--font-mono); color:var(--accent-cyan-light);">
              ${v.toFixed(4)}</div>
          </div>`).join("")}
      </div>` : "";

    // ── Confusion matrix ──
    const cm = allMetrics.confusion_matrix;
    const classLabels = allMetrics.class_labels || [];
    const cmHtml = cm ? (() => {
      const rows = cm.map((row, i) => {
        const cells = row.map((val, j) => {
          const isCorrect = i === j;
          const bg = isCorrect ? "rgba(52,211,153,0.15)" : "rgba(239,68,68,0.1)";
          return `<td style="text-align:center; padding:8px 12px; font-family:var(--font-mono);
                  font-weight:${isCorrect?700:400}; font-size:13px;
                  color:${isCorrect?'#34d399':'#f87171'}; background:${bg};
                  border:1px solid rgba(255,255,255,0.06);">${val}</td>`;
        }).join("");
        const lbl = classLabels[i] || `Class ${i}`;
        return `<tr>
          <td style="padding:8px 12px; font-size:11px; color:var(--text-muted); font-weight:600;
                     border:1px solid rgba(255,255,255,0.06); white-space:nowrap;">Actual: ${lbl}</td>
          ${cells}</tr>`;
      }).join("");
      const headerCells = cm[0].map((_, j) => {
        const lbl = classLabels[j] || `Class ${j}`;
        return `<th style="padding:6px 12px; font-size:11px; color:var(--text-muted);
                font-weight:600; border:1px solid rgba(255,255,255,0.06);
                white-space:nowrap;">Pred: ${lbl}</th>`;
      }).join("");
      return `
        <div style="margin-bottom:18px;">
          <div style="font-size:12px; font-weight:600; color:var(--text-muted); margin-bottom:8px;
                      text-transform:uppercase; letter-spacing:0.06em;">Confusion Matrix</div>
          <div style="overflow-x:auto;">
            <table style="border-collapse:collapse; font-size:13px;">
              <thead><tr>
                <th style="padding:6px 12px; border:1px solid rgba(255,255,255,0.06);"></th>
                ${headerCells}</tr></thead>
              <tbody>${rows}</tbody>
            </table>
          </div>
        </div>`;
    })() : "";

    // ── Per-class breakdown ──
    const perClass = allMetrics.per_class;
    const perClassHtml = perClass ? `
      <div style="margin-bottom:18px;">
        <div style="font-size:12px; font-weight:600; color:var(--text-muted); margin-bottom:8px;
                    text-transform:uppercase; letter-spacing:0.06em;">Per-Class Metrics</div>
        <div style="overflow-x:auto;">
          <table style="border-collapse:collapse; width:100%; font-size:12px;">
            <thead><tr style="border-bottom:1px solid rgba(255,255,255,0.08);">
              ${["Class","Precision","Recall","F1","Support"].map(h =>
                `<th style="padding:8px 12px; text-align:left; color:var(--text-muted); font-weight:600;">${h}</th>`).join("")}
            </tr></thead>
            <tbody>
              ${Object.entries(perClass).map(([cls, m]) => `
                <tr style="border-bottom:1px solid rgba(255,255,255,0.04);">
                  <td style="padding:8px 12px; font-family:var(--font-mono); color:var(--accent-cyan-light); font-weight:600;">${cls}</td>
                  <td style="padding:8px 12px; font-family:var(--font-mono);">${m.precision?.toFixed(4) ?? "—"}</td>
                  <td style="padding:8px 12px; font-family:var(--font-mono);">${m.recall?.toFixed(4) ?? "—"}</td>
                  <td style="padding:8px 12px; font-family:var(--font-mono);">${m.f1?.toFixed(4) ?? "—"}</td>
                  <td style="padding:8px 12px; font-family:var(--font-mono); color:var(--text-muted);">${m.support ?? "—"}</td>
                </tr>`).join("")}
            </tbody>
          </table>
        </div>
      </div>` : "";

    // ── CV fold breakdown ──
    const cvHtml = cvRows.length ? `
      <div>
        <div style="font-size:12px; font-weight:600; color:var(--text-muted); margin-bottom:8px;
                    text-transform:uppercase; letter-spacing:0.06em;">Cross-Validation Folds</div>
        <div style="display:flex; gap:8px; flex-wrap:wrap;">
          ${cvRows.map(ev => `
            <div style="background:rgba(99,102,241,0.1); border:1px solid rgba(99,102,241,0.2);
                        border-radius:6px; padding:8px 12px; text-align:center;">
              <div style="font-size:10px; color:var(--text-muted);">${ev.metric_name}</div>
              <div style="font-family:var(--font-mono); font-size:14px; font-weight:700;
                          color:#a5b4fc;">${Number(ev.metric_value).toFixed(4)}</div>
            </div>`).join("")}
        </div>
      </div>` : "";

    const displayEl = document.getElementById("ml-result-display");
    if (displayEl) {
      displayEl.innerHTML = `
        <div style="background:var(--bg-card); padding:20px; border-radius:var(--radius-md);
                    border:1px solid var(--border-card); box-shadow:var(--shadow-sm);">
          <div style="display:flex; justify-content:space-between; align-items:center;
                      margin-bottom:18px; border-bottom:1px solid var(--border-subtle); padding-bottom:12px;">
            <div>
              <h4 style="color:var(--text-bright); font-size:15px; font-weight:700; margin-bottom:2px;">
                ${algorithm.replace(/_/g," ").toUpperCase()} Results</h4>
              <div style="font-size:12px; color:var(--text-muted);">
                Task: ${state.taskType} &nbsp;|&nbsp; Test split: ${(testSize*100).toFixed(0)}%
              </div>
            </div>
            <span class="badge badge-valid">CONVERGED</span>
          </div>
          ${scalarHtml}
          ${cmHtml}
          ${perClassHtml}
          ${cvHtml}
        </div>
      `;
    }

    await refreshModelsList();
    // ── Render visualization charts for this trained model
    renderModelVisualizations(allMetrics, cvRows, algorithm);
  } catch (err) {
    showToast(`Training failed: ${err.message}`, "error");
  }
}


async function runModelComparison() {
  if (!state.activeDatasetId) return showToast("Select a dataset first", "info");
  const targetSel = document.getElementById("ml-target-col");
  const targetCol = targetSel ? targetSel.value : "";
  if (!targetCol) return showToast("Select target column", "info");

  showToast("Benchmarking all algorithms on dataset...", "info");

  try {
    const res = await fetch("/api/ml/compare", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        dataset_id: state.activeDatasetId,
        target_column: targetCol,
        task_type: state.taskType,
      }),
    });
    if (!res.ok) throw new Error((await res.json()).detail || await res.text());
    const data = await res.json();
    _lastComparisonData = data;
    showToast(`Comparison complete! Best: ${data.best_algorithm}`, "success");

    // Collect all scalar metric keys across all results
    const allMetricKeys = new Set();
    (data.results || []).forEach(r => {
      Object.keys(r.metrics || {}).forEach(k => {
        if (!_SKIP_METRICS.has(k) && typeof r.metrics[k] === "number") allMetricKeys.add(k);
      });
    });

    // Populate metric dropdown
    const sel = document.getElementById("compare-metric-select");
    sel.innerHTML = Array.from(allMetricKeys).map(k =>
      `<option value="${k}" ${k === data.ranking_metric ? "selected" : ""}>${_METRIC_LABELS[k] || k}</option>`
    ).join("");

    // Task badge
    const taskBadge = document.getElementById("compare-task-badge");
    if (taskBadge) taskBadge.textContent = `Task: ${data.task_type || "?"}`;

    // Show panels
    document.getElementById("compare-panel").style.display = "";
    document.getElementById("overall-ranking-panel").style.display = "";

    reRenderComparison();
    // ─ Render multi-metric grouped chart
    initGroupedMetricsSelect(data.results || []);
    await refreshModelsList();
  } catch (err) {
    showToast(`Comparison failed: ${err.message}`, "error");
  }
}

function reRenderComparison() {
  if (!_lastComparisonData) return;
  const sel = document.getElementById("compare-metric-select");
  const selectedMetric = sel ? sel.value : (_lastComparisonData.ranking_metric || "accuracy");
  const data = _lastComparisonData;

  // Sort results by selected metric
  const results = [...(data.results || [])].filter(r => r.metrics && typeof r.metrics[selectedMetric] === "number");
  const lowerBetter = _LOWER_IS_BETTER.has(selectedMetric);
  results.sort((a, b) => {
    const av = a.metrics[selectedMetric] ?? (lowerBetter ? Infinity : -Infinity);
    const bv = b.metrics[selectedMetric] ?? (lowerBetter ? Infinity : -Infinity);
    return lowerBetter ? av - bv : bv - av;
  });

  renderCompareChart(results, selectedMetric, data);
  renderCompareTable(results, selectedMetric, data);
  renderOverallRankings(data.results || []);
}

function renderCompareChart(results, selectedMetric, data) {
  const el = document.getElementById("compare-chart");
  if (!el || !results.length) return;

  const algorithms = results.map(r => r.algorithm);
  const scores = results.map(r => +(r.metrics[selectedMetric] ?? 0).toFixed(4));
  const metricLabel = _METRIC_LABELS[selectedMetric] || selectedMetric;

  // Collect all available scalar metrics for a secondary grouped bar
  const allScalarKeys = new Set();
  results.forEach(r => Object.keys(r.metrics || {}).forEach(k => {
    if (!_SKIP_METRICS.has(k) && typeof r.metrics[k] === "number") allScalarKeys.add(k);
  }));

  // Primary trace: selected metric
  const traces = [{
    x: algorithms,
    y: scores,
    type: "bar",
    name: metricLabel,
    marker: {
      color: scores.map((_, i) => i === 0 ? "rgba(56,205,186,0.9)" : "rgba(99,120,210,0.7)"),
      line: { color: "rgba(255,255,255,0.15)", width: 1 },
    },
    text: scores.map(s => s.toFixed(4)),
    textposition: "outside",
    textfont: { size: 11, color: "#c9d1d9"},
  }];

  const layout = {
    paper_bgcolor: "transparent",
    plot_bgcolor: "transparent",
    font: { family: "Inter, sans-serif", color: "#8b949e", size: 12 },
    margin: { t: 30, b: 60, l: 50, r: 20 },
    xaxis: {
      tickfont: { color: "#c9d1d9", size: 11 },
      gridcolor: "rgba(255,255,255,0.05)",
    },
    yaxis: {
      title: metricLabel,
      titlefont: { color: "#8b949e", size: 11 },
      tickfont: { color: "#c9d1d9", size: 10 },
      gridcolor: "rgba(255,255,255,0.07)",
      range: [0, Math.min(1.15, Math.max(...scores) * 1.12)],
    },
    showlegend: false,
    bargap: 0.25,
  };

  Plotly.newPlot(el, traces, layout, { responsive: true, displayModeBar: false });
}

function renderCompareTable(results, selectedMetric, data) {
  const thead = document.getElementById("compare-full-thead");
  const tbody = document.getElementById("compare-full-tbody");
  if (!thead || !tbody || !results.length) return;

  // Collect all scalar metric keys
  const metricKeys = [];
  const seen = new Set();
  results.forEach(r => Object.keys(r.metrics || {}).forEach(k => {
    if (!_SKIP_METRICS.has(k) && typeof r.metrics[k] === "number" && !seen.has(k)) {
      seen.add(k); metricKeys.push(k);
    }
  }));

  // Header
  thead.innerHTML = `<th>Rank</th><th>Algorithm</th><th>Time</th>` +
    metricKeys.map(k =>
      `<th style="${ k === selectedMetric ? 'color:var(--accent-amber-light); background:rgba(251,191,36,0.08);' : '' }">
        ${_METRIC_LABELS[k] || k}</th>`
    ).join("");

  // Rows
  tbody.innerHTML = results.map((r, idx) => {
    const rankEmoji = idx === 0 ? "🥇" : idx === 1 ? "🥈" : idx === 2 ? "🥉" : `#${idx+1}`;
    const cells = metricKeys.map(k => {
      const v = r.metrics[k];
      const isSelected = k === selectedMetric;
      const display = typeof v === "number" ? v.toFixed(4) : "—";
      return `<td style="${ isSelected ? 'color:var(--accent-amber-light); font-weight:600; background:rgba(251,191,36,0.05);' : '' } font-family:var(--font-mono); font-size:12px;">${display}</td>`;
    }).join("");
    return `<tr>
      <td><strong style="color:${ idx===0?'var(--accent-amber-light)':idx===1?'#adbac7':'var(--text-muted)'}; font-family:var(--font-mono);">${rankEmoji}</strong></td>
      <td><span class="badge ${idx===0?'badge-valid':'badge-algo'}">${r.algorithm}</span></td>
      <td style="font-family:var(--font-mono); font-size:12px;">${(r.training_time_sec??0).toFixed(3)}s</td>
      ${cells}
    </tr>`;
  }).join("");
}

function renderOverallRankings(allResults) {
  // Only work with results that have metrics (skip failed runs)
  const validResults = allResults.filter(r => r.metrics && Object.keys(r.metrics).length > 0);
  if (!validResults.length) return;

  const tbody = document.querySelector("#ml-leaderboard-table tbody");
  const thead = document.querySelector("#ml-leaderboard-table thead tr");
  if (!tbody || !thead) return;

  // Dynamically discover all scalar metric keys from results
  const metricKeys = [];
  const seen = new Set();
  validResults.forEach(r => {
    Object.keys(r.metrics).forEach(k => {
      if (!_SKIP_METRICS.has(k) && typeof r.metrics[k] === "number" && !seen.has(k)) {
        seen.add(k);
        metricKeys.push(k);
      }
    });
  });

  if (!metricKeys.length) return;

  // Rank each algorithm per metric
  const rankSums = {};
  const countMap = {};
  validResults.forEach(r => { rankSums[r.algorithm] = 0; countMap[r.algorithm] = 0; });

  metricKeys.forEach(k => {
    const lowerBetter = _LOWER_IS_BETTER.has(k);
    const withMetric = validResults.filter(r => typeof r.metrics[k] === "number");
    const sorted = [...withMetric].sort((a, b) =>
      lowerBetter ? a.metrics[k] - b.metrics[k] : b.metrics[k] - a.metrics[k]
    );
    sorted.forEach((r, i) => { rankSums[r.algorithm] += (i + 1); countMap[r.algorithm]++; });
  });

  // Build sorted ranked list — keep direct reference to original r.metrics
  const ranked = validResults
    .map(r => ({
      algorithm: r.algorithm,
      avgRank: countMap[r.algorithm] > 0 ? rankSums[r.algorithm] / countMap[r.algorithm] : 999,
      metrics: r.metrics,            // direct reference, NOT a copy
      training_time_sec: r.training_time_sec || 0,
    }))
    .sort((a, b) => a.avgRank - b.avgRank);  // lowest avgRank = best

  // Preferred display order for columns, then any extras
  const PREFERRED_ORDER = [
    "accuracy", "f1", "precision", "recall",
    "mcc", "roc_auc", "balanced_accuracy",
    "f1_macro", "precision_macro", "recall_macro",
    "specificity", "sensitivity",
    "r2", "mae", "rmse", "mape",
    "cv_mean_accuracy", "cv_std_accuracy", "cv_mean_rmse",
  ];
  const displayKeys = PREFERRED_ORDER.filter(k => seen.has(k))
    .concat(metricKeys.filter(k => !PREFERRED_ORDER.includes(k)));

  // Write header dynamically
  thead.innerHTML =
    "<th>Rank</th><th>Algorithm</th>" +
    `<th style="color:var(--accent-amber-light);">Composite Score</th>` +
    displayKeys.map(k => `<th>${_METRIC_LABELS[k] || k}</th>`).join("") +
    "<th>Train Time</th>";

  // Write rows
  const MEDAL = ["🥇", "🥈", "🥉"];
  tbody.innerHTML = ranked.map((r, idx) => {
    const compositeScore = r.avgRank < 999 ? (1 / r.avgRank).toFixed(4) : "0.0000";
    const metricCells = displayKeys.map(k => {
      const v = r.metrics[k];
      return `<td style="font-family:var(--font-mono); font-size:12px; ${
        idx === 0 ? "color:#34d399; font-weight:600;" : ""
      }">${typeof v === "number" ? v.toFixed(4) : "—"}</td>`;
    }).join("");
    const rankLabel = MEDAL[idx] || `<span style="font-family:var(--font-mono);">#${idx + 1}</span>`;
    return `<tr>
      <td><strong style="color:${
        idx === 0 ? "var(--accent-amber-light)" :
        idx === 1 ? "#adbac7" : "var(--text-muted)"
      }; font-family:var(--font-mono);">${rankLabel}</strong></td>
      <td><span class="badge ${idx === 0 ? "badge-valid" : "badge-algo"}">${r.algorithm}</span></td>
      <td><strong style="color:var(--accent-cyan-light); font-family:var(--font-mono); font-size:15px;">${compositeScore}</strong></td>
      ${metricCells}
      <td style="font-family:var(--font-mono); font-size:12px;">${r.training_time_sec.toFixed(3)}s</td>
    </tr>`;
  }).join("");
}


// \u2500\u2500 ML Model Visualization Functions \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
// Stored last trained model data for viz tab switching
let _lastTrainedData = null;

const _VIZ_COLORS = [
  "rgba(56,205,186,0.85)","rgba(99,102,241,0.85)","rgba(251,191,36,0.85)",
  "rgba(239,68,68,0.85)","rgba(16,185,129,0.85)","rgba(245,158,11,0.85)",
  "rgba(139,92,246,0.85)","rgba(59,130,246,0.85)","rgba(236,72,153,0.85)",
];
const _PLOTLY_LAYOUT_BASE = {
  paper_bgcolor: "transparent", plot_bgcolor: "transparent",
  font: { family: "Inter, sans-serif", color: "#8b949e", size: 12 },
  margin: { t: 30, b: 50, l: 60, r: 20 },
};

// Switch between viz tabs (8 tabs)
const _ALL_VIZ_TABS = ["confusion","radar","cv","bar","feature","perclass","gauge","history"];
let _cmNormalized = false;  // confusion matrix normalize toggle state

function showVizTab(tab) {
  _ALL_VIZ_TABS.forEach(t => {
    const container = document.getElementById(`viz-container-${t}`);
    const btn = document.getElementById(`viz-tab-${t}`);
    if (container) container.style.display = t === tab ? "" : "none";
    if (btn) {
      btn.style.background = t === tab ? "rgba(56,205,186,0.2)" : "";
      btn.style.color = t === tab ? "var(--accent-cyan-light)" : "";
      btn.style.borderColor = t === tab ? "var(--accent-cyan)" : "";
    }
  });
  // Show/hide CM normalize toggle
  const cmOpts = document.getElementById("viz-cm-options");
  if (cmOpts) cmOpts.style.display = tab === "confusion" ? "flex" : "none";
  // Plotly resize after reveal
  const el = document.getElementById(`viz-container-${tab}`);
  if (el && el.data) Plotly.Plots.resize(el);
}

// Toggle confusion matrix between count and percentage
function toggleCMNormalize() {
  _cmNormalized = !_cmNormalized;
  const btn = document.getElementById("viz-cm-normalize-btn");
  if (btn) btn.textContent = _cmNormalized ? "Show Counts" : "Show %";
  if (_lastTrainedData) renderConfusionMatrix(_lastTrainedData.modelMetrics);
}

// Main entry point — called after trainSingleModel completes
function renderModelVisualizations(modelMetrics, cvRows, algorithm) {
  _lastTrainedData = { modelMetrics, cvRows, algorithm };
  _cmNormalized = false;
  const btn = document.getElementById("viz-cm-normalize-btn");
  if (btn) btn.textContent = "Show %";

  // Update algo badge
  const badge = document.getElementById("viz-algo-badge");
  if (badge) badge.textContent = algorithm.replace(/_/g," ").toUpperCase();

  // Track history
  if (!window._vizHistory) window._vizHistory = [];
  window._vizHistory.push({
    algorithm, timestamp: new Date().toLocaleTimeString(),
    metrics: {...modelMetrics},
  });

  const panel = document.getElementById("ml-viz-panel");
  if (panel) panel.style.display = "";
  showVizTab("confusion");

  // Render all tabs upfront (hidden) so tab switching is instant
  renderConfusionMatrix(modelMetrics);
  renderMetricRadar(modelMetrics, algorithm);
  renderCVFoldsChart(cvRows, algorithm);
  renderMetricsBars(modelMetrics, algorithm);
  renderFeatureImportance(modelMetrics, algorithm);
  renderPerClassChart(modelMetrics, algorithm);
  renderGaugeCharts(modelMetrics);
  renderTrainingHistory();
}

// ── Chart 1: Confusion Matrix Heatmap (with normalize toggle) ────
function renderConfusionMatrix(metrics) {
  const el = document.getElementById("viz-container-confusion");
  if (!el) return;
  const cm = metrics.confusion_matrix;
  const labels = metrics.class_labels || (cm ? cm.map((_, i) => `Class ${i}`) : []);

  if (!cm || !cm.length) {
    el.innerHTML = `<div style="display:flex;align-items:center;justify-content:center;height:300px;color:var(--text-muted);font-size:14px;">
      Confusion matrix not available (regression task)</div>`;
    return;
  }

  // Row sums for normalization
  const rowSums = cm.map(row => row.reduce((a, b) => a + b, 0));
  const maxVal = _cmNormalized ? 1 : Math.max(...cm.flat());
  const zData = cm.map((row, i) =>
    _cmNormalized
      ? row.map(v => rowSums[i] > 0 ? +(v / rowSums[i]).toFixed(4) : 0)
      : row
  );
  const zNorm = cm.map((row, i) => row.map(v => v / (Math.max(...cm.flat()) || 1)));

  const annotations = [];
  zData.forEach((row, i) => {
    row.forEach((val, j) => {
      const raw = cm[i][j];
      const norm = zNorm[i][j];
      annotations.push({
        x: j, y: i,
        text: _cmNormalized ? `<b>${(val*100).toFixed(1)}%</b>` : `<b>${raw}</b>`,
        font: { color: norm > 0.5 ? "#0b101c" : "#e6edf3", size: 13, family: "JetBrains Mono, monospace" },
        showarrow: false,
      });
    });
  });

  const trace = {
    z: zData,
    x: labels.map(l => `Pred: ${l}`),
    y: labels.map(l => `Act: ${l}`),
    type: "heatmap",
    colorscale: [[0,"rgba(239,68,68,0.2)"],[0.5,"rgba(99,102,241,0.55)"],[1,"rgba(52,211,153,0.92)"]],
    showscale: true,
    colorbar: { thickness: 14, len: 0.85, tickfont: { color: "#8b949e", size: 10 },
                title: { text: _cmNormalized ? "%" : "Count", font: { color: "#8b949e", size: 10 } } },
    hoverongaps: false,
    hovertemplate: "Actual: %{y}<br>Predicted: %{x}<br>" +
      (_cmNormalized ? "Rate: <b>%{z:.3f}</b>" : "Count: <b>%{z}</b>") + "<extra></extra>",
  };

  Plotly.newPlot(el, [trace], {
    ...structuredClone(_PLOTLY_LAYOUT_BASE),
    title: { text: _cmNormalized ? "Confusion Matrix (Normalized)" : "Confusion Matrix (Counts)",
             font: { color: "#c9d1d9", size: 14 }, x: 0.05 },
    annotations,
    xaxis: { tickfont: { color: "#c9d1d9", size: 11 }, side: "bottom", gridcolor: "rgba(255,255,255,0.04)" },
    yaxis: { tickfont: { color: "#c9d1d9", size: 11 }, autorange: "reversed", gridcolor: "rgba(255,255,255,0.04)" },
    margin: { t: 55, b: 90, l: 110, r: 80 },
  }, { responsive: true, displayModeBar: false });
}

// ── Chart 2: Metric Radar with perfect-model reference ───────────
function renderMetricRadar(metrics, algorithm) {
  const el = document.getElementById("viz-container-radar");
  if (!el) return;
  const RADAR_KEYS = ["accuracy","precision","recall","f1","balanced_accuracy","mcc","roc_auc","specificity","sensitivity"];
  const available = RADAR_KEYS.filter(k => typeof metrics[k] === "number" && metrics[k] >= 0 && metrics[k] <= 1);

  if (available.length < 3) {
    el.innerHTML = `<div style="display:flex;align-items:center;justify-content:center;height:300px;color:var(--text-muted);">
      Need ≥ 3 bounded [0,1] metrics for radar (not available for regression)</div>`;
    return;
  }

  const thetaLabels = available.map(k => _METRIC_LABELS[k] || k);
  const modelVals = available.map(k => metrics[k]);
  const perfect = available.map(() => 1);

  // Close polygon
  const closedTheta = [...thetaLabels, thetaLabels[0]];
  const closedModel = [...modelVals, modelVals[0]];
  const closedPerfect = [...perfect, perfect[0]];

  const traces = [
    {
      type: "scatterpolar", r: closedPerfect, theta: closedTheta,
      fill: "toself", fillcolor: "rgba(255,255,255,0.03)",
      line: { color: "rgba(255,255,255,0.15)", width: 1, dash: "dot" },
      name: "Perfect Model",
      hovertemplate: "Perfect: 1.0000<extra></extra>",
    },
    {
      type: "scatterpolar", r: closedModel, theta: closedTheta,
      fill: "toself", fillcolor: "rgba(56,205,186,0.18)",
      line: { color: "rgba(56,205,186,0.9)", width: 2.5 },
      marker: { color: "rgba(56,205,186,1)", size: 7 },
      name: algorithm.replace(/_/g," "),
      hovertemplate: "<b>%{theta}</b>: %{r:.4f}<extra></extra>",
    },
  ];

  Plotly.newPlot(el, traces, {
    paper_bgcolor: "transparent",
    polar: {
      bgcolor: "rgba(0,0,0,0)",
      radialaxis: { visible: true, range: [0, 1], color: "#8b949e",
                    gridcolor: "rgba(255,255,255,0.08)", tickfont: { size: 9, color: "#8b949e" } },
      angularaxis: { tickfont: { size: 12, color: "#c9d1d9" },
                     linecolor: "rgba(255,255,255,0.12)", gridcolor: "rgba(255,255,255,0.07)" },
    },
    title: { text: `${algorithm.replace(/_/g," ")} vs Perfect Model`, font: { color: "#c9d1d9", size: 13 }, x: 0.05 },
    showlegend: true,
    legend: { font: { color: "#8b949e", size: 11 }, bgcolor: "rgba(0,0,0,0.3)",
              bordercolor: "rgba(255,255,255,0.08)", borderwidth: 1 },
    margin: { t: 60, b: 30, l: 60, r: 60 },
  }, { responsive: true, displayModeBar: false });
}

// ── Chart 3: CV Fold Scores (bar + mean line + std band) ─────────
function renderCVFoldsChart(cvRows, algorithm) {
  const el = document.getElementById("viz-container-cv");
  if (!el) return;
  const folds = cvRows.filter(e => /^cv_fold_\d+$/.test(e.metric_name))
                      .sort((a, b) => a.metric_name.localeCompare(b.metric_name));

  if (!folds.length) {
    el.innerHTML = `<div style="display:flex;align-items:center;justify-content:center;height:220px;color:var(--text-muted);">No CV fold data</div>`;
    return;
  }

  const xLabels = folds.map(f => f.metric_name.replace("cv_fold_", "Fold "));
  const yValues = folds.map(f => +f.metric_value.toFixed(4));
  const avg = yValues.reduce((s, v) => s + v, 0) / yValues.length;
  const std = Math.sqrt(yValues.map(v => (v - avg) ** 2).reduce((s, v) => s + v, 0) / yValues.length);

  const traces = [
    {
      x: xLabels, y: yValues, type: "bar",
      marker: { color: yValues.map(v => v >= avg ? "rgba(56,205,186,0.85)" : "rgba(239,68,68,0.75)"),
                line: { color: "rgba(255,255,255,0.12)", width: 1 } },
      text: yValues.map(v => v.toFixed(4)), textposition: "outside",
      textfont: { size: 11, color: "#c9d1d9" },
      hovertemplate: "%{x}: <b>%{y:.4f}</b><extra></extra>",
      name: "Fold Score",
    },
    {
      x: xLabels, y: xLabels.map(() => avg), type: "scatter", mode: "lines",
      line: { color: "rgba(251,191,36,0.85)", width: 2.5, dash: "dot" },
      name: `Mean ${avg.toFixed(4)}`, hovertemplate: `Mean: <b>${avg.toFixed(4)}</b><extra></extra>`,
    },
    {
      x: [...xLabels, ...xLabels.slice().reverse()],
      y: [...xLabels.map(() => avg + std), ...xLabels.map(() => avg - std)],
      type: "scatter", fill: "toself", fillcolor: "rgba(251,191,36,0.07)",
      line: { color: "transparent" }, name: `±1 STD (${std.toFixed(4)})`,
      hoverinfo: "skip",
    },
  ];

  Plotly.newPlot(el, traces, {
    ...structuredClone(_PLOTLY_LAYOUT_BASE),
    title: { text: `CV Fold Scores — ${algorithm.replace(/_/g," ")} (μ=${avg.toFixed(4)}, σ=${std.toFixed(4)})`,
             font: { color: "#c9d1d9", size: 13 }, x: 0.05 },
    yaxis: { title: "Score", titlefont: { color: "#8b949e", size: 11 }, tickfont: { color: "#c9d1d9", size: 10 },
             gridcolor: "rgba(255,255,255,0.07)",
             range: [Math.max(0, Math.min(...yValues) - 0.08), Math.min(1.06, Math.max(...yValues) + 0.06)] },
    xaxis: { tickfont: { color: "#c9d1d9", size: 11 } },
    showlegend: true, legend: { font: { color: "#8b949e", size: 11 }, bgcolor: "transparent" },
    bargap: 0.3, margin: { t: 55, b: 50, l: 65, r: 20 },
  }, { responsive: true, displayModeBar: false });
}

// ── Chart 4: All Metrics Horizontal Bar (color-coded) ───────────
function renderMetricsBars(metrics, algorithm) {
  const el = document.getElementById("viz-container-bar");
  if (!el) return;
  const pairs = Object.entries(metrics)
    .filter(([k, v]) => !_SKIP_METRICS.has(k) && typeof v === "number" && v >= 0)
    .sort((a, b) => b[1] - a[1]);

  if (!pairs.length) { el.innerHTML = `<div style="height:220px;display:flex;align-items:center;justify-content:center;color:var(--text-muted);">No metrics</div>`; return; }

  const trace = {
    x: pairs.map(([, v]) => +v.toFixed(4)),
    y: pairs.map(([k]) => _METRIC_LABELS[k] || k),
    type: "bar", orientation: "h",
    marker: {
      color: pairs.map(([, v]) => v >= 0.9 ? "rgba(52,211,153,0.88)" : v >= 0.7 ? "rgba(56,205,186,0.78)"
                                 : v >= 0.5 ? "rgba(251,191,36,0.78)" : "rgba(239,68,68,0.72)"),
      line: { color: "rgba(255,255,255,0.1)", width: 1 },
    },
    text: pairs.map(([, v]) => v.toFixed(4)), textposition: "outside",
    textfont: { size: 10, color: "#c9d1d9" },
    hovertemplate: "<b>%{y}</b>: %{x:.4f}<extra></extra>",
  };

  // Threshold line at 0.9
  const shapes = [{
    type: "line", x0: 0.9, x1: 0.9, y0: -0.5, y1: pairs.length - 0.5,
    line: { color: "rgba(52,211,153,0.4)", width: 1.5, dash: "dot" },
  }];

  el.style.minHeight = Math.max(320, pairs.length * 30 + 80) + "px";
  Plotly.newPlot(el, [trace], {
    ...structuredClone(_PLOTLY_LAYOUT_BASE),
    title: { text: `All Metrics — ${algorithm.replace(/_/g," ")}`, font: { color: "#c9d1d9", size: 13 }, x: 0.05 },
    xaxis: { range: [0, 1.18], tickfont: { color: "#c9d1d9", size: 10 }, gridcolor: "rgba(255,255,255,0.07)" },
    yaxis: { tickfont: { color: "#c9d1d9", size: 11 }, automargin: true },
    shapes, bargap: 0.22, margin: { t: 55, b: 40, l: 170, r: 70 },
  }, { responsive: true, displayModeBar: false });
}

// ── Chart 5: Feature Importance / Coefficients ───────────────────
function renderFeatureImportance(metrics, algorithm) {
  const el = document.getElementById("viz-container-feature");
  if (!el) return;

  const fi = metrics.feature_importances;
  const fc = metrics.feature_coef;
  const data = fi || fc;

  if (!data || !data.length) {
    el.innerHTML = `<div style="display:flex;align-items:center;justify-content:center;height:300px;color:var(--text-muted);font-size:14px;">
      Feature importance not available for <b style="color:var(--text-bright);">${algorithm.replace(/_/g," ")}</b><br>
      <span style="font-size:12px;">(Available for: Random Forest, Decision Tree, Gradient Boosting, Logistic Regression)</span>
    </div>`;
    return;
  }

  const isCoef = !!fc && !fi;
  const key = isCoef ? "coef" : "importance";
  const sorted = [...data].sort((a, b) => Math.abs(b[key]) - Math.abs(a[key])).slice(0, 20);
  const vals = sorted.map(d => +d[key].toFixed(6));
  const names = sorted.map(d => d.name);

  const trace = {
    x: vals,
    y: names,
    type: "bar", orientation: "h",
    marker: {
      color: isCoef
        ? vals.map(v => v >= 0 ? "rgba(56,205,186,0.85)" : "rgba(239,68,68,0.8)")
        : vals.map(v => {
            const norm = v / (Math.max(...vals) || 1);
            return `rgba(${Math.round(56+183*(1-norm))},${Math.round(205-155*(1-norm))},${Math.round(186-100*(1-norm))},0.85)`;
          }),
      line: { color: "rgba(255,255,255,0.1)", width: 1 },
    },
    text: vals.map(v => v.toFixed(4)), textposition: "outside",
    textfont: { size: 10, color: "#c9d1d9" },
    hovertemplate: `<b>%{y}</b><br>${isCoef ? "Coefficient" : "Importance"}: %{x:.6f}<extra></extra>`,
  };

  el.style.minHeight = Math.max(360, sorted.length * 28 + 100) + "px";
  Plotly.newPlot(el, [trace], {
    ...structuredClone(_PLOTLY_LAYOUT_BASE),
    title: { text: isCoef ? `Feature Coefficients — ${algorithm.replace(/_/g," ")}` :
                            `Top Feature Importances — ${algorithm.replace(/_/g," ")}`,
             font: { color: "#c9d1d9", size: 13 }, x: 0.05 },
    xaxis: { tickfont: { color: "#c9d1d9", size: 10 }, gridcolor: "rgba(255,255,255,0.07)",
             zeroline: isCoef, zerolinecolor: "rgba(255,255,255,0.2)" },
    yaxis: { tickfont: { color: "#c9d1d9", size: 11 }, automargin: true },
    bargap: 0.2, margin: { t: 55, b: 40, l: 180, r: 80 },
  }, { responsive: true, displayModeBar: false });
}

// ── Chart 6: Per-Class Grouped Bar ──────────────────────────────
function renderPerClassChart(metrics, algorithm) {
  const el = document.getElementById("viz-container-perclass");
  if (!el) return;
  const perClass = metrics.per_class;

  if (!perClass || !Object.keys(perClass).length) {
    el.innerHTML = `<div style="display:flex;align-items:center;justify-content:center;height:260px;color:var(--text-muted);">
      Per-class metrics not available (binary or regression)</div>`;
    return;
  }

  const classes = Object.keys(perClass);
  const METRIC_SETS = [
    { key: "precision", color: "rgba(56,205,186,0.85)", label: "Precision" },
    { key: "recall",    color: "rgba(99,102,241,0.85)", label: "Recall" },
    { key: "f1",        color: "rgba(251,191,36,0.85)", label: "F1" },
  ];

  const traces = METRIC_SETS.map(({ key, color, label }) => ({
    x: classes,
    y: classes.map(cls => perClass[cls]?.[key] ?? 0),
    type: "bar",
    name: label,
    marker: { color, line: { color: "rgba(255,255,255,0.1)", width: 1 } },
    hovertemplate: `<b>${label}</b><br>Class %{x}: %{y:.4f}<extra></extra>`,
  }));

  // Support bars (secondary axis)
  const supportTrace = {
    x: classes,
    y: classes.map(cls => perClass[cls]?.support ?? 0),
    type: "scatter", mode: "markers",
    yaxis: "y2",
    name: "Support",
    marker: { color: "rgba(239,68,68,0.8)", size: 10, symbol: "diamond" },
    hovertemplate: "Support: <b>%{y}</b><extra></extra>",
  };

  Plotly.newPlot(el, [...traces, supportTrace], {
    ...structuredClone(_PLOTLY_LAYOUT_BASE),
    barmode: "group",
    title: { text: `Per-Class Metrics — ${algorithm.replace(/_/g," ")}`, font: { color: "#c9d1d9", size: 13 }, x: 0.05 },
    xaxis: { title: "Class", titlefont: { color: "#8b949e", size: 11 }, tickfont: { color: "#c9d1d9", size: 11 } },
    yaxis: { title: "Score", titlefont: { color: "#8b949e", size: 11 }, tickfont: { color: "#c9d1d9", size: 10 },
             gridcolor: "rgba(255,255,255,0.07)", range: [0, 1.1] },
    yaxis2: { title: "Support", titlefont: { color: "#8b949e", size: 10 }, tickfont: { color: "#8b949e", size: 9 },
              overlaying: "y", side: "right", gridcolor: "transparent" },
    showlegend: true, legend: { font: { color: "#8b949e", size: 11 }, bgcolor: "rgba(0,0,0,0.3)",
                                bordercolor: "rgba(255,255,255,0.08)", borderwidth: 1 },
    bargap: 0.2, bargroupgap: 0.05,
    margin: { t: 55, b: 55, l: 60, r: 60 },
  }, { responsive: true, displayModeBar: false });
}

// ── Chart 7: Gauge Indicators ────────────────────────────────────
function renderGaugeCharts(metrics) {
  const el = document.getElementById("viz-container-gauge");
  if (!el) return;

  const GAUGE_METRICS = ["accuracy","f1","precision","recall","mcc","roc_auc","balanced_accuracy","specificity"];
  const available = GAUGE_METRICS.filter(k => typeof metrics[k] === "number");

  if (!available.length) {
    el.innerHTML = `<div style="display:flex;align-items:center;justify-content:center;height:220px;color:var(--text-muted);">No gauge metrics available</div>`;
    return;
  }

  el.innerHTML = "";
  el.style.display = "flex";
  el.style.flexWrap = "wrap";

  available.forEach(k => {
    const v = metrics[k];
    const div = document.createElement("div");
    div.style.cssText = "flex:0 0 200px; height:200px;";
    el.appendChild(div);

    const color = v >= 0.9 ? "#34d399" : v >= 0.7 ? "#38cdba" : v >= 0.5 ? "#fbbf24" : "#ef4444";
    Plotly.newPlot(div, [{
      type: "indicator", mode: "gauge+number+delta",
      value: +(v * 100).toFixed(2),
      title: { text: `<b>${_METRIC_LABELS[k] || k}</b>`, font: { color: "#c9d1d9", size: 11 } },
      number: { suffix: "%", font: { color: color, size: 22 }, valueformat: ".2f" },
      delta: { reference: 90, suffix: "%", font: { size: 11 } },
      gauge: {
        axis: { range: [0, 100], tickfont: { size: 9, color: "#8b949e" }, tickcolor: "#8b949e" },
        bar: { color, thickness: 0.7 },
        bgcolor: "rgba(0,0,0,0)",
        borderwidth: 0,
        steps: [
          { range: [0, 50], color: "rgba(239,68,68,0.1)" },
          { range: [50, 70], color: "rgba(251,191,36,0.1)" },
          { range: [70, 90], color: "rgba(56,205,186,0.1)" },
          { range: [90, 100], color: "rgba(52,211,153,0.15)" },
        ],
        threshold: { line: { color: "rgba(255,255,255,0.4)", width: 2 }, thickness: 0.75, value: 90 },
      },
    }], {
      paper_bgcolor: "transparent",
      font: { family: "Inter, sans-serif", color: "#8b949e" },
      margin: { t: 40, b: 10, l: 15, r: 15 },
    }, { responsive: true, displayModeBar: false });
  });
}

// ── Chart 8: Training History (across multiple train sessions) ───
function renderTrainingHistory() {
  const el = document.getElementById("viz-container-history");
  if (!el) return;
  const hist = window._vizHistory || [];

  if (hist.length < 2) {
    el.innerHTML = `<div style="display:flex;flex-direction:column;align-items:center;justify-content:center;height:260px;color:var(--text-muted);">
      <div style="font-size:32px;margin-bottom:12px;">📅</div>
      <div>Train <b style="color:var(--text-bright);">2+ models</b> to see history trends</div>
      <div style="font-size:12px;margin-top:6px;">Currently have ${hist.length} session${hist.length===1?'':'s'}</div>
    </div>`;
    return;
  }

  const TRACK = ["accuracy","f1","precision","recall","mcc","balanced_accuracy","roc_auc"];
  const available = TRACK.filter(k => hist.some(h => typeof h.metrics[k] === "number"));
  const xLabels = hist.map((h, i) => `${i+1}: ${h.algorithm.replace(/_/g," ")}<br>${h.timestamp}`);

  const traces = available.map((k, i) => ({
    x: xLabels,
    y: hist.map(h => typeof h.metrics[k] === "number" ? +h.metrics[k].toFixed(4) : null),
    type: "scatter", mode: "lines+markers",
    name: _METRIC_LABELS[k] || k,
    line: { color: _VIZ_COLORS[i % _VIZ_COLORS.length], width: 2 },
    marker: { color: _VIZ_COLORS[i % _VIZ_COLORS.length], size: 7 },
    connectgaps: false,
    hovertemplate: `<b>${_METRIC_LABELS[k]||k}</b>: %{y:.4f}<br>%{x}<extra></extra>`,
  }));

  Plotly.newPlot(el, traces, {
    ...structuredClone(_PLOTLY_LAYOUT_BASE),
    title: { text: "Training History — Metric Trends", font: { color: "#c9d1d9", size: 13 }, x: 0.05 },
    yaxis: { title: "Score", titlefont: { color: "#8b949e", size: 11 }, tickfont: { color: "#c9d1d9", size: 10 },
             gridcolor: "rgba(255,255,255,0.07)", range: [0, 1.05] },
    xaxis: { tickfont: { color: "#c9d1d9", size: 10 }, gridcolor: "rgba(255,255,255,0.04)" },
    showlegend: true, legend: { font: { color: "#8b949e", size: 11 }, bgcolor: "rgba(0,0,0,0.3)",
                                bordercolor: "rgba(255,255,255,0.08)", borderwidth: 1 },
    margin: { t: 55, b: 60, l: 65, r: 20 },
  }, { responsive: true, displayModeBar: false });
}

// ═══════════════════════════════════════════════════════════════
// COMPARE PANEL CHART TABS
// ═══════════════════════════════════════════════════════════════

const _COMPARE_TABS = ["bar","grouped","heatmap","scatter","cv","radar"];

function showCompareTab(tab) {
  const wrapMap = {
    bar: null,   // compare-chart is always visible as the container
    grouped: "compare-grouped-wrap",
    heatmap: "compare-heatmap-wrap",
    scatter: "compare-scatter-wrap",
    cv: "compare-cv-wrap",
    radar: "compare-radar-wrap",
  };

  // The single-metric bar is always present but we toggle its visibility
  const barEl = document.getElementById("compare-chart");
  if (barEl) barEl.style.display = tab === "bar" ? "" : "none";

  _COMPARE_TABS.forEach(t => {
    if (t === "bar") return;
    const wrap = document.getElementById(wrapMap[t]);
    if (wrap) wrap.style.display = t === tab ? "" : "none";
    const btn = document.getElementById(`ctab-${t}`);
    if (btn) {
      btn.style.background = t === tab ? "rgba(56,205,186,0.2)" : "";
      btn.style.color = t === tab ? "var(--accent-cyan-light)" : "";
      btn.style.borderColor = t === tab ? "var(--accent-cyan)" : "";
    }
  });
  // Also style bar button
  const barBtn = document.getElementById("ctab-bar");
  if (barBtn) {
    barBtn.style.background = tab === "bar" ? "rgba(56,205,186,0.2)" : "";
    barBtn.style.color = tab === "bar" ? "var(--accent-cyan-light)" : "";
    barBtn.style.borderColor = tab === "bar" ? "var(--accent-cyan)" : "";
  }

  // Lazy render on first open
  if (!_lastComparisonData) return;
  const results = (_lastComparisonData.results || []).filter(r => r.metrics && Object.keys(r.metrics).length > 0);
  if (tab === "heatmap") renderAlgoHeatmap(results);
  else if (tab === "scatter") renderScatterChart();
  else if (tab === "cv") renderCVStabilityChart(results);
  else if (tab === "radar") renderRadarOverlay(results);
}

// ── Compare Chart A: Algorithm × Metric Heatmap ─────────────────
function renderAlgoHeatmap(results) {
  const el = document.getElementById("compare-heatmap-chart");
  if (!el || !results.length) return;

  const PREFERRED = ["accuracy","f1","precision","recall","mcc","roc_auc","balanced_accuracy",
                     "f1_macro","specificity","sensitivity","cv_mean_accuracy"];
  const allKeys = new Set();
  results.forEach(r => Object.keys(r.metrics).forEach(k => {
    if (!_SKIP_METRICS.has(k) && typeof r.metrics[k] === "number") allKeys.add(k);
  }));
  const metricKeys = PREFERRED.filter(k => allKeys.has(k))
    .concat([...allKeys].filter(k => !PREFERRED.includes(k)));

  const algos = results.map(r => r.algorithm);
  // z: rows=metrics, cols=algos; annotate with values
  const z = metricKeys.map(k => algos.map(a => {
    const r = results.find(r => r.algorithm === a);
    return r && typeof r.metrics[k] === "number" ? +r.metrics[k].toFixed(4) : null;
  }));

  const annotations = [];
  z.forEach((row, i) => row.forEach((val, j) => {
    if (val !== null) annotations.push({
      x: algos[j], y: metricKeys[i],
      text: val.toFixed(3),
      font: { color: val > 0.7 ? "#0b101c" : "#e6edf3", size: 10, family: "monospace" },
      showarrow: false,
    });
  }));

  Plotly.newPlot(el, [{
    z, x: algos, y: metricKeys.map(k => _METRIC_LABELS[k] || k),
    type: "heatmap",
    colorscale: [[0,"rgba(239,68,68,0.3)"],[0.5,"rgba(251,191,36,0.6)"],[1,"rgba(52,211,153,0.9)"]],
    showscale: true, colorbar: { thickness: 14, tickfont: { color: "#8b949e", size: 10 } },
    hovertemplate: "<b>%{y}</b><br>%{x}: %{z:.4f}<extra></extra>",
  }], {
    ...structuredClone(_PLOTLY_LAYOUT_BASE),
    title: { text: "Algorithm × Metric Heatmap", font: { color: "#c9d1d9", size: 13 }, x: 0.05 },
    annotations,
    xaxis: { tickfont: { color: "#c9d1d9", size: 11 }, tickangle: -30 },
    yaxis: { tickfont: { color: "#c9d1d9", size: 11 }, automargin: true },
    margin: { t: 55, b: 80, l: 170, r: 80 },
  }, { responsive: true, displayModeBar: false });
}

// ── Compare Chart B: Speed vs Accuracy Scatter ──────────────────
function renderScatterChart() {
  if (!_lastComparisonData) return;
  const el = document.getElementById("compare-scatter-chart");
  if (!el) return;

  const results = (_lastComparisonData.results || []).filter(r => r.metrics && Object.keys(r.metrics).length > 0);
  if (!results.length) return;

  // Populate metric select
  const sel = document.getElementById("compare-scatter-metric");
  if (sel && !sel.options.length) {
    const allKeys = new Set();
    results.forEach(r => Object.keys(r.metrics).forEach(k => {
      if (!_SKIP_METRICS.has(k) && typeof r.metrics[k] === "number") allKeys.add(k);
    }));
    const PREFERRED = ["accuracy","f1","mcc","balanced_accuracy","roc_auc"];
    const keys = PREFERRED.filter(k => allKeys.has(k)).concat([...allKeys].filter(k => !PREFERRED.includes(k)));
    sel.innerHTML = keys.map(k => `<option value="${k}">${_METRIC_LABELS[k] || k}</option>`).join("");
  }
  const yMetric = sel ? sel.value || "accuracy" : "accuracy";

  const traces = results.map((r, i) => ({
    x: [r.training_time_sec],
    y: [typeof r.metrics[yMetric] === "number" ? r.metrics[yMetric] : 0],
    type: "scatter", mode: "markers+text",
    text: [r.algorithm.replace(/_/g," ")],
    textposition: "top center",
    textfont: { color: "#c9d1d9", size: 11 },
    marker: { color: _VIZ_COLORS[i % _VIZ_COLORS.length], size: 16,
              line: { color: "rgba(255,255,255,0.3)", width: 2 } },
    name: r.algorithm,
    hovertemplate: `<b>${r.algorithm.replace(/_/g," ")}</b><br>Time: %{x:.3f}s<br>${_METRIC_LABELS[yMetric]||yMetric}: %{y:.4f}<extra></extra>`,
  }));

  Plotly.newPlot(el, traces, {
    ...structuredClone(_PLOTLY_LAYOUT_BASE),
    title: { text: `Speed vs ${_METRIC_LABELS[yMetric]||yMetric} Trade-off`, font: { color: "#c9d1d9", size: 13 }, x: 0.05 },
    xaxis: { title: "Training Time (seconds)", titlefont: { color: "#8b949e", size: 11 },
             tickfont: { color: "#c9d1d9", size: 10 }, gridcolor: "rgba(255,255,255,0.07)" },
    yaxis: { title: _METRIC_LABELS[yMetric] || yMetric, titlefont: { color: "#8b949e", size: 11 },
             tickfont: { color: "#c9d1d9", size: 10 }, gridcolor: "rgba(255,255,255,0.07)" },
    showlegend: false,
    margin: { t: 55, b: 60, l: 70, r: 30 },
  }, { responsive: true, displayModeBar: false });
}

// ── Compare Chart C: CV Stability (Box Plots) ───────────────────
function renderCVStabilityChart(results) {
  const el = document.getElementById("compare-cv-chart");
  if (!el) return;

  const withCV = results.filter(r => Array.isArray(r.cv_scores) && r.cv_scores.length > 0);
  if (!withCV.length) {
    el.innerHTML = `<div style="display:flex;align-items:center;justify-content:center;height:300px;color:var(--text-muted);">No CV fold scores available in compare results</div>`;
    return;
  }

  const traces = withCV.map((r, i) => ({
    y: r.cv_scores,
    type: "box",
    name: r.algorithm.replace(/_/g," "),
    boxpoints: "all", jitter: 0.35, pointpos: -1.6,
    marker: { color: _VIZ_COLORS[i % _VIZ_COLORS.length], size: 6,
              line: { color: "rgba(255,255,255,0.3)", width: 1 } },
    line: { color: _VIZ_COLORS[i % _VIZ_COLORS.length], width: 2 },
    fillcolor: _VIZ_COLORS[i % _VIZ_COLORS.length].replace("0.85","0.15"),
    hovertemplate: "<b>%{x}</b><br>%{y:.4f}<extra></extra>",
  }));

  Plotly.newPlot(el, traces, {
    ...structuredClone(_PLOTLY_LAYOUT_BASE),
    title: { text: "CV Score Stability — Algorithm Comparison", font: { color: "#c9d1d9", size: 13 }, x: 0.05 },
    yaxis: { title: "CV Score", titlefont: { color: "#8b949e", size: 11 }, tickfont: { color: "#c9d1d9", size: 10 },
             gridcolor: "rgba(255,255,255,0.07)" },
    xaxis: { tickfont: { color: "#c9d1d9", size: 11 } },
    showlegend: false, boxmode: "group",
    margin: { t: 55, b: 60, l: 65, r: 20 },
  }, { responsive: true, displayModeBar: false });
}

// ── Compare Chart D: Radar Overlay (all algos on same radar) ────
function renderRadarOverlay(results) {
  const el = document.getElementById("compare-radar-chart");
  if (!el) return;

  const RADAR_KEYS = ["accuracy","precision","recall","f1","balanced_accuracy","mcc","roc_auc"];
  const available = RADAR_KEYS.filter(k =>
    results.some(r => typeof r.metrics[k] === "number" && r.metrics[k] >= 0 && r.metrics[k] <= 1)
  );

  if (available.length < 3) {
    el.innerHTML = `<div style="display:flex;align-items:center;justify-content:center;height:380px;color:var(--text-muted);">Not enough [0,1] metrics for radar overlay</div>`;
    return;
  }

  const thetaLabels = available.map(k => _METRIC_LABELS[k] || k);
  const closed = [...thetaLabels, thetaLabels[0]];

  const traces = results.map((r, i) => {
    const vals = available.map(k => typeof r.metrics[k] === "number" ? r.metrics[k] : 0);
    return {
      type: "scatterpolar",
      r: [...vals, vals[0]], theta: closed,
      fill: "toself", fillcolor: _VIZ_COLORS[i % _VIZ_COLORS.length].replace("0.85","0.08"),
      line: { color: _VIZ_COLORS[i % _VIZ_COLORS.length], width: 2 },
      marker: { size: 5 },
      name: r.algorithm.replace(/_/g," "),
      hovertemplate: "<b>%{theta}</b>: %{r:.4f}<extra></extra>",
    };
  });

  Plotly.newPlot(el, traces, {
    paper_bgcolor: "transparent",
    polar: {
      bgcolor: "rgba(0,0,0,0)",
      radialaxis: { visible: true, range: [0, 1], color: "#8b949e",
                    gridcolor: "rgba(255,255,255,0.08)", tickfont: { size: 9, color: "#8b949e" } },
      angularaxis: { tickfont: { size: 12, color: "#c9d1d9" },
                     linecolor: "rgba(255,255,255,0.12)", gridcolor: "rgba(255,255,255,0.07)" },
    },
    title: { text: "Algorithm Radar Overlay", font: { color: "#c9d1d9", size: 13 }, x: 0.05 },
    showlegend: true, legend: { font: { color: "#8b949e", size: 11 }, bgcolor: "rgba(0,0,0,0.35)",
                                bordercolor: "rgba(255,255,255,0.1)", borderwidth: 1 },
    margin: { t: 55, b: 30, l: 60, r: 60 },
  }, { responsive: true, displayModeBar: false });
}

// ── Grouped Bar (Multi-metric, existing, enhanced) ───────────────
function renderGroupedBarChart() {
  if (!_lastComparisonData) return;
  const el = document.getElementById("compare-grouped-chart");
  if (!el) return;

  const results = (_lastComparisonData.results || []).filter(r => r.metrics && Object.keys(r.metrics).length > 0);
  if (!results.length) return;

  const sel = document.getElementById("compare-grouped-metrics");
  let selectedMetrics = sel ? Array.from(sel.selectedOptions).map(o => o.value) : [];
  if (!selectedMetrics.length) {
    const PREFERRED = ["accuracy","f1","precision","recall","mcc","balanced_accuracy","roc_auc"];
    const allKeys = new Set();
    results.forEach(r => Object.keys(r.metrics).forEach(k => {
      if (!_SKIP_METRICS.has(k) && typeof r.metrics[k] === "number") allKeys.add(k);
    }));
    selectedMetrics = PREFERRED.filter(k => allKeys.has(k)).slice(0, 6);
  }
  if (!selectedMetrics.length) return;

  const algos = results.map(r => r.algorithm);
  const traces = selectedMetrics.map((metric, mi) => ({
    name: _METRIC_LABELS[metric] || metric, x: algos,
    y: algos.map(algo => {
      const r = results.find(r => r.algorithm === algo);
      return r && typeof r.metrics[metric] === "number" ? +r.metrics[metric].toFixed(4) : 0;
    }),
    type: "bar",
    marker: { color: _VIZ_COLORS[mi % _VIZ_COLORS.length],
              line: { color: "rgba(255,255,255,0.1)", width: 1 } },
    hovertemplate: `<b>${_METRIC_LABELS[metric] || metric}</b><br>%{x}: %{y:.4f}<extra></extra>`,
  }));

  Plotly.newPlot(el, traces, {
    ...structuredClone(_PLOTLY_LAYOUT_BASE),
    barmode: "group",
    title: { text: "Multi-Metric Algorithm Comparison", font: { color: "#c9d1d9", size: 13 }, x: 0.05 },
    xaxis: { tickfont: { color: "#c9d1d9", size: 11 }, tickangle: -20 },
    yaxis: { title: "Score", titlefont: { color: "#8b949e", size: 11 },
             tickfont: { color: "#c9d1d9", size: 10 }, gridcolor: "rgba(255,255,255,0.07)", range: [0, 1.12] },
    legend: { font: { color: "#8b949e", size: 11 }, bgcolor: "rgba(0,0,0,0.3)",
              bordercolor: "rgba(255,255,255,0.08)", borderwidth: 1 },
    bargap: 0.15, bargroupgap: 0.05,
    margin: { t: 55, b: 70, l: 60, r: 20 },
  }, { responsive: true, displayModeBar: false });
}

function initGroupedMetricsSelect(results) {
  const sel = document.getElementById("compare-grouped-metrics");
  if (!sel || !results.length) return;

  const allKeys = new Set();
  results.forEach(r => Object.keys(r.metrics || {}).forEach(k => {
    if (!_SKIP_METRICS.has(k) && typeof r.metrics[k] === "number") allKeys.add(k);
  }));

  const PREFERRED = ["accuracy","f1","precision","recall","mcc","balanced_accuracy","roc_auc",
                     "f1_macro","specificity","r2","mae","rmse"];
  const orderedKeys = PREFERRED.filter(k => allKeys.has(k))
    .concat([...allKeys].filter(k => !PREFERRED.includes(k)));

  sel.multiple = true;
  sel.size = Math.min(6, orderedKeys.length);
  sel.innerHTML = orderedKeys.map((k, i) =>
    `<option value="${k}" ${i < 6 ? "selected" : ""}>${_METRIC_LABELS[k] || k}</option>`
  ).join("");

  // Populate scatter metric select too
  const scatterSel = document.getElementById("compare-scatter-metric");
  if (scatterSel) {
    scatterSel.innerHTML = orderedKeys.map(k =>
      `<option value="${k}">${_METRIC_LABELS[k] || k}</option>`
    ).join("");
  }

  // Default to bar tab and render grouped
  showCompareTab("bar");
  renderGroupedBarChart();
}



// \u2500\u2500 Chart 1: Confusion Matrix Heatmap \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
function renderConfusionMatrix(metrics) {
  const el = document.getElementById("viz-container-confusion");
  if (!el) return;

  const cm = metrics.confusion_matrix;
  const labels = metrics.class_labels || (cm ? cm.map((_, i) => `Class ${i}`) : []);

  if (!cm || !cm.length) {
    el.innerHTML = `<div style="display:flex;align-items:center;justify-content:center;height:300px;color:var(--text-muted);font-size:14px;">
      Confusion matrix not available (regression task or missing data)</div>`;
    return;
  }

  // Normalize values for color intensity
  const maxVal = Math.max(...cm.flat());
  const zNorm = cm.map(row => row.map(v => v / (maxVal || 1)));

  // Annotations
  const annotations = [];
  cm.forEach((row, i) => {
    row.forEach((val, j) => {
      annotations.push({
        x: labels[j], y: labels[i],
        text: `<b>${val}</b>`,
        font: { color: val / maxVal > 0.5 ? "#0b101c" : "#e6edf3", size: 14, family: "JetBrains Mono" },
        showarrow: false,
      });
    });
  });

  const trace = {
    z: zNorm,
    x: labels.map(l => `Predicted: ${l}`),
    y: labels.map(l => `Actual: ${l}`),
    type: "heatmap",
    colorscale: [
      [0, "rgba(239,68,68,0.15)"],
      [0.5, "rgba(99,102,241,0.5)"],
      [1, "rgba(52,211,153,0.9)"],
    ],
    showscale: true,
    colorbar: { thickness: 12, len: 0.8, tickfont: { color: "#8b949e", size: 10 } },
    hoverongaps: false,
    hovertemplate: "Actual: %{y}<br>Predicted: %{x}<br>Count: <b>%{customdata}</b><extra></extra>",
    customdata: cm,
  };

  const layout = {
    ...structuredClone(_PLOTLY_LAYOUT_BASE),
    title: { text: "Confusion Matrix", font: { color: "#c9d1d9", size: 14 }, x: 0.05 },
    annotations,
    xaxis: { tickfont: { color: "#c9d1d9", size: 11 }, side: "bottom" },
    yaxis: { tickfont: { color: "#c9d1d9", size: 11 }, autorange: "reversed" },
    margin: { t: 50, b: 80, l: 100, r: 30 },
  };

  Plotly.newPlot(el, [trace], layout, { responsive: true, displayModeBar: false });
}

// \u2500\u2500 Chart 2: Metric Radar Chart \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
function renderMetricRadar(metrics, algorithm) {
  const el = document.getElementById("viz-container-radar");
  if (!el) return;

  // Only include bounded [0,1] metrics for radar meaningfulness
  const RADAR_KEYS = ["accuracy","precision","recall","f1",
                      "balanced_accuracy","mcc","roc_auc","specificity","sensitivity"];
  const available = RADAR_KEYS.filter(k => typeof metrics[k] === "number" && metrics[k] >= 0 && metrics[k] <= 1);

  if (available.length < 3) {
    el.innerHTML = `<div style="display:flex;align-items:center;justify-content:center;height:300px;color:var(--text-muted);font-size:14px;">
      Not enough bounded metrics for radar chart (need \u2265 3 metrics in [0,1])</div>`;
    return;
  }

  const labels = [...available, available[0]]; // close the polygon
  const values = available.map(k => metrics[k]);
  const r = [...values, values[0]];

  const trace = {
    type: "scatterpolar",
    r,
    theta: labels.map(k => _METRIC_LABELS[k] || k),
    fill: "toself",
    fillcolor: "rgba(56,205,186,0.15)",
    line: { color: "rgba(56,205,186,0.9)", width: 2 },
    marker: { color: "rgba(56,205,186,1)", size: 6 },
    name: algorithm,
    hovertemplate: "<b>%{theta}</b><br>Value: %{r:.4f}<extra></extra>",
  };

  const layout = {
    paper_bgcolor: "transparent",
    polar: {
      bgcolor: "rgba(0,0,0,0)",
      radialaxis: {
        visible: true, range: [0, 1],
        color: "#8b949e", gridcolor: "rgba(255,255,255,0.08)",
        tickfont: { size: 9, color: "#8b949e" },
      },
      angularaxis: {
        tickfont: { size: 11, color: "#c9d1d9" },
        linecolor: "rgba(255,255,255,0.1)",
        gridcolor: "rgba(255,255,255,0.06)",
      },
    },
    title: { text: `${algorithm.replace(/_/g," ")} — Metric Radar`, font: { color: "#c9d1d9", size: 13 }, x: 0.05 },
    showlegend: false,
    margin: { t: 50, b: 20, l: 30, r: 30 },
  };

  Plotly.newPlot(el, [trace], layout, { responsive: true, displayModeBar: false });
}

// \u2500\u2500 Chart 3: CV Fold Scores Bar Chart \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
function renderCVFoldsChart(cvRows, algorithm) {
  const el = document.getElementById("viz-container-cv");
  if (!el) return;

  const folds = cvRows.filter(e => /^cv_fold_\d+$/.test(e.metric_name))
                      .sort((a, b) => a.metric_name.localeCompare(b.metric_name));

  if (!folds.length) {
    el.innerHTML = `<div style="display:flex;align-items:center;justify-content:center;height:220px;color:var(--text-muted);">No CV fold data available</div>`;
    return;
  }

  const xLabels = folds.map(f => f.metric_name.replace("cv_fold_", "Fold "));
  const yValues = folds.map(f => +f.metric_value.toFixed(4));
  const avg = yValues.reduce((s, v) => s + v, 0) / yValues.length;

  const trace = {
    x: xLabels, y: yValues, type: "bar",
    marker: {
      color: yValues.map(v => v >= avg ? "rgba(56,205,186,0.85)" : "rgba(239,68,68,0.7)"),
      line: { color: "rgba(255,255,255,0.1)", width: 1 },
    },
    text: yValues.map(v => v.toFixed(4)),
    textposition: "outside",
    textfont: { size: 11, color: "#c9d1d9" },
    hovertemplate: "%{x}: <b>%{y:.4f}</b><extra></extra>",
  };

  // Average line
  const avgLine = {
    type: "scatter", mode: "lines",
    x: xLabels, y: xLabels.map(() => avg),
    line: { color: "rgba(251,191,36,0.8)", width: 2, dash: "dot" },
    name: `Mean: ${avg.toFixed(4)}`,
    hovertemplate: "Mean: <b>" + avg.toFixed(4) + "</b><extra></extra>",
  };

  const layout = {
    ...structuredClone(_PLOTLY_LAYOUT_BASE),
    title: { text: "Cross-Validation Fold Scores", font: { color: "#c9d1d9", size: 13 }, x: 0.05 },
    yaxis: {
      title: "Score", titlefont: { color: "#8b949e", size: 11 },
      tickfont: { color: "#c9d1d9", size: 10 },
      gridcolor: "rgba(255,255,255,0.07)",
      range: [Math.max(0, Math.min(...yValues) - 0.05), Math.min(1.05, Math.max(...yValues) + 0.05)],
    },
    xaxis: { tickfont: { color: "#c9d1d9", size: 11 }, gridcolor: "rgba(255,255,255,0.04)" },
    showlegend: true,
    legend: { font: { color: "#8b949e", size: 11 }, bgcolor: "transparent" },
    bargap: 0.3,
    margin: { t: 50, b: 50, l: 60, r: 20 },
  };

  Plotly.newPlot(el, [trace, avgLine], layout, { responsive: true, displayModeBar: false });
}

// \u2500\u2500 Chart 4: All Metrics Horizontal Bar \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
function renderMetricsBars(metrics, algorithm) {
  const el = document.getElementById("viz-container-bar");
  if (!el) return;

  const pairs = Object.entries(metrics)
    .filter(([k, v]) => !_SKIP_METRICS.has(k) && typeof v === "number" && v >= 0)
    .sort((a, b) => b[1] - a[1]);

  if (!pairs.length) {
    el.innerHTML = `<div style="display:flex;align-items:center;justify-content:center;height:220px;color:var(--text-muted);">No scalar metrics available</div>`;
    return;
  }

  const trace = {
    x: pairs.map(([, v]) => +v.toFixed(4)),
    y: pairs.map(([k]) => _METRIC_LABELS[k] || k),
    type: "bar",
    orientation: "h",
    marker: {
      color: pairs.map(([, v]) => {
        if (v >= 0.9) return "rgba(52,211,153,0.85)";
        if (v >= 0.7) return "rgba(56,205,186,0.75)";
        if (v >= 0.5) return "rgba(251,191,36,0.75)";
        return "rgba(239,68,68,0.7)";
      }),
      line: { color: "rgba(255,255,255,0.08)", width: 1 },
    },
    text: pairs.map(([, v]) => v.toFixed(4)),
    textposition: "outside",
    textfont: { size: 10, color: "#c9d1d9" },
    hovertemplate: "<b>%{y}</b>: %{x:.4f}<extra></extra>",
  };

  const layout = {
    ...structuredClone(_PLOTLY_LAYOUT_BASE),
    title: { text: `All Metrics — ${algorithm.replace(/_/g," ")}`, font: { color: "#c9d1d9", size: 13 }, x: 0.05 },
    xaxis: {
      range: [0, 1.12], tickfont: { color: "#c9d1d9", size: 10 },
      gridcolor: "rgba(255,255,255,0.07)",
    },
    yaxis: {
      tickfont: { color: "#c9d1d9", size: 11 },
      automargin: true,
    },
    bargap: 0.25,
    margin: { t: 50, b: 40, l: 160, r: 60 },
  };

  const height = Math.max(280, pairs.length * 32 + 80);
  el.style.minHeight = height + "px";
  Plotly.newPlot(el, [trace], layout, { responsive: true, displayModeBar: false });
}

// \u2500\u2500 Chart 5: Multi-Metric Grouped Bar (Compare) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
function renderGroupedBarChart() {
  if (!_lastComparisonData) return;
  const el = document.getElementById("compare-grouped-chart");
  if (!el) return;

  const results = (_lastComparisonData.results || []).filter(r => r.metrics && Object.keys(r.metrics).length > 0);
  if (!results.length) return;

  // Figure out which metrics to show
  const sel = document.getElementById("compare-grouped-metrics");
  let selectedMetrics = sel ? Array.from(sel.selectedOptions).map(o => o.value) : [];
  if (!selectedMetrics.length) {
    // Default: top 6 preferred metrics that exist
    const PREFERRED = ["accuracy","f1","precision","recall","mcc","balanced_accuracy","roc_auc"];
    const allKeys = new Set();
    results.forEach(r => Object.keys(r.metrics).forEach(k => {
      if (!_SKIP_METRICS.has(k) && typeof r.metrics[k] === "number") allKeys.add(k);
    }));
    selectedMetrics = PREFERRED.filter(k => allKeys.has(k)).slice(0, 6);
  }

  if (!selectedMetrics.length) return;

  const algorithms = results.map(r => r.algorithm);
  const traces = selectedMetrics.map((metric, mi) => ({
    name: _METRIC_LABELS[metric] || metric,
    x: algorithms,
    y: algorithms.map(algo => {
      const r = results.find(r => r.algorithm === algo);
      return r && typeof r.metrics[metric] === "number" ? +r.metrics[metric].toFixed(4) : 0;
    }),
    type: "bar",
    marker: { color: _VIZ_COLORS[mi % _VIZ_COLORS.length] },
    hovertemplate: `<b>${_METRIC_LABELS[metric] || metric}</b><br>%{x}: %{y:.4f}<extra></extra>`,
  }));

  const layout = {
    ...structuredClone(_PLOTLY_LAYOUT_BASE),
    barmode: "group",
    title: { text: "Algorithm Comparison \u2014 Multiple Metrics", font: { color: "#c9d1d9", size: 13 }, x: 0.05 },
    xaxis: { tickfont: { color: "#c9d1d9", size: 11 }, gridcolor: "rgba(255,255,255,0.04)" },
    yaxis: {
      title: "Score", titlefont: { color: "#8b949e", size: 11 },
      tickfont: { color: "#c9d1d9", size: 10 },
      gridcolor: "rgba(255,255,255,0.07)",
      range: [0, 1.1],
    },
    legend: { font: { color: "#8b949e", size: 11 }, bgcolor: "rgba(0,0,0,0.3)",
              bordercolor: "rgba(255,255,255,0.08)", borderwidth: 1 },
    bargap: 0.15, bargroupgap: 0.05,
    margin: { t: 50, b: 70, l: 55, r: 20 },
  };

  Plotly.newPlot(el, traces, layout, { responsive: true, displayModeBar: false });
}

// Populate the grouped-metrics multi-select and render on compare complete
function initGroupedMetricsSelect(results) {
  const sel = document.getElementById("compare-grouped-metrics");
  if (!sel || !results.length) return;

  const allKeys = new Set();
  results.forEach(r => Object.keys(r.metrics || {}).forEach(k => {
    if (!_SKIP_METRICS.has(k) && typeof r.metrics[k] === "number") allKeys.add(k);
  }));

  const PREFERRED = ["accuracy","f1","precision","recall","mcc","balanced_accuracy","roc_auc",
                     "f1_macro","specificity","r2","mae","rmse"];
  const orderedKeys = PREFERRED.filter(k => allKeys.has(k))
    .concat([...allKeys].filter(k => !PREFERRED.includes(k)));

  // Change to a proper multi-select with checkboxes via size
  sel.multiple = true;
  sel.size = Math.min(5, orderedKeys.length);
  sel.innerHTML = orderedKeys.map((k, i) =>
    `<option value="${k}" ${i < 6 ? "selected" : ""}>${_METRIC_LABELS[k] || k}</option>`
  ).join("");

  renderGroupedBarChart();
}



async function refreshModelsList() {
  try {
    const res = await fetch("/api/models");
    if (!res.ok) return;
    state.models = await res.json();
    const countEl = document.getElementById("kpi-models-count");
    if (countEl) countEl.innerText = state.models.length;

    const sel = document.getElementById("predict-model-select");
    if (sel) {
      sel.innerHTML = '<option value="">Select a saved model...</option>';
      state.models.forEach(m => {
        const opt = document.createElement("option");
        opt.value = m.id;
        opt.textContent = `${m.model_name} [${m.algorithm}]`;
        sel.appendChild(opt);
      });

      if (state.models.length > 0 && !sel.value) {
        sel.value = state.models[0].id;
        renderPredictionForm(state.models[0].id);
      }
    }
  } catch (err) {
    console.error(err);
  }
}

async function renderPredictionForm(modelId) {
  if (!modelId) return;
  const container = document.getElementById("prediction-fields-container");
  if (!container) return;
  container.innerHTML = "";

  const modelMeta = state.models.find(m => m.id === modelId);
  state.selectedModel = modelMeta;

  // Use features from active dataset or defaults
  const features = state.previewData?.columns?.filter(c => c.toLowerCase() !== "target" && c.toLowerCase() !== "survived") || ["feature_1", "feature_2"];

  features.forEach(f => {
    const group = document.createElement("div");
    group.className = "form-group";
    group.innerHTML = `
      <label class="form-label">${f}</label>
      <input type="number" step="any" class="form-input pred-input" data-col="${f}" placeholder="Enter value..." required>
    `;
    container.appendChild(group);
  });
}

async function runPrediction() {
  const sel = document.getElementById("predict-model-select");
  const modelId = sel ? sel.value : "";
  if (!modelId) return showToast("Select a model first", "info");

  const inputs = {};
  document.querySelectorAll(".pred-input").forEach(inp => {
    inputs[inp.getAttribute("data-col")] = parseFloat(inp.value) || 0.0;
  });

  try {
    const res = await fetch("/api/ml/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model_id: modelId,
        input_data: inputs,
      }),
    });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();

    const box = document.getElementById("prediction-result-box");
    if (box) {
      box.innerHTML = `
        <div style="background:var(--bg-card); border:1px solid var(--border-card); border-radius:var(--radius-lg); padding:32px 20px; box-shadow:var(--shadow-md);">
          <div style="font-size:36px; margin-bottom:8px;">🎯</div>
          <div class="kpi-label">Predicted Telemetry Output</div>
          <div style="font-size:42px; font-weight:800; color:var(--accent-emerald-light); font-family:var(--font-mono); margin:12px 0; text-shadow:0 0 20px rgba(16, 185, 129, 0.4);">
            ${data.prediction}
          </div>
          ${data.confidence !== undefined && data.confidence !== null ? `
            <div style="margin-top:14px; display:inline-block;">
              <span class="badge badge-info" style="font-size:12px; padding:6px 14px;">
                Confidence Score: ${(data.confidence * 100).toFixed(1)}%
              </span>
            </div>
          ` : ''}
        </div>
      `;
    }
    showToast("Inference executed successfully!", "success");
  } catch (err) {
    showToast(`Prediction failed: ${err.message}`, "error");
  }
}

// ── Reports ─────────────────────────────────────────────────────
async function generateReport() {
  if (!state.activeDatasetId) return showToast("Select a dataset first", "info");
  const rtype = document.getElementById("rep-type").value;
  const fmt = document.getElementById("rep-format").value;

  showToast(`Generating ${rtype} report in ${fmt.toUpperCase()} format...`, "info");

  try {
    const res = await fetch("/api/reports/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        report_type: rtype,
        format: fmt,
        dataset_id: state.activeDatasetId,
      }),
    });
    if (!res.ok) throw new Error(await res.text());
    const rep = await res.json();
    showToast("Report generated successfully!", "success");
    await refreshReportsList();
  } catch (err) {
    showToast(`Report generation failed: ${err.message}`, "error");
  }
}

async function refreshReportsList() {
  try {
    const res = await fetch("/api/reports");
    if (!res.ok) return;
    const reports = await res.json();

    const tbody = reports.map(r => `
      <tr>
        <td><code>${r.id}</code></td>
        <td><span class="badge badge-info">${r.report_type}</span></td>
        <td><strong style="color:var(--text-bright);">${r.format.toUpperCase()}</strong></td>
        <td style="font-family:var(--font-mono);">${(r.file_size / 1024).toFixed(1)} KB</td>
        <td style="font-family:var(--font-mono); color:var(--text-dim);">${new Date(r.created_at).toLocaleString()}</td>
        <td>
          <a class="btn btn-secondary btn-sm" href="/api/reports/${r.id}/download" target="_blank">⬇ Download</a>
        </td>
      </tr>
    `).join("");
    const repTable = document.querySelector("#reports-catalog-table tbody");
    if (repTable) repTable.innerHTML = tbody;
  } catch (err) {
    console.error(err);
  }
}

// ═══════════════════════════════════════════════════════════════════
// Pipeline Auditor & Leakage Detection Controller
// ═══════════════════════════════════════════════════════════════════

async function runPipelineAudit() {
  if (!state.activeDatasetId) {
    showToast("Please select a dataset from the Datasets Hub first!", "error");
    showTab("datasets");
    return;
  }

  const btn = document.getElementById("btn-run-audit");
  const model = document.getElementById("auditor-model-select")?.value || "random_forest";
  const targetSel = document.getElementById("ml-target-col");
  const targetCol = targetSel ? targetSel.value : "";

  if (btn) {
    btn.disabled = true;
    btn.innerHTML = `<span class="spinner" style="width:14px;height:14px;border-width:2px;display:inline-block;margin-right:6px;"></span> Auditing Pipeline...`;
  }

  try {
    let url = `/api/auditor/audit/${state.activeDatasetId}?model_name=${encodeURIComponent(model)}`;
    if (targetCol) url += `&target_column=${encodeURIComponent(targetCol)}`;

    const res = await fetch(url, { method: "POST" });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();

    // 1. Update KPIs
    const scoreVal = document.getElementById("auditor-integrity-val");
    if (scoreVal) scoreVal.innerText = `${data.integrity_score ?? 100} / 100`;

    const badge = document.getElementById("auditor-integrity-badge");
    if (badge) {
      const score = data.integrity_score ?? 100;
      if (score >= 80) {
        badge.innerText = "SOUND EVALUATION";
        badge.className = "badge badge-valid";
      } else if (score >= 50) {
        badge.innerText = "MODERATE DEFECT RISK";
        badge.className = "badge badge-warning";
      } else {
        badge.innerText = "CRITICAL LEAKAGE RISK";
        badge.className = "badge badge-invalid";
      }
    }

    const confVal = document.getElementById("auditor-confirmed-val");
    if (confVal) confVal.innerText = data.summary?.confirmed_count ?? 0;
    const critVal = document.getElementById("auditor-critical-val");
    if (critVal) critVal.innerText = `${data.summary?.critical_count ?? 0} critical defects`;

    const suspVal = document.getElementById("auditor-suspected-val");
    if (suspVal) suspVal.innerText = data.summary?.suspected_count ?? 0;
    const warnVal = document.getElementById("auditor-warning-val");
    if (warnVal) warnVal.innerText = `${data.summary?.high_count ?? 0} high / warnings`;

    const metricVal = document.getElementById("auditor-metric-val");
    const metricLabel = document.getElementById("auditor-metric-label");
    const bm = data.baseline_metrics || {};
    const firstKey = Object.keys(bm)[0];
    if (metricVal && firstKey) {
      metricVal.innerText = typeof bm[firstKey] === "number" ? bm[firstKey].toFixed(3) : bm[firstKey];
      if (metricLabel) metricLabel.innerText = `${firstKey.toUpperCase()} (Baseline)`;
    }

    // 2. Render Findings
    state.allAuditFindings = data.findings || [];
    renderAuditFindings(state.allAuditFindings);

    showToast("Pipeline audit complete! Evaluation findings identified.", "success");
  } catch (err) {
    showToast(`Audit failed: ${err.message}`, "error");
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.innerHTML = "▶ Run Deep Pipeline Audit";
    }
  }
}

function filterAuditFindings(filter) {
  if (!state.allAuditFindings) return;
  if (filter === "all") {
    renderAuditFindings(state.allAuditFindings);
  } else if (filter === "critical" || filter === "high") {
    renderAuditFindings(state.allAuditFindings.filter(f => f.severity === filter));
  } else {
    renderAuditFindings(state.allAuditFindings.filter(f => (f.category || "").toLowerCase().includes(filter)));
  }
}

function renderAuditFindings(findings) {
  const container = document.getElementById("auditor-findings-container");
  if (!container) return;

  if (!findings || findings.length === 0) {
    container.innerHTML = `
      <div style="text-align:center; padding:30px 10px; color:var(--text-muted); background:rgba(255,255,255,0.02); border-radius:var(--radius-sm);">
        <div style="font-size:32px; margin-bottom:8px;">✅</div>
        <strong style="color:var(--text-bright); font-size:14px; display:block; margin-bottom:4px;">No Flaws Found For This Filter</strong>
        <p style="font-size:12px; margin:0;">The evaluated pipeline satisfies all verification criteria without detected leakage.</p>
      </div>
    `;
    return;
  }

  container.innerHTML = findings.map(f => {
    const sev = f.severity || "medium";
    const sevClass = sev === "critical" ? "badge-critical" : (sev === "high" ? "badge-warning" : "badge-info");
    const confPct = f.confidence ? Math.round(f.confidence * 100) : 95;
    const cols = (f.affected_columns && f.affected_columns.length)
      ? f.affected_columns.map(c => `<code style="font-size:11px;">${c}</code>`).join(" ")
      : '<span style="color:var(--text-muted); font-size:11px;">Pipeline Split</span>';

    return `
      <div class="eda-insight-card ${sev}" style="display:flex; flex-direction:column; gap:8px;">
        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;">
          <div style="display:flex; align-items:center; gap:8px;">
            <span class="badge ${sevClass}">${sev.toUpperCase()}</span>
            <strong style="color:var(--text-bright); font-size:14px;">${f.category ? f.category.replace(/_/g, ' ').toUpperCase() : 'AUDIT FINDING'}</strong>
            <span class="badge" style="background:rgba(255,255,255,0.06); color:var(--text-muted); font-size:10px;">${confPct}% Confidence</span>
            ${f.auto_correctable ? '<span class="badge badge-success" style="font-size:10px;">⚡ Auto-Correctable</span>' : ''}
          </div>
          <span style="font-size:11px; color:var(--text-muted); font-family:var(--font-mono);">${f.finding_id || ''}</span>
        </div>
        <p style="color:var(--text-main); font-size:13px; margin:2px 0 6px 0; line-height:1.5;">${f.description || ''}</p>
        <div style="display:flex; gap:16px; flex-wrap:wrap; font-size:12px; color:var(--text-muted); background:var(--bg-card-hover); border:1px solid var(--border-subtle); padding:8px 12px; border-radius:var(--radius-sm);">
          <div><strong>Affected Features:</strong> ${cols}</div>
          ${f.recommendation ? `<div><strong>Action:</strong> <span style="color:var(--accent-cyan);">${f.recommendation}</span></div>` : ''}
        </div>
        ${f.auto_correctable ? `
          <div style="margin-top:6px; display:flex; justify-content:flex-end;">
            <button class="btn btn-secondary btn-sm" onclick="showTab('repair'); runPipelineRepair();" style="font-size:11px; padding:4px 10px;">
              ⚡ Fix in Pipeline Repair →
            </button>
          </div>
        ` : ''}
      </div>
    `;
  }).join("");
}

function exportAuditFindingsCSV() {
  window.open("/api/auditor/findings/export", "_blank");
}

// ═══════════════════════════════════════════════════════════════════
// Pipeline Repair Controller
// ═══════════════════════════════════════════════════════════════════

async function runPipelineRepair() {
  if (!state.activeDatasetId) {
    showToast("Please select a dataset from the Datasets Hub first!", "error");
    showTab("datasets");
    return;
  }

  const btn = document.getElementById("btn-run-repair");
  if (btn) {
    btn.disabled = true;
    btn.innerHTML = `<span class="spinner" style="width:14px;height:14px;border-width:2px;display:inline-block;margin-right:6px;"></span> Repairing Pipeline...`;
  }

  try {
    const res = await fetch(`/api/auditor/repair/${state.activeDatasetId}`, { method: "POST" });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();

    const banner = document.getElementById("repair-status-banner");
    if (banner) {
      banner.innerHTML = `<strong style="color:var(--accent-emerald-light);">Status: ${data.status.toUpperCase()}</strong> — ${data.message || ''} (${data.resolved ?? 0} defects resolved, ${data.unresolved ?? 0} unresolved).`;
    }

    // Actions count badge
    const actBadge = document.getElementById("repair-actions-count");
    if (actBadge) actBadge.innerText = `${(data.actions || []).length} Actions Applied`;

    // Render Actions List
    const actionsList = document.getElementById("repair-actions-list");
    if (actionsList) {
      if (!data.actions || data.actions.length === 0) {
        actionsList.innerHTML = `<span style="color:var(--text-muted); font-size:13px;">No repair actions were required. The pipeline already adheres to leakage-free evaluation protocols.</span>`;
      } else {
        actionsList.innerHTML = data.actions.map((act, i) => `
          <div style="background:var(--bg-card-hover); border:1px solid var(--border-subtle); border-radius:var(--radius-sm); padding:10px 14px; display:flex; justify-content:space-between; align-items:center;">
            <div>
              <strong style="color:var(--text-bright); font-size:13px;">#${i + 1} — ${act.action_type || act.category || 'Correction'}</strong>
              <div style="color:var(--text-muted); font-size:12px; margin-top:2px;">${act.description || JSON.stringify(act)}</div>
            </div>
            <span class="badge badge-success">Applied</span>
          </div>
        `).join("");
      }
    }

    // Render Metrics Before vs After
    renderMetricComparisonList("repair-before-metrics-list", data.before_metrics || {});
    renderMetricComparisonList("repair-after-metrics-list", data.after_metrics || {});

    showToast("Pipeline repair successfully executed!", "success");
  } catch (err) {
    showToast(`Pipeline repair failed: ${err.message}`, "error");
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.innerHTML = "⚡ Apply 1-Click Pipeline Repair";
    }
  }
}

function renderMetricComparisonList(containerId, metrics) {
  const box = document.getElementById(containerId);
  if (!box) return;

  const entries = Object.entries(metrics).filter(([k, v]) => typeof v === "number" || typeof v === "string");
  if (entries.length === 0) {
    box.innerHTML = `<span style="color:var(--text-muted); font-size:13px;">No metric telemetry available.</span>`;
    return;
  }

  box.innerHTML = entries.map(([k, v]) => `
    <div style="display:flex; justify-content:space-between; align-items:center; padding:8px 12px; background:var(--bg-card-hover); border:1px solid var(--border-subtle); border-radius:var(--radius-sm);">
      <span style="color:var(--text-muted); font-size:12px; text-transform:uppercase; font-family:var(--font-mono);">${k.replace(/_/g, ' ')}</span>
      <strong style="color:var(--text-bright); font-family:var(--font-mono); font-size:13px;">${typeof v === "number" ? v.toFixed(4) : v}</strong>
    </div>
  `).join("");
}

// ═══════════════════════════════════════════════════════════════════
// Auditor Benchmark Lab Controller
// ═══════════════════════════════════════════════════════════════════

async function runBenchmarkLab() {
  const btn = document.getElementById("btn-run-benchmark");
  const model = document.getElementById("benchmark-model-select")?.value || "random_forest";

  if (btn) {
    btn.disabled = true;
    btn.innerHTML = `<span class="spinner" style="width:14px;height:14px;border-width:2px;display:inline-block;margin-right:6px;"></span> Evaluating Benchmark...`;
  }

  try {
    const payload = {
      dataset_id: state.activeDatasetId || null,
      model_name: model,
    };

    const res = await fetch("/api/auditor/benchmark", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    const overall = data.overall || {};

    // KPIs
    const prec = document.getElementById("bm-precision-val");
    if (prec) prec.innerText = `${Math.round((overall.overall_precision || 0) * 100)}%`;

    const rec = document.getElementById("bm-recall-val");
    if (rec) rec.innerText = `${Math.round((overall.overall_recall || 0) * 100)}%`;

    const f1 = document.getElementById("bm-f1-val");
    if (f1) f1.innerText = `${Math.round((overall.overall_f1 || 0) * 100)}%`;

    const scen = document.getElementById("bm-scenarios-val");
    if (scen) scen.innerText = `${overall.scenario_count ?? (data.scenarios || []).length} Scenarios`;

    const tm = document.getElementById("bm-time-val");
    if (tm) tm.innerText = `${overall.total_time ?? 0}s duration`;

    // Table
    const tbody = (data.scenarios || []).map(s => `
      <tr>
        <td><strong style="color:var(--text-bright);">${s.name}</strong></td>
        <td><span class="badge badge-info">${s.type}</span></td>
        <td style="font-family:var(--font-mono); font-size:11px; color:var(--text-muted);">${(s.ground_truth || []).join(", ") || 'None'}</td>
        <td style="font-family:var(--font-mono); font-size:11px; color:var(--accent-cyan);">${(s.detected || []).join(", ") || 'None'}</td>
        <td style="font-family:var(--font-mono);">${(s.precision * 100).toFixed(0)}%</td>
        <td style="font-family:var(--font-mono);">${(s.recall * 100).toFixed(0)}%</td>
        <td><strong style="color:#34d399; font-family:var(--font-mono);">${(s.f1 * 100).toFixed(0)}%</strong></td>
      </tr>
    `).join("");

    const tableBody = document.querySelector("#benchmark-scenarios-table tbody");
    if (tableBody) tableBody.innerHTML = tbody || '<tr><td colspan="7" style="text-align:center;">No scenarios returned.</td></tr>';

    showToast("Synthetic benchmark evaluation complete!", "success");
  } catch (err) {
    showToast(`Benchmark failed: ${err.message}`, "error");
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.innerHTML = "▶ Run Synthetic Benchmark";
    }
  }
}

