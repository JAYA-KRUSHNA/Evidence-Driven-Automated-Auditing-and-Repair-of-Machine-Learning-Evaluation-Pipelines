/**
 * Universal AI/ML Studio — Client Dashboard Controller
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

// ── Toast Notifications ─────────────────────────────────────────
function showToast(message, type = "info") {
  const container = document.getElementById("toast-container");
  const toast = document.createElement("div");
  toast.className = `toast ${type}`;
  toast.innerHTML = `<span>${type === 'success' ? '✅' : type === 'error' ? '❌' : 'ℹ️'}</span><span>${message}</span>`;
  container.appendChild(toast);
  setTimeout(() => {
    toast.style.opacity = '0';
    setTimeout(() => toast.remove(), 300);
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
  };
  document.getElementById("page-heading").innerText = titles[tabName] || "Studio Dashboard";
}

document.querySelectorAll(".nav-item").forEach(item => {
  item.addEventListener("click", () => {
    const tab = item.getAttribute("data-tab");
    showTab(tab);
  });
});

// ── Init on Page Load ───────────────────────────────────────────
document.addEventListener("DOMContentLoaded", async () => {
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
    container.innerHTML = "";

    samples.forEach(s => {
      const card = document.createElement("div");
      card.className = "sample-card";
      card.innerHTML = `
        <div>
          <div class="sample-title">
            <span>${s.display_name}</span>
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
  if (selector.value !== datasetId) selector.value = datasetId;

  try {
    const res = await fetch(`/api/datasets/${datasetId}?preview_rows=10`);
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    state.activeDatasetMeta = data.metadata;
    state.previewData = data;

    // Update Topbar and Overview
    document.getElementById("active-dataset-name").innerText = data.metadata.name;
    document.getElementById("kpi-dataset-name").innerText = data.metadata.name;
    document.getElementById("kpi-dimensions").innerText = `${data.shape.rows} × ${data.shape.columns}`;
    document.getElementById("kpi-validation-status").innerText = data.metadata.validation_status || "Pending";

    // Populate Target column selector
    const targetSel = document.getElementById("ml-target-col");
    targetSel.innerHTML = '<option value="">Select target column...</option>';
    data.columns.forEach(col => {
      const opt = document.createElement("option");
      opt.value = col;
      opt.textContent = col;
      if (col.toLowerCase() === "target" || col.toLowerCase() === "survived") opt.selected = true;
      targetSel.appendChild(opt);
    });

    renderDatasetPreviewTable(data.columns, data.preview);
    if (targetSel.value) detectTaskType(targetSel.value);
  } catch (err) {
    showToast(`Error loading dataset details: ${err.message}`, "error");
  }
}

function renderDatasetPreviewTable(columns, rows) {
  const table = document.getElementById("dataset-preview-table");
  let thead = "<thead><tr>" + columns.map(c => `<th>${c}</th>`).join("") + "</tr></thead>";
  let tbody = "<tbody>";
  rows.forEach(r => {
    tbody += "<tr>" + columns.map(c => `<td>${r[c] !== null && r[c] !== undefined ? r[c] : '<span style="color:var(--text-dim);">null</span>'}</td>`).join("") + "</tr>";
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
    const targetCol = document.getElementById("ml-target-col").value;
    const url = `/api/validation/run/${state.activeDatasetId}${targetCol ? `?target_column=${encodeURIComponent(targetCol)}` : ''}`;
    const res = await fetch(url, { method: "POST" });
    if (!res.ok) throw new Error(await res.text());
    const val = await res.json();

    document.getElementById("val-status").innerText = val.status.toUpperCase();
    document.getElementById("val-total").innerText = val.total_checks;
    document.getElementById("val-passed").innerText = val.passed_checks;
    document.getElementById("val-warnings").innerText = val.warning_count;
    document.getElementById("val-failed").innerText = val.failed_checks;
    document.getElementById("kpi-validation-status").innerText = val.status;

    // Render checks list
    const list = document.getElementById("validation-checks-list");
    list.innerHTML = "";
    val.checks.forEach(c => {
      const card = document.createElement("div");
      card.className = "check-card";
      const badgeCls = c.status === "passed" ? "badge-valid" : c.status === "warning" ? "badge-warning" : "badge-invalid";
      card.innerHTML = `
        <div class="check-left">
          <span class="badge ${badgeCls}">${c.status.toUpperCase()}</span>
          <div>
            <div class="check-msg"><strong>${c.check_name}</strong>: ${c.message}</div>
            <div class="check-sub">Category: ${c.tier} ${c.suggestion ? `• Suggestion: ${c.suggestion}` : ''}</div>
          </div>
        </div>
      `;
      list.appendChild(card);
    });

    showToast("Validation completed!", "success");
  } catch (err) {
    showToast(`Validation error: ${err.message}`, "error");
  }
}

// ── EDA ─────────────────────────────────────────────────────────
async function runEDA() {
  if (!state.activeDatasetId) return showToast("Select a dataset first", "info");
  showToast("Computing full Exploratory Data Analysis...", "info");

  try {
    const res = await fetch(`/api/eda/run/${state.activeDatasetId}`, { method: "POST" });
    if (!res.ok) throw new Error(await res.text());
    const payload = await res.json();
    renderEDAResults(payload.eda);
    showToast("EDA analysis ready!", "success");
  } catch (err) {
    showToast(`EDA failed: ${err.message}`, "error");
  }
}

function renderEDAResults(eda) {
  // Statistics
  const numStats = eda.statistics?.numerical || {};
  const tbody = Object.entries(numStats).map(([col, s]) => `
    <tr>
      <td><strong>${col}</strong></td>
      <td>${s.mean?.toFixed(2) ?? '-'}</td>
      <td>${s.std?.toFixed(2) ?? '-'}</td>
      <td>${s.min?.toFixed(2) ?? '-'}</td>
      <td>${s['25%']?.toFixed(2) ?? '-'}</td>
      <td>${s['50%']?.toFixed(2) ?? '-'}</td>
      <td>${s['75%']?.toFixed(2) ?? '-'}</td>
      <td>${s.max?.toFixed(2) ?? '-'}</td>
      <td>${s.skewness?.toFixed(2) ?? '-'}</td>
    </tr>
  `).join("");
  document.querySelector("#eda-stats-table tbody").innerHTML = tbody;

  // Correlation Matrix
  const corr = eda.correlation?.matrix || {};
  const cols = Object.keys(corr);
  if (cols.length > 0) {
    const thead = "<thead><tr><th>Feature</th>" + cols.map(c => `<th>${c}</th>`).join("") + "</tr></thead>";
    const tbodyCorr = "<tbody>" + cols.map(rowCol => {
      return `<tr><td><strong>${rowCol}</strong></td>` + cols.map(colCol => {
        const val = corr[rowCol]?.[colCol] ?? 0;
        const color = val > 0.7 ? '#34d399' : val < -0.5 ? '#f87171' : 'var(--text-main)';
        return `<td style="color:${color}">${val.toFixed(2)}</td>`;
      }).join("") + "</tr>";
    }).join("") + "</tbody>";
    document.getElementById("eda-corr-table").innerHTML = thead + tbodyCorr;
  }
}

// ── Preprocessing ───────────────────────────────────────────────
async function runAutoPreprocess() {
  if (!state.activeDatasetId) return showToast("Select a dataset first", "info");
  showToast("Running automated preprocessing pipeline...", "info");

  try {
    const targetCol = document.getElementById("ml-target-col").value;
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
        <td><strong>${h.operation}</strong></td>
        <td><span class="badge badge-info">${h.category}</span></td>
        <td>${(h.affected_columns || []).join(", ") || 'All'}</td>
        <td>${h.rows_before}</td>
        <td>${h.rows_after}</td>
        <td>${new Date(h.applied_at).toLocaleTimeString()}</td>
      </tr>
    `).join("");
    document.querySelector("#prep-history-table tbody").innerHTML = tbody;
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
    badge.innerText = `Task: ${info.task_type.toUpperCase()} (${info.nunique} unique values, ${info.confidence} confidence)`;
    badge.className = `badge ${info.task_type === 'classification' ? 'badge-info' : 'badge-algo'}`;

    // Populate algorithm selector
    const algoSel = document.getElementById("ml-algorithm-select");
    algoSel.innerHTML = "";
    const algos = state.algorithms[info.task_type] || [];
    algos.forEach(a => {
      const opt = document.createElement("option");
      opt.value = a;
      opt.textContent = a.replace(/_/g, " ").toUpperCase();
      algoSel.appendChild(opt);
    });
  } catch (err) {
    console.error(err);
  }
}

async function trainSingleModel() {
  if (!state.activeDatasetId) return showToast("Select a dataset first", "info");
  const targetCol = document.getElementById("ml-target-col").value;
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

    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    showToast("Model trained successfully!", "success");

    // Display result card
    const evals = data.evaluations || [];
    const metricsHtml = evals.map(ev => `
      <div style="display:flex; justify-content:space-between; padding:8px 0; border-bottom:1px solid rgba(255,255,255,0.05);">
        <span style="color:var(--text-muted);">${ev.metric_name}:</span>
        <strong style="color:var(--accent-cyan);">${Number(ev.metric_value).toFixed(4)}</strong>
      </div>
    `).join("");

    document.getElementById("ml-result-display").innerHTML = `
      <div style="background:rgba(255,255,255,0.02); padding:16px; border-radius:var(--radius-md); border:1px solid var(--border-color);">
        <h4 style="margin-bottom:10px; color:#fff;">${algorithm.toUpperCase()} Results</h4>
        ${metricsHtml}
      </div>
    `;

    await refreshModelsList();
  } catch (err) {
    showToast(`Training failed: ${err.message}`, "error");
  }
}

async function runModelComparison() {
  if (!state.activeDatasetId) return showToast("Select a dataset first", "info");
  const targetCol = document.getElementById("ml-target-col").value;
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
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    showToast(`Comparison complete! Best algorithm: ${data.best_algorithm}`, "success");

    // Render Leaderboard
    const tbody = data.results.map((r, idx) => `
      <tr>
        <td><strong>#${idx + 1}</strong></td>
        <td><span class="badge ${idx === 0 ? 'badge-valid' : 'badge-algo'}">${r.algorithm}</span></td>
        <td>${data.ranking_metric}</td>
        <td><strong style="color:var(--accent-cyan);">${r.metrics?.[data.ranking_metric]?.toFixed(4) ?? '-'}</strong></td>
        <td>${r.training_time_sec?.toFixed(3) ?? '-'}s</td>
        <td><span class="badge badge-valid">Completed</span></td>
      </tr>
    `).join("");
    document.querySelector("#ml-leaderboard-table tbody").innerHTML = tbody;
    await refreshModelsList();
  } catch (err) {
    showToast(`Comparison failed: ${err.message}`, "error");
  }
}

// ── Prediction ──────────────────────────────────────────────────
async function refreshModelsList() {
  try {
    const res = await fetch("/api/models");
    if (!res.ok) return;
    state.models = await res.json();
    document.getElementById("kpi-models-count").innerText = state.models.length;

    const sel = document.getElementById("predict-model-select");
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
  } catch (err) {
    console.error(err);
  }
}

async function renderPredictionForm(modelId) {
  if (!modelId) return;
  const container = document.getElementById("prediction-fields-container");
  container.innerHTML = "";

  const modelMeta = state.models.find(m => m.id === modelId);
  state.selectedModel = modelMeta;

  // Use features from active dataset or defaults
  const features = state.previewData?.columns?.filter(c => c !== "target" && c !== "Survived") || ["feature_1", "feature_2"];

  features.forEach(f => {
    const group = document.createElement("div");
    group.className = "form-group";
    group.innerHTML = `
      <label class="form-label">${f}</label>
      <input type="number" step="any" class="form-input pred-input" data-col="${f}" placeholder="Value..." value="1.0" required>
    `;
    container.appendChild(group);
  });
}

async function runPrediction() {
  const modelId = document.getElementById("predict-model-select").value;
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
    box.innerHTML = `
      <div style="font-size:36px; margin-bottom:8px;">🎯</div>
      <div class="kpi-label">Predicted Output</div>
      <div style="font-size:36px; font-weight:800; color:var(--accent-emerald); font-family:'Outfit'; margin:8px 0;">
        ${data.prediction}
      </div>
      ${data.confidence !== undefined && data.confidence !== null ? `
        <div style="margin-top:10px;">
          <span class="badge badge-info">Confidence: ${(data.confidence * 100).toFixed(1)}%</span>
        </div>
      ` : ''}
    `;
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
        <td><strong>${r.format.toUpperCase()}</strong></td>
        <td>${(r.file_size / 1024).toFixed(1)} KB</td>
        <td>${new Date(r.created_at).toLocaleString()}</td>
        <td>
          <a class="btn btn-secondary btn-sm" href="/api/reports/${r.id}/download" target="_blank">⬇ Download</a>
        </td>
      </tr>
    `).join("");
    document.querySelector("#reports-catalog-table tbody").innerHTML = tbody;
  } catch (err) {
    console.error(err);
  }
}
