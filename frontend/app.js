/* ── Charting helpers (inline SVG, no dependencies) ── */

// Feature importance horizontal bar chart
function buildFeatureImportanceChart(importances, maxBars = 15) {
    if (!importances || !Object.keys(importances).length) return '<p class="text-muted">No feature importances available for this model.</p>';
    const entries = Object.entries(importances).slice(0, maxBars);
    const maxVal = Math.max(...entries.map(e => e[1]));
    return entries.map(([feat, val]) => {
        const pct = maxVal > 0 ? (val / maxVal * 100).toFixed(1) : 0;
        const displayName = feat.length > 28 ? feat.slice(0, 28) + '…' : feat;
        return `<div style="display:flex;align-items:center;gap:10px;margin-bottom:7px;font-size:0.82rem">
            <span style="color:var(--text-secondary);min-width:160px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;font-family:monospace" title="${escHtml(feat)}">${escHtml(displayName)}</span>
            <div style="flex:1;height:10px;background:var(--border);border-radius:5px;overflow:hidden">
                <div style="width:${pct}%;height:100%;background:linear-gradient(90deg,var(--accent),var(--purple));border-radius:5px;transition:width 0.4s ease"></div>
            </div>
            <span style="color:var(--text-muted);min-width:54px;text-align:right;font-family:monospace">${(val * 100).toFixed(2)}%</span>
        </div>`;
    }).join('');
}

// Confusion matrix heatmap
function buildConfusionMatrix(cm, labels) {
    if (!cm || !cm.length) return '<p class="text-muted">No confusion matrix available.</p>';
    const n = cm.length;
    const maxVal = Math.max(...cm.flat());
    const cellLabels = labels && labels.length === n ? labels : cm.map((_, i) => `Class ${i}`);
    const cellSize = Math.max(48, Math.min(80, Math.floor(320 / n)));
    let html = `<div style="overflow-x:auto"><table style="border-collapse:collapse;font-size:0.8rem">`;
    html += `<tr><td style="width:${cellSize}px;height:${cellSize}px;color:var(--text-muted);font-size:0.7rem;padding:4px">Pred ↓<br>True →</td>`;
    cellLabels.forEach(l => { html += `<td style="text-align:center;padding:4px;color:var(--text-muted);font-size:0.72rem;font-weight:600">${escHtml(String(l))}</td>`; });
    html += `</tr>`;
    cm.forEach((row, ri) => {
        html += `<tr><td style="text-align:right;padding:4px;color:var(--text-muted);font-size:0.72rem;font-weight:600">${escHtml(String(cellLabels[ri]))}</td>`;
        row.forEach((val, ci) => {
            const intensity = maxVal > 0 ? val / maxVal : 0;
            const isDiag = ri === ci;
            const bg = isDiag
                ? `rgba(16,185,129,${0.15 + intensity * 0.75})`
                : `rgba(239,68,68,${intensity * 0.7})`;
            const textColor = intensity > 0.6 ? '#fff' : 'var(--text-primary)';
            html += `<td style="width:${cellSize}px;height:${cellSize}px;text-align:center;background:${bg};color:${textColor};font-weight:${isDiag?700:400};border:1px solid rgba(255,255,255,0.06);border-radius:4px;font-family:monospace;font-size:${n > 4 ? '0.7rem':'0.85rem'}">${val}</td>`;
        });
        html += `</tr>`;
    });
    html += `</table></div>`;
    return html;
}

// CV fold per-fold table
function buildCVFoldsTable(folds) {
    if (!folds || !folds.length) return '<p class="text-muted">No per-fold data available.</p>';
    const metrics = Object.keys(folds[0]).filter(k => k !== 'fold');
    let html = `<div class="table-wrap"><table><thead><tr><th>Fold</th>${metrics.map(m => `<th>${m.replace(/_/g,' ')}</th>`).join('')}</tr></thead><tbody>`;
    folds.forEach(f => {
        html += `<tr><td style="font-weight:700;color:var(--accent)">Fold ${f.fold}</td>`;
        metrics.forEach(m => { html += `<td class="font-mono">${(f[m]??'').toFixed ? f[m].toFixed(4) : f[m]}</td>`; });
        html += `</tr>`;
    });
    // Average row
    html += `<tr style="border-top:2px solid var(--border);background:rgba(99,102,241,0.06)"><td style="font-weight:700;color:var(--text-primary)">Mean</td>`;
    metrics.forEach(m => {
        const vals = folds.map(f => f[m]).filter(v => typeof v === 'number');
        const avg = vals.length ? vals.reduce((a,b) => a+b, 0) / vals.length : 0;
        html += `<td class="font-mono" style="font-weight:700;color:var(--accent)">${avg.toFixed(4)}</td>`;
    });
    html += `</tr></tbody></table></div>`;
    return html;
}

// Train vs Test metric comparison mini-bars
function buildTrainTestComparison(trainMetrics, testMetrics, baselineMetrics) {
    const keys = Object.keys(testMetrics).filter(k => typeof testMetrics[k] === 'number' && !['confusion_matrix'].includes(k));
    if (!keys.length) return '<p class="text-muted">No metrics available.</p>';
    return keys.map(k => {
        const train = trainMetrics[k];
        const test = testMetrics[k];
        const baseline = baselineMetrics && baselineMetrics[k];
        const trainPct = typeof train === 'number' ? (train * 100).toFixed(1) : null;
        const testPct = (test * 100).toFixed(1);
        const gap = typeof train === 'number' ? Math.abs(train - test) : 0;
        const gapColor = gap > 0.15 ? 'var(--red)' : gap > 0.05 ? 'var(--amber)' : 'var(--emerald)';
        return `<div style="margin-bottom:12px">
            <div style="display:flex;justify-content:space-between;font-size:0.8rem;margin-bottom:5px">
                <span style="color:var(--text-secondary);font-weight:600">${k.replace(/_/g,' ').replace(/\b\w/g,c=>c.toUpperCase())}</span>
                <span style="display:flex;gap:10px">
                    ${typeof train === 'number' ? `<span style="color:var(--text-muted)">Train: ${trainPct}%</span>` : ''}
                    <span style="color:var(--accent);font-weight:700">Test: ${testPct}%</span>
                    ${typeof baseline === 'number' ? `<span style="color:var(--text-muted)">Base: ${(baseline*100).toFixed(1)}%</span>` : ''}
                    ${gap > 0.05 ? `<span style="color:${gapColor};font-size:0.72rem">Δ${(gap*100).toFixed(1)}%</span>` : ''}
                </span>
            </div>
            <div style="display:flex;flex-direction:column;gap:3px">
                ${typeof train === 'number' ? `<div style="display:flex;align-items:center;gap:6px;font-size:0.72rem;color:var(--text-muted)">
                    <span style="min-width:40px">Train</span>
                    <div style="flex:1;height:6px;background:var(--border);border-radius:3px;overflow:hidden">
                        <div style="width:${trainPct}%;height:100%;background:rgba(99,102,241,0.5);border-radius:3px"></div>
                    </div></div>` : ''}
                <div style="display:flex;align-items:center;gap:6px;font-size:0.72rem;color:var(--text-muted)">
                    <span style="min-width:40px">Test</span>
                    <div style="flex:1;height:6px;background:var(--border);border-radius:3px;overflow:hidden">
                        <div style="width:${testPct}%;height:100%;background:linear-gradient(90deg,var(--accent),var(--purple));border-radius:3px"></div>
                    </div></div>
                ${typeof baseline === 'number' ? `<div style="display:flex;align-items:center;gap:6px;font-size:0.72rem;color:var(--text-muted)">
                    <span style="min-width:40px">Base</span>
                    <div style="flex:1;height:6px;background:var(--border);border-radius:3px;overflow:hidden">
                        <div style="width:${(baseline*100).toFixed(1)}%;height:100%;background:rgba(255,255,255,0.15);border-radius:3px"></div>
                    </div></div>` : ''}
            </div>
        </div>`;
    }).join('');
}

const API = '';  // Same origin

// ══════════════════════════════════════════════════════════════════════════════
// Global State
// ══════════════════════════════════════════════════════════════════════════════

const State = {
    datasets: [],
    currentDataset: null,
    columns: [],
    dtypes: {},
    previewData: [],
    profile: null,
    targetCandidates: [],
    experimentResult: null,
    auditResult: null,
    correctionResult: null,
    benchmarkResult: null,
};


// ══════════════════════════════════════════════════════════════════════════════
// Router
// ══════════════════════════════════════════════════════════════════════════════

function navigate(page) {
    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
    document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));

    const pageEl = document.getElementById(`page-${page}`);
    if (pageEl) pageEl.classList.add('active');

    const navEl = document.querySelector(`[data-page="${page}"]`);
    if (navEl) navEl.classList.add('active');

    const renderer = {
        dashboard:  renderDashboard,
        datasets:   renderDatasets,
        experiment: renderExperiment,
        findings:   renderFindings,
        comparison: renderComparison,
        benchmark:  renderBenchmark,
        reports:    renderReports,
    }[page];

    if (renderer) renderer();
}

// ══════════════════════════════════════════════════════════════════════════════
// API helpers
// ══════════════════════════════════════════════════════════════════════════════

async function apiFetch(url, options = {}) {
    const res = await fetch(API + url, options);
    if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail || res.statusText);
    }
    return res.json();
}

function showLoading(text = 'Processing...') {
    document.getElementById('loadingText').textContent = text;
    document.getElementById('loadingOverlay').classList.add('show');
}

function hideLoading() {
    document.getElementById('loadingOverlay').classList.remove('show');
}

function toast(message, type = 'info') {
    const container = document.getElementById('toastContainer');
    const icons = { success: '✅', error: '❌', info: 'ℹ️' };
    const el = document.createElement('div');
    el.className = `toast ${type}`;
    el.innerHTML = `<span>${icons[type]}</span><span>${message}</span>`;
    container.appendChild(el);
    setTimeout(() => el.remove(), 3200);
}

// ══════════════════════════════════════════════════════════════════════════════
// Dashboard Page
// ══════════════════════════════════════════════════════════════════════════════

async function renderDashboard() {
    const page = document.getElementById('page-dashboard');
    page.innerHTML = `
        <div class="page-header">
            <h1 class="page-title">🔬 Dashboard</h1>
            <p class="page-subtitle">Evidence-driven ML experiment auditing — all data is real, all training is real.</p>
        </div>

        <div class="stats-grid" id="statsGrid">
            ${[...Array(6)].map(() => `<div class="stat-card"><div class="stat-value">—</div><div class="stat-label">Loading...</div></div>`).join('')}
        </div>

        <div class="grid-2 mb-24">
            <div class="card">
                <div class="section-title">🚀 Quick Actions</div>
                <div style="display:flex;flex-direction:column;gap:10px;">
                    <button class="btn btn-primary btn-full" onclick="navigate('datasets')">📚 Browse & Download Datasets</button>
                    <button class="btn btn-secondary btn-full" onclick="navigate('experiment')">🧪 Run New Experiment</button>
                    <button class="btn btn-secondary btn-full" onclick="navigate('benchmark')">🎯 Run Benchmark Suite</button>
                </div>
            </div>
            <div class="card">
                <div class="section-title">🔍 What This Tool Audits</div>
                <div style="display:flex;flex-direction:column;gap:8px;font-size:0.85rem;color:var(--text-secondary);">
                    <div>✓ <strong style="color:var(--text-primary)">Target Leakage</strong> — Detects features that encode the target</div>
                    <div>✓ <strong style="color:var(--text-primary)">Preprocessing Leakage</strong> — AST code analysis + statistical checks</div>
                    <div>✓ <strong style="color:var(--text-primary)">Split Contamination</strong> — Exact row overlap + near-duplicate detection</div>
                    <div>✓ <strong style="color:var(--text-primary)">Performance Anomalies</strong> — Suspicious accuracy, large train-test gaps</div>
                    <div>✓ <strong style="color:var(--text-primary)">Evaluation Integrity</strong> — Metric selection, class imbalance awareness</div>
                    <div>✓ <strong style="color:var(--text-primary)">Data Quality</strong> — Missing values, infinities, constants, duplicates</div>
                    <div>✓ <strong style="color:var(--text-primary)">Auto Correction</strong> — Proposes and applies verified fixes</div>
                </div>
            </div>
        </div>

        <div class="card">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px">
                <div class="section-title" style="margin-bottom:0">📋 Experiment History</div>
                <button class="btn btn-secondary btn-sm" onclick="renderDashboard()">↻ Refresh</button>
            </div>
            <div id="experimentHistory">
                <div class="empty-state">
                    <div class="empty-icon">🧪</div>
                    <div class="empty-title">No experiments yet</div>
                    <div class="empty-desc">Start by selecting a dataset and running an experiment.</div>
                </div>
            </div>
        </div>`;

    try {
        const stats = await apiFetch('/api/dashboard/stats');

        const statDefs = [
            { icon: '📚', key: 'datasets',           label: 'Datasets',         gradient: 'linear-gradient(90deg,#6366f1,#a855f7)' },
            { icon: '🧪', key: 'experiments',        label: 'Experiments',      gradient: 'linear-gradient(90deg,#06b6d4,#6366f1)' },
            { icon: '✅', key: 'completed',          label: 'Completed',        gradient: 'linear-gradient(90deg,#10b981,#06b6d4)' },
            { icon: '🔍', key: 'audits',             label: 'Audit Runs',       gradient: 'linear-gradient(90deg,#f59e0b,#f97316)' },
            { icon: '🔴', key: 'confirmed_findings', label: 'Confirmed Issues', gradient: 'linear-gradient(90deg,#ef4444,#f97316)' },
            { icon: '🟡', key: 'suspected_findings', label: 'Suspected Issues', gradient: 'linear-gradient(90deg,#f59e0b,#fbbf24)' },
        ];

        const grid = document.getElementById('statsGrid');
        grid.innerHTML = statDefs.map(s => `
            <div class="stat-card" style="--gradient:${s.gradient}">
                <span class="stat-icon">${s.icon}</span>
                <div class="stat-value">${stats[s.key] ?? 0}</div>
                <div class="stat-label">${s.label}</div>
            </div>`).join('');

    } catch (e) {
        toast('Failed to load dashboard stats: ' + e.message, 'error');
    }

    // Load experiment history from DB
    try {
        const history = await apiFetch('/api/experiments/history?limit=20');
        const histEl = document.getElementById('experimentHistory');
        if (!histEl) return;

        if (!history.length) {
            histEl.innerHTML = `<div class="empty-state"><div class="empty-icon">🧪</div>
                <div class="empty-title">No experiments yet</div>
                <div class="empty-desc">Start by selecting a dataset and running an experiment.</div></div>`;
            return;
        }

        const statusIcon = { completed: '✅', failed: '❌', running: '⏳', pending: '⏸️' };
        const primaryMetric = (m) => {
            if (m.accuracy !== undefined) return `Acc ${(m.accuracy*100).toFixed(1)}%`;
            if (m.r2 !== undefined) return `R² ${m.r2.toFixed(3)}`;
            if (m.f1_score !== undefined) return `F1 ${(m.f1_score*100).toFixed(1)}%`;
            const keys = Object.keys(m);
            return keys.length ? `${keys[0]}: ${m[keys[0]]}` : '—';
        };

        histEl.innerHTML = `<div class="table-wrap"><table>
            <thead><tr>
                <th>Status</th><th>Name</th><th>Task</th><th>Target</th>
                <th>Models</th><th>Test Metric</th><th>Duration</th><th>Date</th>
            </tr></thead>
            <tbody>${history.map(e => `
                <tr>
                    <td>${statusIcon[e.status] || '❓'}</td>
                    <td style="font-weight:600;color:var(--text-primary)">${escHtml(e.name)}</td>
                    <td><span class="badge badge-purple">${e.task_type}</span></td>
                    <td><code class="font-mono" style="font-size:0.78rem">${escHtml(e.target_column)}</code></td>
                    <td style="font-size:0.78rem;color:var(--text-muted)">${(e.model_config?.models || []).join(', ') || '—'}</td>
                    <td style="font-family:monospace;color:var(--accent);font-weight:700">${primaryMetric(e.test_metrics)}</td>
                    <td>${e.duration ? e.duration.toFixed(1)+'s' : '—'}</td>
                    <td style="color:var(--text-muted);font-size:0.78rem">${e.created_at?.slice(0,16).replace('T',' ') || ''}</td>
                </tr>`).join('')}
            </tbody></table></div>`;
    } catch (e) {
        const histEl = document.getElementById('experimentHistory');
        if (histEl) histEl.innerHTML = `<p style="color:var(--text-muted)">Could not load history: ${escHtml(e.message)}</p>`;
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Dataset Library Page
// ══════════════════════════════════════════════════════════════════════════════

async function renderDatasets() {
    const page = document.getElementById('page-datasets');
    page.innerHTML = `
        <div class="page-header">
            <h1 class="page-title">📚 Dataset Library</h1>
            <p class="page-subtitle">Browse, download and preview real public datasets from sklearn, OpenML, and UCI.</p>
        </div>

        <div class="filter-bar">
            <input id="dsSearch" class="form-control filter-input" placeholder="🔍  Search datasets..." oninput="filterDatasets()">
            <select id="dsSource" class="form-control" style="width:140px" onchange="filterDatasets()">
                <option value="">All Sources</option>
                <option value="sklearn">sklearn</option>
                <option value="openml">OpenML</option>
                <option value="uci">UCI</option>
            </select>
            <select id="dsTask" class="form-control" style="width:160px" onchange="filterDatasets()">
                <option value="">All Tasks</option>
                <option value="classification">Classification</option>
                <option value="regression">Regression</option>
            </select>
        </div>

        <div class="tab-bar">
            <div class="tab active" onclick="switchDsTab('catalog',this)">Catalog</div>
            <div class="tab" onclick="switchDsTab('upload',this)">Upload CSV</div>
        </div>

        <div id="ds-catalog">
            <div id="datasetGrid" style="display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:16px;">
                <div class="card" style="text-align:center;padding:40px;color:var(--text-muted)">Loading datasets…</div>
            </div>
        </div>

        <div id="ds-upload" class="hidden">
            <div class="card">
                <div class="section-title">📁 Upload Your Dataset</div>
                <div class="upload-zone" id="uploadZone" onclick="document.getElementById('fileInput').click()"
                     ondragover="event.preventDefault();this.classList.add('dragover')"
                     ondragleave="this.classList.remove('dragover')"
                     ondrop="handleFileDrop(event)">
                    <div class="upload-icon">📂</div>
                    <div class="upload-text">Drop a CSV file here or <strong style="color:var(--accent-bright)">click to browse</strong></div>
                    <div class="upload-hint">Supports: .csv, .parquet — max 200 MB</div>
                </div>
                <input type="file" id="fileInput" accept=".csv,.parquet" class="hidden" onchange="handleFileUpload(event)">
                <div id="uploadPreview" class="mt-16"></div>
            </div>
        </div>`;

    await loadCatalog();
}

async function loadCatalog() {
    try {
        const datasets = await apiFetch('/api/datasets/catalog');
        State.datasets = datasets;
        renderDatasetGrid(datasets);
    } catch (e) {
        document.getElementById('datasetGrid').innerHTML = `<div class="card"><p style="color:var(--red)">Failed to load catalog: ${e.message}</p></div>`;
    }
}

function filterDatasets() {
    const q = document.getElementById('dsSearch')?.value.toLowerCase() || '';
    const src = document.getElementById('dsSource')?.value || '';
    const task = document.getElementById('dsTask')?.value || '';
    const filtered = State.datasets.filter(d =>
        (!q || d.name.toLowerCase().includes(q) || d.description?.toLowerCase().includes(q)) &&
        (!src || d.source === src) &&
        (!task || d.task_type === task || d.task_type === 'both')
    );
    renderDatasetGrid(filtered);
}

function renderDatasetGrid(datasets) {
    const grid = document.getElementById('datasetGrid');
    if (!grid) return;

    if (!datasets.length) {
        grid.innerHTML = `<div class="empty-state"><div class="empty-icon">🔍</div><div class="empty-title">No datasets found</div></div>`;
        return;
    }

    const srcClass = { sklearn: 'sklearn', openml: 'openml', uci: 'uci', upload: 'upload' };

    grid.innerHTML = datasets.map(d => `
        <div class="dataset-card">
            <div class="dataset-card-header">
                <div>
                    <div class="dataset-name">${escHtml(d.name)}</div>
                    <div class="dataset-meta mt-4">
                        <span class="dataset-tag ${srcClass[d.source] || ''}">${d.source}</span>
                        <span class="dataset-tag ${d.task_type === 'classification' ? 'sklearn' : 'uci'}">${d.task_type || '?'}</span>
                        ${d.is_cached ? '<span class="dataset-tag cached">✓ Cached</span>' : ''}
                    </div>
                </div>
            </div>
            <div class="dataset-desc">${escHtml((d.description || '').slice(0,140))}${(d.description||'').length > 140 ? '…' : ''}</div>
            <div class="dataset-stats">
                <span class="dataset-stat">📊 ${d.row_count?.toLocaleString() || '?'} rows</span>
                <span class="dataset-stat">📋 ${d.col_count || '?'} cols</span>
                ${d.missing_pct > 0 ? `<span class="dataset-stat">⚠️ ${d.missing_pct?.toFixed(1)}% missing</span>` : ''}
            </div>
            <div class="dataset-actions">
                <button class="btn btn-primary btn-sm" onclick="loadDatasetCard('${d.source}','${d.source_id}','${escHtml(d.name)}')"
                    ${d.is_cached ? 'style="background:linear-gradient(135deg,#059669,#10b981)"' : ''}>
                    ${d.is_cached ? '✅ Load' : '⬇️ Download & Load'}
                </button>
                ${d.source_url ? `<a href="${d.source_url}" target="_blank" class="btn btn-secondary btn-sm">🔗 Source</a>` : ''}
            </div>
        </div>`).join('');
}

function switchDsTab(tab, el) {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    el.classList.add('active');
    document.getElementById('ds-catalog').classList.toggle('hidden', tab !== 'catalog');
    document.getElementById('ds-upload').classList.toggle('hidden', tab !== 'upload');
}

async function loadDatasetCard(source, sourceId, name) {
    showLoading(`Downloading ${name}…`);
    try {
        const data = await apiFetch(`/api/datasets/load?source=${source}&source_id=${sourceId}&name=${encodeURIComponent(name)}`, { method: 'POST' });
        State.currentDataset = { source, source_id: sourceId, name };
        State.columns = data.columns;
        State.dtypes = data.dtypes;
        State.previewData = data.preview;
        State.profile = data.profile;
        State.targetCandidates = data.target_candidates;

        toast(`✅ Loaded ${name} — ${data.rows.toLocaleString()} rows × ${data.cols} cols`, 'success');
        await loadCatalog(); // Refresh cached status
    } catch (e) {
        toast('Failed to load dataset: ' + e.message, 'error');
    } finally {
        hideLoading();
    }
}

async function handleFileUpload(event) {
    const file = event.target.files[0];
    if (!file) return;
    await uploadFile(file);
}

async function handleFileDrop(event) {
    event.preventDefault();
    document.getElementById('uploadZone').classList.remove('dragover');
    const file = event.dataTransfer.files[0];
    if (file) await uploadFile(file);
}

async function uploadFile(file) {
    showLoading(`Uploading ${file.name}…`);
    try {
        const formData = new FormData();
        formData.append('file', file);
        const res = await fetch('/api/datasets/upload', { method: 'POST', body: formData });
        if (!res.ok) throw new Error((await res.json()).detail);
        const data = await res.json();

        State.currentDataset = { source: 'upload', source_id: 'uploaded', name: file.name };
        State.columns = data.columns;
        State.dtypes = data.dtypes;
        State.previewData = data.preview;
        State.profile = data.profile;
        State.targetCandidates = data.target_candidates;

        document.getElementById('uploadPreview').innerHTML = `
            <div class="alert alert-success"><span class="alert-icon">✅</span>
            Loaded <strong>${file.name}</strong> — ${data.rows.toLocaleString()} rows × ${data.cols} columns</div>
            <div class="section-title mt-16">Preview</div>
            ${buildPreviewTable(data.preview, data.columns.slice(0,10))}`;

        toast('File uploaded successfully!', 'success');
    } catch (e) {
        toast('Upload failed: ' + e.message, 'error');
    } finally {
        hideLoading();
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// New Experiment Page
// ══════════════════════════════════════════════════════════════════════════════

async function renderExperiment() {
    const page = document.getElementById('page-experiment');

    // Load dataset preview stats if we have a dataset
    let datasetInfoHtml = '';
    if (State.currentDataset) {
        try {
            const cur = await apiFetch('/api/datasets/current');
            if (cur.loaded) {
                datasetInfoHtml = `<div class="alert alert-success"><span class="alert-icon">✅</span>
                    <div><strong>${escHtml(cur.name)}</strong>
                    <span style="color:var(--text-muted);font-size:0.82rem;margin-left:10px">
                        ${cur.rows?.toLocaleString()} rows &times; ${cur.cols} cols &nbsp;•&nbsp;
                        ${cur.numeric_cols} numeric, ${cur.categorical_cols} categorical &nbsp;•&nbsp;
                        ${cur.missing_pct}% missing
                    </span></div></div>
                    <button class="btn btn-secondary btn-sm mt-8" onclick="navigate('datasets')">Change Dataset</button>`;
            }
        } catch (e) {}
    }

    page.innerHTML = `
        <div class="page-header">
            <h1 class="page-title">🧪 New Experiment</h1>
            <p class="page-subtitle">Train real ML models with automatic audit for data leakage and evaluation errors.</p>
        </div>

        <div class="steps">
            <div class="step active" id="step1-ind"><div class="step-num">1</div><span>Dataset</span></div>
            <div class="step-connector"></div>
            <div class="step" id="step2-ind"><div class="step-num">2</div><span>Configure</span></div>
            <div class="step-connector"></div>
            <div class="step" id="step3-ind"><div class="step-num">3</div><span>Run &amp; Audit</span></div>
            <div class="step-connector"></div>
            <div class="step" id="step4-ind"><div class="step-num">4</div><span>Results</span></div>
        </div>

        <div class="grid-2">
            <!-- Left: configuration -->
            <div>
                <!-- Dataset selection -->
                <div class="card mb-16">
                    <div class="section-title">1. Dataset</div>
                    ${State.currentDataset ? datasetInfoHtml :
                        `<div class="alert alert-warning"><span class="alert-icon">⚠️</span>No dataset loaded. Go to Dataset Library first.</div>
                         <button class="btn btn-primary btn-sm mt-8" onclick="navigate('datasets')">📚 Load Dataset</button>`
                    }
                </div>

                <!-- Configuration -->
                <div class="card mb-16" ${!State.currentDataset ? 'style="opacity:0.5;pointer-events:none"' : ''}>
                    <div class="section-title">2. Configuration</div>

                    <div class="form-group">
                        <label class="form-label">Experiment Name</label>
                        <input id="expName" class="form-control" type="text" placeholder="my_experiment"
                               value="${State.currentDataset ? State.currentDataset.name + '_exp' : ''}">
                    </div>

                    <div class="grid-2">
                        <div class="form-group">
                            <label class="form-label">Target Column</label>
                            <select id="expTarget" class="form-control" onchange="updateTargetAnalysis()">
                                ${(State.targetCandidates.length ? State.targetCandidates : State.columns).map(c =>
                                    `<option>${escHtml(c)}</option>`).join('')}
                            </select>
                        </div>
                        <div class="form-group">
                            <label class="form-label">Task Type</label>
                            <select id="expTask" class="form-control" onchange="refreshModels()">
                                <option value="classification">Classification</option>
                                <option value="regression">Regression</option>
                            </select>
                        </div>
                    </div>

                    <div class="form-group">
                        <label class="form-label">Models to Train</label>
                        <div class="checkbox-grid" id="modelCheckboxes">Loading...</div>
                    </div>

                    <div class="grid-2">
                        <div class="form-group">
                            <label class="form-label">Test Size <span class="range-value" id="testSizeVal">20%</span></label>
                            <input id="expTestSize" class="form-control" type="range" min="10" max="40" step="5" value="20"
                                   oninput="document.getElementById('testSizeVal').textContent=this.value+'%'">
                        </div>
                        <div class="form-group">
                            <label class="form-label">CV Folds <span class="range-value" id="cvFoldsVal">5</span></label>
                            <input id="expCvFolds" class="form-control" type="range" min="2" max="10" step="1" value="5"
                                   oninput="document.getElementById('cvFoldsVal').textContent=this.value">
                        </div>
                    </div>

                    <div class="form-group">
                        <label class="form-label">Random Seed</label>
                        <input id="expSeed" class="form-control" type="number" value="42" min="0" max="99999">
                    </div>
                </div>

                <button class="btn btn-primary btn-full btn-lg" id="runBtn" onclick="runExperiment()"
                        ${!State.currentDataset ? 'disabled' : ''}>
                    🚀 Run Experiment &amp; Audit
                </button>
            </div>

            <!-- Right: target analysis + progress + results -->
            <div>
                <div id="targetAnalysisPanel" class="card mb-16 hidden">
                    <div class="section-title">Target Analysis</div>
                    <div id="targetAnalysisContent"></div>
                </div>

                <div id="progressPanel" class="hidden">
                    <div class="progress-wrap">
                        <div class="progress-label">
                            <span id="progressLabel">Running experiment...</span>
                            <span id="progressPct">0%</span>
                        </div>
                        <div class="progress-bar"><div class="progress-fill" id="progressFill" style="width:0%"></div></div>
                        <div class="progress-step" id="progressStep">Initializing...</div>
                    </div>
                </div>

                <div id="resultsPanel" class="hidden">
                    <div class="card">
                        <div class="section-title">📊 Results</div>
                        <div id="resultsContent"></div>
                    </div>
                </div>
            </div>
        </div>`;

    await refreshModels();
    if (State.currentDataset && State.targetCandidates.length) {
        updateTargetAnalysis();
    }
}

async function refreshModels() {
    const task = document.getElementById('expTask')?.value || 'classification';
    const container = document.getElementById('modelCheckboxes');
    if (!container) return;
    try {
        container.innerHTML = '<span style="color:var(--text-muted);font-size:0.85rem">Loading models…</span>';
        const data = await apiFetch(`/api/experiments/models?task_type=${task}`);
        const defaultSelected = ['random_forest', 'logistic_regression'];
        container.innerHTML = Object.entries(data.models).map(([key, label]) => {
            const checked = defaultSelected.includes(key);
            return `<div class="checkbox-item ${checked ? 'checked' : ''}" data-value="${key}" onclick="toggleCheckbox(this)">
                <div class="check-box"></div>
                <span class="checkbox-label">${label}</span>
            </div>`;
        }).join('');
    } catch (e) {
        if (container) container.innerHTML = `<span style="color:var(--red);font-size:0.84rem">Failed to load models: ${escHtml(e.message)}</span>`;
    }
}

function toggleCheckbox(el) {
    el.classList.toggle('checked');
}

function getSelectedModels() {
    return Array.from(document.querySelectorAll('#modelCheckboxes .checkbox-item.checked'))
        .map(el => el.dataset.value)
        .filter(Boolean);
}

async function updateTargetAnalysis() {
    const target = document.getElementById('expTarget')?.value;
    if (!target || !State.currentDataset) return;

    try {
        const data = await apiFetch(`/api/datasets/current/profile?target_column=${encodeURIComponent(target)}`);
        const ta = data.target_analysis;
        if (!ta) return;

        const panel = document.getElementById('targetAnalysisPanel');
        const content = document.getElementById('targetAnalysisContent');
        if (!panel || !content) return;

        panel.classList.remove('hidden');

        // Auto-set task type
        if (ta.task_type) {
            const taskSelect = document.getElementById('expTask');
            if (taskSelect) taskSelect.value = ta.task_type;
        }

        let html = `
            <div class="metrics-grid" style="grid-template-columns:repeat(3,1fr)">
                <div class="metric-item">
                    <div class="metric-name">Unique Values</div>
                    <div class="metric-val">${ta.n_unique || '?'}</div>
                </div>
                <div class="metric-item">
                    <div class="metric-name">Task Type</div>
                    <div class="metric-val" style="font-size:1rem">${ta.task_type || '?'}</div>
                </div>
                <div class="metric-item">
                    <div class="metric-name">Missing</div>
                    <div class="metric-val">${ta.missing_pct?.toFixed(1) || 0}%</div>
                </div>
            </div>`;

        if (ta.is_imbalanced) {
            html += `<div class="alert alert-warning mt-8"><span class="alert-icon">⚠️</span>
                Imbalanced classes detected. Stratified splitting and F1/PR-AUC metrics recommended.</div>`;
        }

        if (ta.class_distribution) {
            const dist = ta.class_distribution;
            const maxVal = Math.max(...Object.values(dist));
            html += `<div class="mt-12" style="font-size:0.82rem;color:var(--text-muted);margin-bottom:8px;font-weight:600;text-transform:uppercase;letter-spacing:.04em">Class Distribution</div>`;
            html += Object.entries(dist).slice(0,12).map(([cls, cnt]) => {
                const pct = (cnt / Object.values(dist).reduce((a,b) => a+b, 0) * 100);
                const w = (cnt / maxVal * 100);
                return `<div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;font-size:0.83rem">
                    <span style="color:var(--text-secondary);min-width:60px;font-family:monospace">${escHtml(String(cls))}</span>
                    <div style="flex:1;height:6px;background:var(--border);border-radius:3px;overflow:hidden">
                        <div style="width:${w.toFixed(1)}%;height:100%;background:linear-gradient(90deg,var(--accent),var(--purple));border-radius:3px"></div>
                    </div>
                    <span style="color:var(--text-muted);min-width:50px;text-align:right">${cnt} (${pct.toFixed(1)}%)</span>
                </div>`;
            }).join('');
        }

        content.innerHTML = html;
    } catch (e) { /* silent */ }
}

async function runExperiment() {
    if (!State.currentDataset) { toast('No dataset loaded', 'error'); return; }

    const models = getSelectedModels();
    if (!models.length) { toast('Select at least one model — click a model card to select it', 'error'); return; }

    const req = {
        dataset_source: State.currentDataset.source,
        dataset_source_id: State.currentDataset.source_id,
        dataset_name: State.currentDataset.name,
        target_column: document.getElementById('expTarget').value,
        task_type: document.getElementById('expTask').value,
        model_names: models,
        test_size: parseInt(document.getElementById('expTestSize').value) / 100,
        cv_folds: parseInt(document.getElementById('expCvFolds').value),
        random_seed: parseInt(document.getElementById('expSeed').value),
        experiment_name: document.getElementById('expName').value || State.currentDataset.name + '_exp',
    };

    // Show progress
    document.getElementById('progressPanel').classList.remove('hidden');
    document.getElementById('resultsPanel').classList.add('hidden');
    document.getElementById('runBtn').disabled = true;

    // Simulate progress while waiting
    let prog = 0;
    const steps = ['Profiling dataset…', 'Preparing features…', 'Splitting data…', 'Training models…', 'Running audit…', 'Finalizing…'];
    const progInterval = setInterval(() => {
        if (prog < 90) {
            prog += Math.random() * 8;
            prog = Math.min(prog, 90);
            const pct = prog.toFixed(0);
            document.getElementById('progressFill').style.width = pct + '%';
            document.getElementById('progressPct').textContent = pct + '%';
            document.getElementById('progressStep').textContent = steps[Math.floor(prog / 16)] || steps[4];
        }
    }, 600);

    try {
        const result = await apiFetch('/api/experiments/run', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(req),
        });

        clearInterval(progInterval);
        document.getElementById('progressFill').style.width = '100%';
        document.getElementById('progressPct').textContent = '100%';
        document.getElementById('progressStep').textContent = '✅ Complete!';

        State.experimentResult = result;
        State.auditResult = result.audit;

        if (result.status === 'failed') {
            toast('Experiment failed: ' + result.error, 'error');
            return;
        }

        renderExperimentResults(result);
        document.getElementById('resultsPanel').classList.remove('hidden');
        toast('Experiment and audit complete!', 'success');

        // Mark steps done
        ['step1-ind','step2-ind','step3-ind','step4-ind'].forEach((id,i) => {
            const el = document.getElementById(id);
            if (el) { el.className = 'step done'; }
        });

    } catch (e) {
        clearInterval(progInterval);
        // Reset progress bar on failure
        document.getElementById('progressFill').style.width = '0%';
        document.getElementById('progressPct').textContent = '0%';
        document.getElementById('progressStep').textContent = '❌ Failed';
        setTimeout(() => document.getElementById('progressPanel')?.classList.add('hidden'), 2000);
        toast('Experiment failed: ' + e.message, 'error');
    } finally {
        document.getElementById('runBtn').disabled = false;
    }
}

function renderExperimentResults(result) {
    const container = document.getElementById('resultsContent');

    // Audit summary banner
    const audit = result.audit;
    const summary = audit.summary;

    let html = `
        <div style="background:rgba(99,102,241,0.08);border:1px solid rgba(99,102,241,0.2);border-radius:10px;padding:16px;margin-bottom:20px">
            <div style="font-weight:700;color:var(--text-primary);margin-bottom:10px">🔍 Audit Summary</div>
            <div style="display:flex;gap:16px;flex-wrap:wrap">
                <span>Total: <strong>${summary.total_findings}</strong></span>
                <span style="color:var(--red)">Confirmed: <strong>${summary.confirmed}</strong></span>
                <span style="color:var(--amber)">Suspected: <strong>${summary.suspected}</strong></span>
                <span style="color:var(--emerald)">Auto-correctable: <strong>${summary.auto_correctable}</strong></span>
                <span style="color:var(--text-muted)">Audit time: ${summary.duration_seconds?.toFixed(2)}s</span>
            </div>
        </div>`;

    // Split info
    const si = result.split_info;
    html += `<div style="font-size:0.8rem;color:var(--text-muted);margin-bottom:16px;padding:8px 12px;background:rgba(255,255,255,0.03);border-radius:6px;border:1px solid var(--border)">
        Split: <strong style="color:var(--text-secondary)">${si?.train_size?.toLocaleString()} train</strong> /
        <strong style="color:var(--text-secondary)">${si?.test_size?.toLocaleString()} test</strong> &nbsp;•&nbsp;
        Seed: ${si?.random_seed} &nbsp;•&nbsp;
        Overlap: <span style="color:${si?.overlap?.overlap_count>0?'var(--red)':'var(--emerald)'}">${si?.overlap?.overlap_count || 0} rows</span>
    </div>`;

    // Tab bar for multi-model
    const modelNames = Object.keys(result.training_results);
    if (modelNames.length > 1) {
        html += `<div class="tab-bar" id="modelTabBar">
            ${modelNames.map((m, i) => `<div class="tab ${i===0?'active':''}" onclick="switchModelTab('${m}',this)">
                ${m.replace(/_/g,' ').replace(/\b\w/g,c=>c.toUpperCase())}
            </div>`).join('')}
        </div>`;
    }

    // Model detail panels
    modelNames.forEach((modelName, idx) => {
        const tr = result.training_results[modelName];
        const isActive = idx === 0;
        const displayName = modelName.replace(/_/g,' ').replace(/\b\w/g,c=>c.toUpperCase());

        html += `<div class="model-panel ${isActive?'active':'hidden'}" id="panel-${modelName}">

            <!-- Metrics: Train vs Test vs Baseline -->
            <div class="section-title" style="margin-top:16px">🤖 ${displayName}</div>
            <div class="card" style="margin-bottom:16px">
                <div style="font-size:0.78rem;color:var(--text-muted);margin-bottom:10px;font-weight:600;text-transform:uppercase;letter-spacing:.04em">Train / Test / Baseline Comparison</div>
                ${buildTrainTestComparison(tr.train_metrics, tr.test_metrics, tr.baseline_metrics)}
                <div style="font-size:0.75rem;color:var(--text-muted);margin-top:8px">Training time: ${tr.training_time}s</div>
            </div>

            <!-- Feature Importance -->
            ${tr.feature_importances && Object.keys(tr.feature_importances).length ? `
            <div class="card" style="margin-bottom:16px">
                <div style="font-size:0.78rem;color:var(--text-muted);margin-bottom:10px;font-weight:600;text-transform:uppercase;letter-spacing:.04em">Feature Importance (Top 15)</div>
                ${buildFeatureImportanceChart(tr.feature_importances, 15)}
            </div>` : ''}

            <!-- Confusion Matrix (classification only) -->
            ${tr.confusion_matrix ? `
            <div class="card" style="margin-bottom:16px">
                <div style="font-size:0.78rem;color:var(--text-muted);margin-bottom:10px;font-weight:600;text-transform:uppercase;letter-spacing:.04em">Confusion Matrix (Test Set)</div>
                ${buildConfusionMatrix(tr.confusion_matrix)}
                <div style="font-size:0.72rem;color:var(--text-muted);margin-top:8px">Green diagonal = correct predictions. Red off-diagonal = errors.</div>
            </div>` : ''}

            <!-- CV Fold Breakdown -->
            ${tr.cv_folds && tr.cv_folds.length ? `
            <div class="card" style="margin-bottom:16px">
                <div style="font-size:0.78rem;color:var(--text-muted);margin-bottom:10px;font-weight:600;text-transform:uppercase;letter-spacing:.04em">Cross-Validation Per-Fold Results</div>
                ${buildCVFoldsTable(tr.cv_folds)}
                <div style="font-size:0.72rem;color:var(--text-muted);margin-top:6px">
                    CV Accuracy: ${tr.cv_metrics.cv_accuracy_mean ? (tr.cv_metrics.cv_accuracy_mean*100).toFixed(2)+'% ± '+(tr.cv_metrics.cv_accuracy_std*100).toFixed(2)+'%' : '—'}
                </div>
            </div>` : ''}
        </div>`;
    });

    // Top findings
    if (audit.findings?.length) {
        html += `<div style="margin-top:16px"><div class="section-title">Top Findings</div>`;
        audit.findings.slice(0,5).forEach(f => {
            html += buildFindingMini(f);
        });
        if (audit.findings.length > 5) {
            html += `<button class="btn btn-secondary btn-sm mt-8" onclick="navigate('findings')">View all ${audit.findings.length} findings →</button>`;
        }
        html += `</div>`;
    }

    container.innerHTML = html;
}

function switchModelTab(modelName, tabEl) {
    document.querySelectorAll('#modelTabBar .tab').forEach(t => t.classList.remove('active'));
    tabEl.classList.add('active');
    document.querySelectorAll('.model-panel').forEach(p => p.classList.add('hidden'));
    const panel = document.getElementById(`panel-${modelName}`);
    if (panel) { panel.classList.remove('hidden'); panel.classList.add('active'); }
}

async function exportFindingsCSV() {
    try {
        const res = await fetch('/api/findings/export');
        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: res.statusText }));
            throw new Error(err.detail || res.statusText);
        }
        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        // Use content-disposition filename from server if available
        const cd = res.headers.get('content-disposition') || '';
        const match = cd.match(/filename=([^;]+)/);
        a.download = match ? match[1].trim() : 'audit_findings.csv';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        toast('Findings exported as CSV!', 'success');
    } catch (e) {
        toast('Export failed: ' + e.message, 'error');
    }
}

function buildFindingMini(f) {
    return `<div class="finding-card sev-${f.severity}" style="margin-bottom:8px;padding:12px 16px">
        <div class="finding-header">
            <span class="badge badge-${f.severity}">${f.severity}</span>
            <span class="finding-id">${f.finding_id}</span>
            <span class="badge badge-${f.status}" style="font-size:0.68rem">${f.status}</span>
            ${f.auto_correctable ? '<span class="badge badge-green" style="font-size:0.68rem">🔧 correctable</span>' : ''}
        </div>
        <div class="finding-desc">${escHtml(f.description)}</div>
    </div>`;
}

// ══════════════════════════════════════════════════════════════════════════════
// Audit Findings Page
// ══════════════════════════════════════════════════════════════════════════════

function renderFindings() {
    const page = document.getElementById('page-findings');
    const audit = State.auditResult;

    if (!audit) {
        page.innerHTML = `<div class="page-header"><h1 class="page-title">🔍 Audit Findings</h1></div>
            <div class="empty-state"><div class="empty-icon">🔍</div>
            <div class="empty-title">No audit results yet</div>
            <div class="empty-desc">Run an experiment from the New Experiment page first.</div>
            <button class="btn btn-primary mt-24" onclick="navigate('experiment')">🧪 Run Experiment</button></div>`;
        return;
    }

    const summary = audit.summary;
    const findings = audit.findings || [];

    page.innerHTML = `
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px">
            <div>
                <h1 class="page-title" style="margin-bottom:4px">🔍 Audit Findings</h1>
                <p class="page-subtitle" style="margin:0">All detected issues with full evidence and recommendations.</p>
            </div>
            <div style="display:flex;gap:8px">
                <button class="btn btn-secondary btn-sm" onclick="exportFindingsCSV()">⬇️ Export CSV</button>
                <button class="btn btn-secondary btn-sm" onclick="navigate('comparison')">🔧 Apply Fixes</button>
            </div>
        </div>

        <div class="stats-grid" style="grid-template-columns:repeat(5,1fr);margin-bottom:24px">
            <div class="stat-card" style="--gradient:linear-gradient(90deg,#6366f1,#a855f7)">
                <span class="stat-icon">📋</span><div class="stat-value">${summary.total_findings}</div><div class="stat-label">Total</div></div>
            <div class="stat-card" style="--gradient:linear-gradient(90deg,#ef4444,#f97316)">
                <span class="stat-icon">🔴</span><div class="stat-value">${summary.critical + summary.high}</div><div class="stat-label">Critical/High</div></div>
            <div class="stat-card" style="--gradient:linear-gradient(90deg,#ef4444,#dc2626)">
                <span class="stat-icon">⚠️</span><div class="stat-value">${summary.confirmed}</div><div class="stat-label">Confirmed</div></div>
            <div class="stat-card" style="--gradient:linear-gradient(90deg,#f59e0b,#f97316)">
                <span class="stat-icon">🟡</span><div class="stat-value">${summary.suspected}</div><div class="stat-label">Suspected</div></div>
            <div class="stat-card" style="--gradient:linear-gradient(90deg,#10b981,#06b6d4)">
                <span class="stat-icon">🔧</span><div class="stat-value">${summary.auto_correctable}</div><div class="stat-label">Auto-fixable</div></div>
        </div>

        <div class="grid-2 mb-24">
            <div class="card">
                <div class="section-title">By Category</div>
                ${buildCategoryBars(audit.by_category)}
            </div>
            <div class="card">
                <div class="section-title">By Severity</div>
                ${buildSeverityBars(audit.by_severity)}
            </div>
        </div>

        <div class="filter-bar">
            <input id="findSearch" class="form-control filter-input" placeholder="🔍 Search findings..." oninput="filterFindings()">
            <select id="findSev" class="form-control" style="width:130px" onchange="filterFindings()">
                <option value="">All Severity</option>
                <option value="critical">Critical</option>
                <option value="high">High</option>
                <option value="medium">Medium</option>
                <option value="low">Low</option>
                <option value="info">Info</option>
            </select>
            <select id="findStatus" class="form-control" style="width:160px" onchange="filterFindings()">
                <option value="">All Status</option>
                <option value="confirmed">Confirmed</option>
                <option value="suspected">Suspected</option>
                <option value="informational">Informational</option>
            </select>
        </div>

        <div id="findingsList"></div>`;

    renderFindingsList(findings);
}

function filterFindings() {
    const q = document.getElementById('findSearch')?.value.toLowerCase() || '';
    const sev = document.getElementById('findSev')?.value || '';
    const status = document.getElementById('findStatus')?.value || '';
    const findings = State.auditResult?.findings || [];
    const filtered = findings.filter(f =>
        (!q || f.description?.toLowerCase().includes(q) || f.category?.includes(q)) &&
        (!sev || f.severity === sev) &&
        (!status || f.status === status)
    );
    renderFindingsList(filtered);
}

function renderFindingsList(findings) {
    const list = document.getElementById('findingsList');
    if (!list) return;

    if (!findings.length) {
        list.innerHTML = `<div class="empty-state"><div class="empty-icon">✅</div>
            <div class="empty-title">No matching findings</div></div>`;
        return;
    }

    list.innerHTML = findings.map((f, i) => `
        <div class="finding-card sev-${f.severity}" onclick="toggleFinding(${i})">
            <div class="finding-header">
                <span class="badge badge-${f.severity}">${f.severity}</span>
                <span class="finding-id">${f.finding_id}</span>
                <span class="badge badge-${f.status}">${f.status}</span>
                <span style="color:var(--text-muted);font-size:0.78rem;margin-left:auto">${f.category?.replace(/_/g,' ')}</span>
                ${f.auto_correctable ? '<span class="badge badge-green">🔧 auto-fixable</span>' : ''}
            </div>
            <div class="finding-desc">${escHtml(f.description)}</div>
            <div class="finding-explain">${escHtml(f.explanation || '')}</div>
            <div class="confidence-bar"><div class="confidence-fill" style="width:${((f.confidence||0)*100).toFixed(0)}%"></div></div>
            <div class="finding-details" id="fd-${i}">
                <div class="finding-recommendation"><strong>💡 Recommendation:</strong> ${escHtml(f.recommendation || '')}</div>
                ${f.affected_columns?.length ?
                    `<div class="mt-8" style="font-size:0.82rem;color:var(--text-muted)">Affected: ${f.affected_columns.map(c => `<code style="background:rgba(255,255,255,0.06);padding:1px 6px;border-radius:3px;font-family:monospace">${escHtml(c)}</code>`).join(' ')}</div>` : ''}
                ${f.evidence && Object.keys(f.evidence).length ?
                    `<div class="mt-12" style="font-size:0.82rem;color:var(--text-muted);margin-bottom:4px;font-weight:600">Evidence</div>
                     <div class="code-block">${escHtml(JSON.stringify(f.evidence, null, 2))}</div>` : ''}
                <div class="mt-8" style="font-size:0.78rem;color:var(--text-muted)">Confidence: ${((f.confidence||0)*100).toFixed(0)}%</div>
            </div>
        </div>`).join('');
}

function toggleFinding(i) {
    const el = document.getElementById(`fd-${i}`);
    if (el) el.classList.toggle('open');
}

// ══════════════════════════════════════════════════════════════════════════════
// Before/After Comparison Page
// ══════════════════════════════════════════════════════════════════════════════

function renderComparison() {
    const page = document.getElementById('page-comparison');

    if (!State.experimentResult) {
        page.innerHTML = `<div class="page-header"><h1 class="page-title">📊 Before / After</h1></div>
            <div class="empty-state"><div class="empty-icon">📊</div>
            <div class="empty-title">No experiment run yet</div>
            <button class="btn btn-primary mt-24" onclick="navigate('experiment')">🧪 Run Experiment</button></div>`;
        return;
    }

    const correctable = (State.auditResult?.findings || []).filter(f => f.auto_correctable);
    const cr = State.correctionResult;

    page.innerHTML = `
        <div class="page-header">
            <h1 class="page-title">📊 Before / After Comparison</h1>
            <p class="page-subtitle">Apply corrections and compare original vs. corrected pipeline performance.</p>
        </div>

        ${correctable.length === 0 ?
            `<div class="alert alert-success"><span class="alert-icon">✅</span>
             No auto-correctable issues found. The pipeline appears clean.</div>` :
            `<div class="card mb-16">
                <div class="section-title">🔧 Auto-Correctable Issues (${correctable.length})</div>
                ${correctable.map(f => buildFindingMini(f)).join('')}
                <div class="alert alert-warning mt-16">
                    <span class="alert-icon">⚠️</span>
                    Corrections create a <strong>separate corrected experiment</strong>. The original is preserved.
                    A drop in performance after correction typically means the original score was inflated by data leakage.
                </div>
                <button class="btn btn-primary btn-full btn-lg mt-16" id="applyBtn" onclick="applyCorrections()">
                    🔧 Apply Corrections &amp; Rerun
                </button>
            </div>`
        }

        <div id="comparisonResult" class="${cr ? '' : 'hidden'}"></div>`;

    if (cr) renderCorrectionResult(cr);
}

async function applyCorrections() {
    document.getElementById('applyBtn').disabled = true;
    showLoading('Applying corrections and rerunning experiment…');

    try {
        const result = await apiFetch('/api/corrections/apply', { method: 'POST' });
        State.correctionResult = result;

        if (result.status === 'no_corrections') {
            toast('No corrections could be generated', 'info');
            return;
        }
        if (result.status === 'failed') {
            toast('Correction failed: ' + result.error, 'error');
            return;
        }

        document.getElementById('comparisonResult').classList.remove('hidden');
        renderCorrectionResult(result);
        toast('Corrections applied successfully!', 'success');

    } catch (e) {
        toast('Correction failed: ' + e.message, 'error');
    } finally {
        hideLoading();
        const btn = document.getElementById('applyBtn');
        if (btn) btn.disabled = false;
    }
}

function renderCorrectionResult(cr) {
    const container = document.getElementById('comparisonResult');
    if (!container) return;

    const before = cr.before_metrics || {};
    const after = cr.after_metrics || {};
    const allKeys = [...new Set([...Object.keys(before), ...Object.keys(after)])].filter(k => k !== 'confusion_matrix');

    const rows = allKeys.sort().map(k => {
        const b = before[k];
        const a = after[k];
        if (typeof b !== 'number' || typeof a !== 'number') return null;
        const diff = a - b;
        const pct = b !== 0 ? (diff / Math.abs(b) * 100).toFixed(1) : '—';
        const icon = diff > 0.001 ? '📈' : diff < -0.001 ? '📉' : '➡️';
        const diffClass = diff > 0.001 ? 'text-emerald' : diff < -0.001 ? 'text-red' : '';
        return `<tr>
            <td>${k.replace(/_/g,' ').replace(/\b\w/g, c => c.toUpperCase())}</td>
            <td class="font-mono">${b.toFixed(4)}</td>
            <td class="font-mono">${a.toFixed(4)}</td>
            <td class="font-mono ${diffClass}">${icon} ${diff > 0 ? '+' : ''}${diff.toFixed(4)}</td>
            <td class="font-mono" style="color:var(--text-muted)">${pct !== '—' ? pct + '%' : '—'}</td>
        </tr>`;
    }).filter(Boolean).join('');

    container.innerHTML = `
        <div class="card mb-16">
            <div class="section-title">📊 Metric Comparison</div>
            <div class="alert alert-info mb-16">
                <span class="alert-icon">💡</span>
                A <strong>decrease</strong> in model performance after correction typically means the original score was inflated by data leakage.
                The corrected score is a more honest evaluation.
            </div>
            <div class="table-wrap">
                <table>
                    <thead><tr>
                        <th>Metric</th><th>Before</th><th>After</th><th>Change</th><th>Δ%</th>
                    </tr></thead>
                    <tbody>${rows}</tbody>
                </table>
            </div>
        </div>

        <div class="grid-2">
            <div class="card" style="text-align:center">
                <div style="font-size:2.5rem;font-weight:800;color:var(--emerald)">${cr.resolved || 0}</div>
                <div style="color:var(--text-muted);font-size:0.85rem;margin-top:6px">Issues Resolved</div>
            </div>
            <div class="card" style="text-align:center">
                <div style="font-size:2.5rem;font-weight:800;color:var(--red)">${cr.unresolved || 0}</div>
                <div style="color:var(--text-muted);font-size:0.85rem;margin-top:6px">Still Unresolved (manual action required)</div>
            </div>
        </div>

        ${cr.actions?.length ? `
        <div class="card mt-16">
            <div class="section-title">Corrections Applied</div>
            ${cr.actions.map(a => `
                <div class="card-sm mb-8" style="border-left:3px solid var(--emerald)">
                    <div style="font-weight:600;color:var(--text-primary)">${a.correction_type.replace(/_/g,' ')}</div>
                    <div style="color:var(--text-secondary);font-size:0.84rem;margin-top:4px">${escHtml(a.description)}</div>
                </div>`).join('')}
        </div>` : ''}`;
}

// ══════════════════════════════════════════════════════════════════════════════
// Benchmark Lab Page
// ══════════════════════════════════════════════════════════════════════════════

async function renderBenchmark() {
    const page = document.getElementById('page-benchmark');
    const br = State.benchmarkResult;

    // Render shell first so DOM elements exist
    page.innerHTML = `
        <div class="page-header">
            <h1 class="page-title">🎯 Benchmark Lab</h1>
            <p class="page-subtitle">Evaluate the auditor's own detection accuracy using controlled corruption experiments on real data.</p>
        </div>

        <div class="card mb-24">
            <div class="section-title">Configure Benchmark</div>
            <div class="grid-2">
                <div class="form-group">
                    <label class="form-label">Base Dataset</label>
                    <select id="bmDataset" class="form-control" onchange="updateBmTarget()">
                        <option value="">Loading datasets…</option>
                    </select>
                </div>
                <div class="form-group">
                    <label class="form-label">Task Type</label>
                    <select id="bmTask" class="form-control">
                        <option value="classification">Classification</option>
                        <option value="regression">Regression</option>
                    </select>
                </div>
            </div>
            <div class="form-group">
                <label class="form-label">Target Column</label>
                <input id="bmTarget" class="form-control" type="text" placeholder="e.g. target"
                    value="${State.currentDataset ? (State.targetCandidates.slice(-1)[0] || '') : ''}">
            </div>
            <div class="alert alert-info mb-16">
                <span class="alert-icon">ℹ️</span>
                The benchmark generates <strong>8 controlled corruption scenarios</strong> from real data with known ground truth,
                then evaluates the auditor's precision, recall, and F1 score.
            </div>
            <button class="btn btn-primary btn-full btn-lg" id="bmRunBtn" onclick="runBenchmark()">
                🎯 Run Full Benchmark Suite
            </button>
        </div>

        <div id="benchmarkResults" class="${br ? '' : 'hidden'}"></div>`;

    // Load catalog asynchronously AFTER DOM is ready
    try {
        const datasets = await apiFetch('/api/datasets/catalog');
        const sel = document.getElementById('bmDataset');
        if (sel) {
            sel.innerHTML = datasets.map(d =>
                `<option value="${encodeURIComponent(d.source)}|${encodeURIComponent(d.source_id)}|${encodeURIComponent(d.name)}|${encodeURIComponent(d.target_column||'')}">${escHtml(d.name)}</option>`
            ).join('');
            updateBmTarget();
        }
    } catch (e) {
        const sel = document.getElementById('bmDataset');
        if (sel) sel.innerHTML = '<option value="">Failed to load datasets</option>';
    }

    if (br) renderBenchmarkResults(br);
}

function updateBmTarget() {
    const sel = document.getElementById('bmDataset');
    if (!sel || !sel.value) return;
    try {
        const parts = sel.value.split('|');
        const target = parts[3] ? decodeURIComponent(parts[3]) : '';
        const targetInput = document.getElementById('bmTarget');
        if (targetInput && target) targetInput.value = target;
    } catch (e) { /* ignore decode error */ }
}

async function runBenchmark() {
    const sel = document.getElementById('bmDataset');
    if (!sel || !sel.value) { toast('Select a dataset first', 'error'); return; }
    const parts = sel.value.split('|');
    const source = decodeURIComponent(parts[0] || '');
    const source_id = decodeURIComponent(parts[1] || '');
    const nameEnc = parts[2] || '';
    const target = document.getElementById('bmTarget').value.trim();
    const task = document.getElementById('bmTask').value;

    if (!source || !source_id) { toast('Select a valid dataset', 'error'); return; }
    if (!target) { toast('Enter a target column', 'error'); return; }

    document.getElementById('bmRunBtn').disabled = true;
    showLoading('Running benchmark suite — this may take a few minutes…');

    try {
        const req = {
            dataset_source: source,
            dataset_source_id: source_id,
            dataset_name: decodeURIComponent(nameEnc),
            target_column: target,
            task_type: task,
            model_name: 'random_forest',
        };
        const result = await apiFetch('/api/benchmark/run', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(req),
        });
        State.benchmarkResult = result;
        document.getElementById('benchmarkResults').classList.remove('hidden');
        renderBenchmarkResults(result);
        toast('Benchmark complete!', 'success');
    } catch (e) {
        toast('Benchmark failed: ' + e.message, 'error');
    } finally {
        hideLoading();
        const btn = document.getElementById('bmRunBtn');
        if (btn) btn.disabled = false;
    }
}

function renderBenchmarkResults(result) {
    const container = document.getElementById('benchmarkResults');
    if (!container) return;

    const ov = result.overall || {};
    if (!ov.overall_precision && !ov.overall_recall) {
        container.innerHTML = '<div class="alert alert-warning"><span class="alert-icon">⚠️</span>Benchmark completed but no results were generated.</div>';
        return;
    }

    const f1Class = v => v >= 0.8 ? 'f1-excellent' : v >= 0.6 ? 'f1-good' : v >= 0.4 ? 'f1-poor' : 'f1-fail';

    container.innerHTML = `
        <div class="stats-grid" style="grid-template-columns:repeat(4,1fr);margin-bottom:24px">
            <div class="stat-card" style="--gradient:linear-gradient(90deg,#10b981,#06b6d4)">
                <span class="stat-icon">🎯</span>
                <div class="stat-value">${(ov.overall_precision*100).toFixed(0)}%</div>
                <div class="stat-label">Precision</div></div>
            <div class="stat-card" style="--gradient:linear-gradient(90deg,#6366f1,#a855f7)">
                <span class="stat-icon">📡</span>
                <div class="stat-value">${(ov.overall_recall*100).toFixed(0)}%</div>
                <div class="stat-label">Recall</div></div>
            <div class="stat-card" style="--gradient:linear-gradient(90deg,#a855f7,#ec4899)">
                <span class="stat-icon">⚖️</span>
                <div class="stat-value">${(ov.overall_f1*100).toFixed(0)}%</div>
                <div class="stat-label">F1 Score</div></div>
            <div class="stat-card" style="--gradient:linear-gradient(90deg,#f97316,#ef4444)">
                <span class="stat-icon">🚨</span>
                <div class="stat-value">${(ov.overall_false_positive_rate*100).toFixed(0)}%</div>
                <div class="stat-label">False Positive Rate</div></div>
        </div>

        <div class="card mb-16">
            <div class="section-title">📋 Per-Scenario Results</div>
            <div class="table-wrap">
                <table>
                    <thead><tr>
                        <th>Scenario</th><th>Ground Truth</th><th>Detected</th>
                        <th>TP</th><th>FP</th><th>FN</th>
                        <th>Precision</th><th>Recall</th><th>F1</th>
                    </tr></thead>
                    <tbody>
                    ${(result.scenarios || []).map(s => `<tr>
                        <td>${s.error ? '❌ ' : s.is_holdout ? '🔒 ' : ''}${escHtml(s.name)}</td>
                        <td>${s.ground_truth?.length ? s.ground_truth.map(g => `<span class="badge badge-high" style="margin:1px">${g.replace(/_/g,' ')}</span>`).join('') : '<span class="badge badge-green">Clean</span>'}</td>
                        <td>${s.detected?.length ? s.detected.map(g => `<span class="badge badge-purple" style="margin:1px">${g.replace(/_/g,' ')}</span>`).join('') : '—'}</td>
                        <td class="text-emerald bold">${s.true_positives?.length || 0}</td>
                        <td class="text-red bold">${s.false_positives?.length || 0}</td>
                        <td class="text-amber bold">${s.false_negatives?.length || 0}</td>
                        <td class="font-mono ${f1Class(s.precision)}">${s.precision?.toFixed(2)}</td>
                        <td class="font-mono ${f1Class(s.recall)}">${s.recall?.toFixed(2)}</td>
                        <td class="font-mono bold ${f1Class(s.f1)}">${s.f1?.toFixed(2)}</td>
                    </tr>`).join('')}
                    </tbody>
                </table>
            </div>
        </div>

        ${Object.keys(ov.per_category || {}).length ? `
        <div class="card">
            <div class="section-title">Detection by Category</div>
            <div class="table-wrap">
                <table>
                    <thead><tr><th>Category</th><th>Precision</th><th>Recall</th><th>F1</th><th>TP</th><th>FP</th><th>FN</th></tr></thead>
                    <tbody>
                    ${Object.entries(ov.per_category).map(([cat, m]) => `<tr>
                        <td>${cat.replace(/_/g,' ').replace(/\b\w/g,c=>c.toUpperCase())}</td>
                        <td class="font-mono ${f1Class(m.precision)}">${m.precision?.toFixed(2)}</td>
                        <td class="font-mono ${f1Class(m.recall)}">${m.recall?.toFixed(2)}</td>
                        <td class="font-mono bold ${f1Class(m.f1)}">${m.f1?.toFixed(2)}</td>
                        <td class="text-emerald">${m.tp}</td>
                        <td class="text-red">${m.fp}</td>
                        <td class="text-amber">${m.fn}</td>
                    </tr>`).join('')}
                    </tbody>
                </table>
            </div>
        </div>` : ''}`;
}

// ══════════════════════════════════════════════════════════════════════════════
// Reports Page
// ══════════════════════════════════════════════════════════════════════════════

function renderReports() {
    const page = document.getElementById('page-reports');
    const hasData = !!State.experimentResult && !!State.auditResult;

    page.innerHTML = `
        <div class="page-header">
            <h1 class="page-title">📄 Audit Reports</h1>
            <p class="page-subtitle">Generate and download reproducible reports with full evidence, provenance, and software versions.</p>
        </div>

        ${!hasData ? `<div class="alert alert-warning"><span class="alert-icon">⚠️</span>Run an experiment and audit first.</div>` : ''}

        <div class="grid-2 mb-24" ${!hasData ? 'style="opacity:0.5;pointer-events:none"' : ''}>
            <div class="card">
                <div class="section-title">📑 JSON Report</div>
                <p class="text-muted mb-16">Machine-readable full audit report with all findings, metrics, evidence, and software versions. Suitable for CI/CD integration.</p>
                <button class="btn btn-primary btn-full" onclick="downloadReport('json')" ${!hasData ? 'disabled' : ''}>
                    ⬇️ Download JSON Report
                </button>
            </div>
            <div class="card">
                <div class="section-title">🌐 HTML Report</div>
                <p class="text-muted mb-16">Styled, self-contained HTML report suitable for sharing with stakeholders and inclusion in research papers.</p>
                <button class="btn btn-success btn-full" onclick="downloadReport('html')" ${!hasData ? 'disabled' : ''}>
                    ⬇️ Download HTML Report
                </button>
            </div>
        </div>

        ${hasData ? `
        <div class="card">
            <div class="section-title">📋 Report Contents</div>
            <div class="grid-2" style="font-size:0.85rem;gap:8px">
                <div>✅ Dataset provenance (name, source, license, row/col count)</div>
                <div>✅ Experiment configuration (seed, splits, CV folds)</div>
                <div>✅ All model metrics (train, test, CV)</div>
                <div>✅ Baseline comparisons</div>
                <div>✅ All ${State.auditResult?.findings?.length || 0} findings with evidence</div>
                <div>✅ Before/after correction comparison</div>
                <div>✅ Software version snapshot</div>
                <div>✅ Benchmark results (if run)</div>
            </div>
        </div>` : ''}`;
}

async function downloadReport(format) {
    showLoading(`Generating ${format.toUpperCase()} report…`);
    try {
        const res = await fetch(`/api/reports/generate?format=${format}`, { method: 'POST' });
        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: res.statusText }));
            throw new Error(err.detail || res.statusText);
        }
        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        // Use dataset name in filename if available
        const dsName = (State.currentDataset?.name || 'audit').replace(/[^a-z0-9_-]/gi, '_').toLowerCase();
        const ts = new Date().toISOString().slice(0,10);
        a.download = `${dsName}_audit_${ts}.${format}`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        toast(`${format.toUpperCase()} report downloaded!`, 'success');
    } catch (e) {
        toast('Report generation failed: ' + e.message, 'error');
    } finally {
        hideLoading();
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Helpers
// ══════════════════════════════════════════════════════════════════════════════

function escHtml(str) {
    return String(str || '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function buildPreviewTable(data, columns) {
    if (!data?.length) return '<p class="text-muted">No preview available</p>';
    return `<div class="preview-table-wrap"><table>
        <thead><tr>${columns.map(c => `<th>${escHtml(c)}</th>`).join('')}</tr></thead>
        <tbody>${data.slice(0,20).map(row =>
            `<tr>${columns.map(c => `<td>${escHtml(String(row[c] ?? ''))}</td>`).join('')}</tr>`
        ).join('')}</tbody>
    </table></div>`;
}

function buildCategoryBars(byCat) {
    if (!byCat || !Object.keys(byCat).length) return '<p class="text-muted">No data</p>';
    const max = Math.max(...Object.values(byCat));
    return Object.entries(byCat).map(([cat, cnt]) => `
        <div style="margin-bottom:10px">
            <div style="display:flex;justify-content:space-between;font-size:0.8rem;margin-bottom:4px">
                <span style="color:var(--text-secondary)">${cat.replace(/_/g,' ')}</span>
                <span style="color:var(--text-primary);font-weight:700">${cnt}</span>
            </div>
            <div style="height:6px;background:var(--border);border-radius:3px;overflow:hidden">
                <div style="width:${(cnt/max*100).toFixed(1)}%;height:100%;background:linear-gradient(90deg,var(--accent),var(--purple));border-radius:3px"></div>
            </div>
        </div>`).join('');
}

function buildSeverityBars(bySev) {
    const colors = { critical: 'var(--red)', high: 'var(--orange)', medium: 'var(--amber)', low: 'var(--accent)', info: 'var(--text-muted)' };
    if (!bySev || !Object.keys(bySev).length) return '<p class="text-muted">No data</p>';
    const max = Math.max(...Object.values(bySev));
    return Object.entries(bySev).map(([sev, cnt]) => `
        <div style="margin-bottom:10px">
            <div style="display:flex;justify-content:space-between;font-size:0.8rem;margin-bottom:4px">
                <span style="color:${colors[sev] || 'var(--text-muted)'}">${sev}</span>
                <span style="color:var(--text-primary);font-weight:700">${cnt}</span>
            </div>
            <div style="height:6px;background:var(--border);border-radius:3px;overflow:hidden">
                <div style="width:${(cnt/max*100).toFixed(1)}%;height:100%;background:${colors[sev]||'var(--text-muted)'};border-radius:3px"></div>
            </div>
        </div>`).join('');
}

// ══════════════════════════════════════════════════════════════════════════════
// Boot — restore session state from server, then navigate
// ══════════════════════════════════════════════════════════════════════════════

document.addEventListener('DOMContentLoaded', async () => {
    // Check if server still has a dataset loaded (handles page refreshes)
    try {
        const current = await apiFetch('/api/datasets/current');
        if (current.loaded) {
            State.currentDataset = {
                source: current.id?.split('_')[0] || 'unknown',
                source_id: current.id || '',
                name: current.name || 'Loaded Dataset',
            };
            State.columns = current.columns || [];
            State.dtypes = current.dtypes || {};
        }
    } catch (e) { /* server may not have state — ok */ }
    navigate('dashboard');
});
