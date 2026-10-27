/**
 * SmartChunk — Frontend Application Logic
 * ═══════════════════════════════════════════════════════════════════════
 * Handles: tab navigation, file upload, chunk comparison, heatmap,
 *          diagnostics loop, and interactive Q&A with agent trace.
 */

(function () {
  "use strict";

  const API = "";  // same-origin

  // ── State ───────────────────────────────────────────────────────────
  let currentChunkData = null;
  let currentRunId = null;
  let pollTimer = null;

  // ══════════════════════════════════════════════════════════════════════
  //  TAB NAVIGATION
  // ══════════════════════════════════════════════════════════════════════

  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
      btn.classList.add("active");
      document.getElementById(`panel-${btn.dataset.tab}`).classList.add("active");
    });
  });

  // ══════════════════════════════════════════════════════════════════════
  //  FILE UPLOAD / DROPZONE
  // ══════════════════════════════════════════════════════════════════════

  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("file-input");

  ["dragenter", "dragover"].forEach((e) =>
    dropzone.addEventListener(e, (ev) => {
      ev.preventDefault();
      dropzone.classList.add("drag-over");
    })
  );

  ["dragleave", "drop"].forEach((e) =>
    dropzone.addEventListener(e, (ev) => {
      ev.preventDefault();
      dropzone.classList.remove("drag-over");
    })
  );

  dropzone.addEventListener("drop", (ev) => {
    const file = ev.dataTransfer.files[0];
    if (file) handleFileUpload(file);
  });

  fileInput.addEventListener("change", () => {
    if (fileInput.files[0]) handleFileUpload(fileInput.files[0]);
  });

  async function handleFileUpload(file) {
    showLoading("Chunking " + file.name + "…");
    const form = new FormData();
    form.append("file", file);

    try {
      const res = await fetch(`${API}/api/chunk`, { method: "POST", body: form });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      currentChunkData = data;

      renderStats(data.stats);
      renderChunks(data.naive_chunks, data.structure_aware_chunks);
      renderHeatmap(data.structure_aware_chunks);
      showTokenFilter(data);

      // Auto-ingest into search index
      await fetch(`${API}/api/ingest`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          children: data.children || [],
          parents: data.parents || [],
        }),
      });
    } catch (err) {
      alert("Upload failed: " + err.message);
    } finally {
      hideLoading();
    }
  }

  // ══════════════════════════════════════════════════════════════════════
  //  STATS
  // ══════════════════════════════════════════════════════════════════════

  function renderStats(stats) {
    const grid = document.getElementById("stats-grid");
    const n = stats.naive;
    const s = stats.structure_aware;
    const saved = stats.tokens_saved !== undefined ? stats.tokens_saved : Math.max(0, (n.total_tokens || 0) - (s.total_tokens || 0));

    grid.innerHTML = `
      <div class="stat-pill accent">
        <div class="stat-value">${n.total_tokens || 0}</div>
        <div class="stat-label">Total Tokens (Naive)</div>
      </div>
      <div class="stat-pill emerald">
        <div class="stat-value">${s.total_tokens || 0}</div>
        <div class="stat-label">Total Tokens (Struct)</div>
      </div>
      <div class="stat-pill emerald">
        <div class="stat-value">-${saved}</div>
        <div class="stat-label">Tokens Saved</div>
      </div>
      <div class="stat-pill cyan">
        <div class="stat-value">${n.avg_token_count}</div>
        <div class="stat-label">Avg Tokens (Naive)</div>
      </div>
      <div class="stat-pill emerald">
        <div class="stat-value">${s.avg_token_count}</div>
        <div class="stat-label">Avg Tokens (Struct)</div>
      </div>
      <div class="stat-pill rose">
        <div class="stat-value">${n.split_code_blocks}/${n.total_code_blocks}</div>
        <div class="stat-label">Code Splits (Naive)</div>
      </div>
      <div class="stat-pill emerald">
        <div class="stat-value">${s.split_code_blocks}/${s.total_code_blocks}</div>
        <div class="stat-label">Code Splits (Struct)</div>
      </div>
      <div class="stat-pill rose">
        <div class="stat-value">${n.split_tables}/${n.total_tables}</div>
        <div class="stat-label">Table Splits (Naive)</div>
      </div>
      <div class="stat-pill emerald">
        <div class="stat-value">${s.split_tables}/${s.total_tables}</div>
        <div class="stat-label">Table Splits (Struct)</div>
      </div>
    `;

    document.getElementById("chunk-stats").classList.remove("hidden");
  }

  // ══════════════════════════════════════════════════════════════════════
  //  CHUNK COMPARISON
  // ══════════════════════════════════════════════════════════════════════

  function renderChunks(naive, structured, minTokens = 0) {
    const container = document.getElementById("chunk-compare");
    container.classList.remove("hidden");

    const filterFn = (c) => (c.token_count || 0) >= minTokens;

    container.innerHTML = `
      <div class="chunk-column naive">
        <h3>Naive Chunking <span class="badge">${naive.filter(filterFn).length} chunks</span></h3>
        <div class="chunk-list">
          ${naive.filter(filterFn).map(chunkCard).join("")}
        </div>
      </div>
      <div class="chunk-column structure">
        <h3>Structure-Aware <span class="badge">${structured.filter(filterFn).length} chunks</span></h3>
        <div class="chunk-list">
          ${structured.filter(filterFn).map(chunkCard).join("")}
        </div>
      </div>
    `;

    // Click to expand
    container.querySelectorAll(".chunk-card").forEach((card) => {
      card.addEventListener("click", () => card.classList.toggle("expanded"));
    });
  }

  function chunkCard(chunk) {
    const meta = chunk.metadata || {};
    const hasCode = meta.has_code || chunk.text.includes("```");
    const hasTable = meta.has_table || (chunk.text.includes("|") && chunk.text.split("|").length > 3);
    const headingPath = chunk.heading_path || meta.heading_path || "";
    const tokens = chunk.token_count || Math.ceil(chunk.text.length / 4);

    let badges = "";
    if (hasCode) badges += `<span class="split-badge code">Code</span>`;
    if (hasTable) badges += `<span class="split-badge table">Table</span>`;
    if (!hasCode && !hasTable) badges += `<span class="split-badge clean">Clean</span>`;

    return `
      <div class="chunk-card" data-id="${chunk.id}">
        <div class="chunk-header">
          <span class="chunk-id">${chunk.id}</span>
          <span class="chunk-tokens">${tokens} tokens</span>
        </div>
        ${headingPath ? `<div class="chunk-path">${escapeHtml(headingPath)}</div>` : ""}
        <div class="chunk-text">${escapeHtml(chunk.text)}</div>
        <div class="chunk-badges">${badges}</div>
      </div>
    `;
  }

  // ── Token filter ──────────────────────────────────────────────────
  function showTokenFilter(data) {
    const bar = document.getElementById("token-filter-bar");
    bar.style.display = "flex";

    const slider = document.getElementById("token-filter");
    const label = document.getElementById("token-filter-value");
    const allChunks = [...(data.naive_chunks || []), ...(data.structure_aware_chunks || [])];
    const maxTok = Math.max(...allChunks.map((c) => c.token_count || 0), 1);
    slider.max = maxTok;
    slider.value = 0;
    label.textContent = "0";

    slider.oninput = () => {
      label.textContent = slider.value;
      renderChunks(data.naive_chunks, data.structure_aware_chunks, parseInt(slider.value));
    };
  }

  // ══════════════════════════════════════════════════════════════════════
  //  HEATMAP
  // ══════════════════════════════════════════════════════════════════════

  function renderHeatmap(chunks) {
    const area = document.getElementById("heatmap-area");
    if (!chunks || !chunks.length) {
      area.innerHTML = `<p class="text-muted text-sm">No chunks to display.</p>`;
      return;
    }

    let cells = "";
    chunks.forEach((chunk, i) => {
      const meta = chunk.metadata || {};
      const hasCode = meta.has_code;
      const hasTable = meta.has_table;
      const tokens = chunk.token_count || 0;

      let quality = "high";
      if (hasCode && chunk.text.split("```").length % 2 === 0) quality = "low";
      else if (hasTable && chunk.text.split("|").length < 4) quality = "low";
      else if (tokens > 300) quality = "medium";
      else if (tokens < 20) quality = "medium";

      const heading = chunk.heading_path || meta.heading_path || "(root)";
      cells += `
        <div class="heatmap-cell quality-${quality}" data-index="${i}"
             title="Chunk ${i}: ${tokens} tokens">
          <div class="tooltip">
            <strong>${chunk.id}</strong><br/>
            ${tokens} tokens · ${heading}
            ${hasCode ? " · Code" : ""}${hasTable ? " · Table" : ""}
          </div>
        </div>
      `;
    });

    area.innerHTML = `<div class="heatmap-container">${cells}</div>`;

    // Click handler for boundary detail
    area.querySelectorAll(".heatmap-cell").forEach((cell) => {
      cell.addEventListener("click", () => {
        const idx = parseInt(cell.dataset.index);
        showBoundaryDetail(chunks, idx);
      });
    });
  }

  function showBoundaryDetail(chunks, index) {
    const detail = document.getElementById("boundary-detail");
    const content = document.getElementById("boundary-content");
    detail.classList.remove("hidden");

    const chunk = chunks[index];
    const prev = index > 0 ? chunks[index - 1] : null;
    const next = index < chunks.length - 1 ? chunks[index + 1] : null;

    const meta = chunk.metadata || {};

    content.innerHTML = `
      <div style="display:grid;grid-template-columns:1fr 2fr 1fr;gap:12px;">
        ${prev ? `
          <div class="chunk-card" style="opacity:0.5;">
            <div class="chunk-header"><span class="chunk-id">${prev.id}</span></div>
            <div class="chunk-text">${escapeHtml((prev.text || "").slice(-200))}</div>
          </div>
        ` : `<div></div>`}
        <div class="chunk-card" style="border-color:var(--accent);box-shadow:0 0 20px var(--accent-glow);">
          <div class="chunk-header">
            <span class="chunk-id">${chunk.id}</span>
            <span class="chunk-tokens">${chunk.token_count || 0} tokens</span>
          </div>
          ${meta.heading_path ? `<div class="chunk-path">${escapeHtml(meta.heading_path)}</div>` : ""}
          <div class="chunk-text" style="max-height:none;mask-image:none;-webkit-mask-image:none;">
            ${escapeHtml(chunk.text)}
          </div>
        </div>
        ${next ? `
          <div class="chunk-card" style="opacity:0.5;">
            <div class="chunk-header"><span class="chunk-id">${next.id}</span></div>
            <div class="chunk-text">${escapeHtml((next.text || "").slice(0, 200))}</div>
          </div>
        ` : `<div></div>`}
      </div>
    `;
  }

  // ══════════════════════════════════════════════════════════════════════
  //  DIAGNOSTICS & HEALING LOOP
  // ══════════════════════════════════════════════════════════════════════

  document.getElementById("btn-start-loop").addEventListener("click", startHealingLoop);

  async function startHealingLoop() {
    const statusEl = document.getElementById("loop-status");
    const spinner = document.getElementById("loop-spinner");

    statusEl.textContent = "Starting loop…";
    spinner.classList.remove("hidden");

    try {
      const res = await fetch(`${API}/loop`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ queries: [], expected_chunk_ids: [], dev_ratio: 0.7 }),
      });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      currentRunId = data.run_id;
      statusEl.textContent = `Run ${currentRunId} started`;

      // Poll for status
      pollTimer = setInterval(() => pollRunStatus(currentRunId), 2000);
    } catch (err) {
      statusEl.textContent = "Error: " + err.message;
      spinner.classList.add("hidden");
    }
  }

  async function pollRunStatus(runId) {
    try {
      const res = await fetch(`${API}/run/${runId}/status`);
      if (!res.ok) return;
      const data = await res.json();

      const statusEl = document.getElementById("loop-status");
      statusEl.textContent = `Run ${runId}: ${data.status}`;

      if (data.status === "done" || data.status === "failed") {
        clearInterval(pollTimer);
        document.getElementById("loop-spinner").classList.add("hidden");

        if (data.report) {
          renderDiagReport(data.report);
        }
      }

      if (data.status === "awaiting_review" && data.pending_reviews) {
        renderReviewUI(runId, data.pending_reviews);
      }

      if (data.trace) {
        renderFixLog(data.trace);
      }
    } catch (_) { /* ignore transient errors */ }
  }

  function renderDiagReport(report) {
    // Dev rate
    const devBefore = (report.dev_hit_rate_before * 100).toFixed(1);
    const devAfter = (report.dev_hit_rate_after * 100).toFixed(1);
    document.getElementById("dev-rate").textContent = devAfter || devBefore;
    document.getElementById("dev-progress").style.width = `${devAfter || devBefore}%`;

    if (devAfter && devBefore) {
      const change = (devAfter - devBefore).toFixed(1);
      const el = document.getElementById("dev-rate-change");
      el.textContent = `${change > 0 ? "+" : ""}${change}%`;
      el.className = `rate-change ${change >= 0 ? "positive" : "negative"}`;
      el.classList.remove("hidden");
    }

    // Holdout rate
    const hBefore = (report.holdout_hit_rate_before * 100).toFixed(1);
    const hAfter = (report.holdout_hit_rate_after * 100).toFixed(1);
    document.getElementById("holdout-rate").textContent = hAfter || hBefore;
    document.getElementById("holdout-progress").style.width = `${hAfter || hBefore}%`;

    if (hAfter && hBefore) {
      const change = (hAfter - hBefore).toFixed(1);
      const el = document.getElementById("holdout-rate-change");
      el.textContent = `${change > 0 ? "+" : ""}${change}%`;
      el.className = `rate-change ${change >= 0 ? "positive" : "negative"}`;
      el.classList.remove("hidden");
    }

    // Failure breakdown
    if (report.diagnoses && report.diagnoses.length) {
      const counts = {};
      report.diagnoses.forEach((d) => {
        counts[d.root_cause] = (counts[d.root_cause] || 0) + 1;
      });

      const bd = document.getElementById("failure-breakdown");
      bd.innerHTML = `
        <div class="stat-grid">
          ${Object.entries(counts)
          .map(
            ([cause, cnt]) => `
              <div class="stat-pill">
                <div class="stat-value">${cnt}</div>
                <div class="stat-label"><span class="fix-cause ${cause}">${cause}</span></div>
              </div>`
          )
          .join("")}
        </div>
      `;
    }
  }

  function renderFixLog(trace) {
    const log = document.getElementById("fix-log");
    if (!trace.length) return;

    log.innerHTML = trace
      .map(
        (step) => `
        <li class="fix-entry">
          <span class="fix-cause ${step.agent}">${step.agent}</span>
          <span class="fix-action">${step.action}: ${escapeHtml(step.detail || "")}</span>
          <span class="fix-status text-sm text-muted">${step.duration_ms?.toFixed(0) || 0}ms</span>
        </li>`
      )
      .join("");
  }

  function renderReviewUI(runId, reviews) {
    const bd = document.getElementById("failure-breakdown");
    let html = `<h4 class="mb-4" style="color:var(--amber)">Awaiting Human Review</h4>`;
    reviews.forEach((r, i) => {
      html += `
        <div class="fix-entry" style="flex-direction:column;align-items:stretch;gap:8px;">
          <div>
            <strong>Boundary:</strong> ${r.chunk_a_id} ↔ ${r.chunk_b_id}
            <span class="text-muted text-sm"> — ${r.reason}</span>
          </div>
          <div style="display:flex;gap:8px;">
            <button class="btn btn-sm review-btn" data-run="${runId}" data-chunk="${r.chunk_a_id}" data-decision="accept">Accept</button>
            <button class="btn btn-sm btn-primary review-btn" data-run="${runId}" data-chunk="${r.chunk_a_id}" data-decision="merge">Merge</button>
          </div>
        </div>
      `;
    });
    bd.innerHTML = html;

    bd.querySelectorAll(".review-btn").forEach((btn) => {
      btn.addEventListener("click", async () => {
        const decisions = [
          { chunk_id: btn.dataset.chunk, decision: btn.dataset.decision },
        ];
        await fetch(`${API}/review/${btn.dataset.run}`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ decisions }),
        });
        btn.parentElement.innerHTML = `<span class="text-sm" style="color:var(--emerald)">${btn.dataset.decision}</span>`;
      });
    });
  }

  // ══════════════════════════════════════════════════════════════════════
  //  ASK / Q&A
  // ══════════════════════════════════════════════════════════════════════

  const chatInput = document.getElementById("chat-input");
  const btnAsk = document.getElementById("btn-ask");
  const chatMessages = document.getElementById("chat-messages");

  btnAsk.addEventListener("click", submitQuestion);
  chatInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submitQuestion();
    }
  });

  async function submitQuestion() {
    const query = chatInput.value.trim();
    if (!query) return;

    // Add user message
    appendMessage("user", query);
    chatInput.value = "";

    // Show typing indicator
    const typingId = appendTyping();

    try {
      const searchMode = document.getElementById("search-mode").value;
      const res = await fetch(`${API}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, top_k: 5, search_mode: searchMode }),
      });

      removeTyping(typingId);

      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();

      // Add assistant message
      appendAnswer(data);

      // Render trace
      if (data.trace) renderTrace(data.trace);
    } catch (err) {
      removeTyping(typingId);
      appendMessage("assistant", "Error: " + err.message);
    }
  }

  function appendMessage(role, text) {
    const avatar = role === "user" ? "YOU" : "AI";
    const div = document.createElement("div");
    div.className = `chat-msg ${role}`;
    div.innerHTML = `
      <div class="chat-avatar">${avatar}</div>
      <div class="chat-bubble">${escapeHtml(text)}</div>
    `;
    chatMessages.appendChild(div);
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }

  function appendAnswer(data) {
    const div = document.createElement("div");
    div.className = "chat-msg assistant";

    let citationHtml = "";
    if (data.citations && data.citations.length) {
      citationHtml = `
        <div class="citation-chips">
          ${data.citations
          .map(
            (c) => `
              <span class="citation-chip" title="${escapeHtml(c.text_preview || "")}">
                <span class="chip-icon">[Source]</span>
                ${escapeHtml(c.source || c.chunk_id)}
                ${c.heading_path ? ` · ${escapeHtml(c.heading_path)}` : ""}
                ${c.line_start ? ` · L${c.line_start}` : ""}
              </span>`
          )
          .join("")}
        </div>
      `;
    }

    if (data.null_query) {
      citationHtml += `<div class="text-sm" style="color:var(--amber);margin-top:8px;">Below null-query threshold — no relevant results.</div>`;
    }

    div.innerHTML = `
      <div class="chat-avatar">AI</div>
      <div class="chat-bubble">
        <div style="white-space:pre-wrap">${formatAnswer(data.answer || "No answer generated.")}</div>
        ${citationHtml}
      </div>
    `;

    chatMessages.appendChild(div);
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }

  function formatAnswer(text) {
    // Basic markdown formatting
    return escapeHtml(text)
      .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
      .replace(/`([^`]+)`/g, '<code style="background:var(--bg-elevated);padding:1px 5px;border-radius:3px;">$1</code>')
      .replace(/---/g, '<hr style="border-color:var(--border);margin:8px 0;">');
  }

  let typingCounter = 0;
  function appendTyping() {
    const id = `typing-${++typingCounter}`;
    const div = document.createElement("div");
    div.className = "chat-msg assistant";
    div.id = id;
    div.innerHTML = `
      <div class="chat-avatar">AI</div>
      <div class="chat-bubble" style="display:flex;gap:4px;align-items:center;">
        <span class="text-muted text-sm">Thinking…</span>
      </div>
    `;
    chatMessages.appendChild(div);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return id;
  }

  function removeTyping(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
  }

  function renderTrace(trace) {
    const tbody = document.getElementById("trace-tbody");
    tbody.innerHTML = trace
      .map(
        (step) => `
        <tr>
          <td class="agent-name">${escapeHtml(step.agent)}</td>
          <td>${escapeHtml(step.action)}</td>
          <td class="text-muted">${escapeHtml(step.detail || "")}</td>
          <td class="text-muted">${(step.duration_ms || 0).toFixed(0)}ms</td>
        </tr>`
      )
      .join("");
  }

  // ══════════════════════════════════════════════════════════════════════
  //  UTILITIES
  // ══════════════════════════════════════════════════════════════════════

  function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str || "";
    return div.innerHTML;
  }

  function showLoading(text) {
    document.getElementById("loading-text").textContent = text || "Processing…";
    document.getElementById("loading-overlay").classList.add("active");
  }

  function hideLoading() {
    document.getElementById("loading-overlay").classList.remove("active");
  }
})();
