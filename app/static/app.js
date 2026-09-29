const form = document.getElementById("ask-form");
const emptyState = document.getElementById("empty-state");
const decisionCard = document.getElementById("decision-card");
const answerCard = document.getElementById("answer-card");
const whyCard = document.getElementById("why-card");
const policyCard = document.getElementById("policy-card");
const evidenceCard = document.getElementById("evidence-card");
const runtimeCard = document.getElementById("runtime-card");
const compareCard = document.getElementById("compare-card");
const strategyDescription = document.getElementById("strategy-description");
const strategyOptions = JSON.parse(document.getElementById("strategy-options").textContent || "[]");

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function fmt(value, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "n/a";
  return Number(value).toFixed(digits);
}

function pct(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "n/a";
  return `${Math.round(Number(value) * 100)}%`;
}

function money(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "n/a";
  if (Number(value) === 0) return "$0.0000";
  return `$${Number(value).toFixed(5)}`;
}

function hideAll() {
  emptyState.classList.add("hidden");
  for (const card of [decisionCard, answerCard, whyCard, policyCard, evidenceCard, runtimeCard, compareCard]) {
    card.classList.add("hidden");
    card.innerHTML = "";
  }
}

function setLoading() {
  hideAll();
  decisionCard.classList.remove("hidden");
  decisionCard.innerHTML = `
    <div class="decision-top">
      <div>
        <p class="section-kicker">Running</p>
        <h2>Loading saved results or running the selected strategy...</h2>
      </div>
      <span class="badge action">Local corpus</span>
    </div>
    <div class="decision-metrics">
      <div class="mini-metric"><span>Step</span><strong>Retrieval</strong><small>Hybrid, direct, or policy route</small></div>
      <div class="mini-metric"><span>Status</span><strong>Working</strong><small>Generation may take longer</small></div>
      <div class="mini-metric"><span>External Search</span><strong>Off</strong><small>Controlled corpus only</small></div>
      <div class="mini-metric"><span>Output</span><strong>Pending</strong><small>Decision, answer, evidence</small></div>
    </div>`;
}

function updateStrategyDescription() {
  const mode = document.getElementById("mode").value;
  const option = strategyOptions.find((item) => item.id === mode);
  strategyDescription.textContent = option ? option.description : "";
}

function badgeForAction(action) {
  if (action === "ABSTAIN") return "badge danger";
  if (action === "ESCALATE") return "badge warn";
  return "badge action";
}

function renderDecision(data) {
  const warnings = (data.warnings || [])
    .map((item) => `<span class="badge warn">${escapeHtml(item)}</span>`)
    .join("");
  decisionCard.classList.remove("hidden");
  decisionCard.innerHTML = `
    <div class="decision-top">
      <div>
        <p class="section-kicker">Decision</p>
        <h2>${escapeHtml(data.final_action)}</h2>
      </div>
      <div class="badge-row">
        <span class="${badgeForAction(data.final_action)}">Action ${escapeHtml(data.final_action)}</span>
        <span class="badge">${escapeHtml(data.strategy_label)}</span>
        <span class="badge">Controlled corpus</span>
      </div>
    </div>
    <div class="decision-metrics">
      <div class="mini-metric"><span>Action</span><strong>${escapeHtml(data.final_action)}</strong><small>Runtime decision</small></div>
      <div class="mini-metric"><span>Strategy</span><strong>${escapeHtml(data.strategy_label)}</strong><small>${escapeHtml(data.mode)}</small></div>
      <div class="mini-metric"><span>Retrieval</span><strong>${fmt(data.retrieval_latency_ms)} ms</strong><small>${escapeHtml(data.retrieval_label || "None")}</small></div>
      <div class="mini-metric"><span>Top K</span><strong>${escapeHtml(data.top_k)}</strong><small>${data.policy_stages?.length ? escapeHtml(data.diagnostics?.policy_version) : "Configured request"}</small></div>
    </div>
    ${warnings ? `<div class="warning-list">${warnings}</div>` : ""}`;
}

function renderAnswer(data) {
  const gen = data.generation || {};
  const status = gen.status || "unknown";
  const isAbstain = data.final_action === "ABSTAIN";
  const label = isAbstain ? "Insufficient Evidence" : "Answer";
  const body = isAbstain
    ? "The system chose not to provide an answer after the frozen routing stages. These stage scores are not calibrated answer-confidence estimates."
    : (gen.answer || gen.error || (status === "skipped" ? "Answer generation was skipped. Retrieved evidence is still shown below." : "No answer generated."));
  const badgeClass = status === "error" || status === "not_configured" ? "badge warn" : (isAbstain ? "badge danger" : "badge action");

  answerCard.classList.remove("hidden");
  answerCard.innerHTML = `
    <div class="decision-top">
      <div>
        <p class="section-kicker">${escapeHtml(label)}</p>
        <h2>${escapeHtml(label)}</h2>
      </div>
      <span class="${badgeClass}">${escapeHtml(status)}</span>
    </div>
    <div class="answer-body ${isAbstain ? "abstain" : ""}">${escapeHtml(body)}</div>`;
}

function renderWhy(data) {
  whyCard.classList.remove("hidden");
  const path = (data.decision_path || [])
    .map((step) => `
      <div class="path-step">
        <b>${escapeHtml(step.label)}</b>
        <span>${escapeHtml(step.status)}</span>
      </div>`)
    .join("");
  whyCard.innerHTML = `
    <p class="section-kicker">Why this action?</p>
    <h2>Decision rationale</h2>
    <p>${escapeHtml(data.why || "No rationale was returned for this run.")}</p>
    ${path ? `<div class="path">${path}</div>` : ""}`;
}

function renderPolicy(data) {
  if (!data.policy_stages?.length && !data.routing_signals?.length) {
    policyCard.classList.add("hidden");
    return;
  }
  const stages = (data.policy_stages || [])
    .map((stage) => {
      const width = stage.probability === null || stage.probability === undefined ? 0 : Math.max(0, Math.min(100, Number(stage.probability) * 100));
      return `
        <div class="stage ${stage.reached ? "" : "not-reached"}">
          <span>${escapeHtml(stage.label)}</span>
          <strong>${stage.reached ? pct(stage.probability) : "Not evaluated"}</strong>
          <small>Threshold: ${stage.threshold === null || stage.threshold === undefined ? "n/a" : pct(stage.threshold)} · ${escapeHtml(stage.decision)}</small>
          <div class="progress"><i style="width:${width}%"></i></div>
        </div>`;
    })
    .join("");
  const signals = (data.routing_signals || [])
    .map((signal) => `
      <div class="signal">
        <span>${escapeHtml(signal.family)}</span>
        <strong>${escapeHtml(signal.label)}</strong>
        <small>Value: ${fmt(signal.value, Math.abs(Number(signal.value)) < 1 ? 3 : 1)} · ${escapeHtml(signal.direction)}</small>
        ${signal.contribution === null || signal.contribution === undefined ? "" : `<small>Stage 1 contribution: ${fmt(signal.contribution, 3)}</small>`}
      </div>`)
    .join("");

  policyCard.classList.remove("hidden");
  policyCard.innerHTML = `
    <p class="section-kicker">Routing Decision</p>
    <h2>Policy stage scores</h2>
    <div class="stage-grid">${stages}</div>
    ${signals ? `
      <details>
        <summary>Routing Signals</summary>
        <div class="signal-grid">${signals}</div>
      </details>` : ""}`;
}

function renderEvidence(data) {
  const documents = data.documents || [];
  evidenceCard.classList.remove("hidden");
  if (!documents.length) {
    evidenceCard.innerHTML = `
      <p class="section-kicker">Retrieved Evidence</p>
      <h2>No retrieved documents</h2>
      <p>This strategy did not invoke retrieval for the current question.</p>`;
    return;
  }
  const docs = documents
    .map((doc) => `
      <article class="doc-item">
        <div class="doc-head">
          <span class="rank">${escapeHtml(doc.rank)}</span>
          <div>
            <h3>${escapeHtml(doc.title || `Document ${doc.rank}`)}</h3>
            <div class="doc-meta">
              <span class="badge">${escapeHtml(doc.score_label || "Score")} ${doc.score === null || doc.score === undefined ? "n/a" : fmt(doc.score, 4)}</span>
              <span class="badge">${escapeHtml(doc.domain || data.domain)}</span>
            </div>
          </div>
          <span class="badge">Rank ${escapeHtml(doc.rank)}</span>
        </div>
        <p>${escapeHtml(doc.snippet)}</p>
        <details>
          <summary>Expand passage</summary>
          <div class="doc-full">${escapeHtml(doc.text || doc.snippet)}</div>
          <div class="doc-meta">
            <span class="badge">Document ID: ${escapeHtml(doc.document_id)}</span>
            ${doc.source ? `<span class="badge">Source: ${escapeHtml(doc.source)}</span>` : ""}
          </div>
        </details>
      </article>`)
    .join("");
  evidenceCard.innerHTML = `
    <p class="section-kicker">Retrieved Evidence</p>
    <h2>Evidence used by the run</h2>
    <div class="docs-list">${docs}</div>`;
}

function renderRuntime(data) {
  const runtime = data.runtime || {};
  runtimeCard.classList.remove("hidden");
  runtimeCard.innerHTML = `
    <p class="section-kicker">Runtime</p>
    <h2>${runtime.replay ? "Saved experiment cost and latency" : "Live request latency"}</h2>
    <p>${escapeHtml(runtime.note)}</p>
    <div class="runtime-grid">
      <div class="mini-metric"><span>Retrieval</span><strong>${fmt(runtime.retrieval_ms)} ms</strong><small>${runtime.replay ? "Historical experiment timing" : "Live local retrieval"}</small></div>
      <div class="mini-metric"><span>Reranking</span><strong>${fmt(runtime.reranking_ms)} ms</strong><small>When invoked</small></div>
      <div class="mini-metric"><span>Generation</span><strong>${fmt(runtime.generation_ms)} ms</strong><small>${escapeHtml(data.generation?.status || "unknown")}</small></div>
      <div class="mini-metric"><span>Total</span><strong>${fmt(runtime.total_ms)} ms</strong><small>${runtime.replay ? "Unavailable for saved Batch calls" : "Measured app request"}</small></div>
    </div>
    <details>
      <summary>Advanced Diagnostics</summary>
      <div class="signal-grid">
        <div class="signal"><span>Internal Strategy</span><strong>${escapeHtml(data.diagnostics?.internal_mode)}</strong><small>${escapeHtml(data.retrieval_method || "none")}</small></div>
        <div class="signal"><span>Generator</span><strong>${escapeHtml(data.diagnostics?.generator)}</strong><small>Historical Batch cost: ${money(runtime.estimated_api_cost_usd)}</small></div>
        <div class="signal"><span>Policy</span><strong>${escapeHtml(data.diagnostics?.policy_version)}</strong><small>Frozen artifact preserved</small></div>
        <div class="signal"><span>Live Web</span><strong>Disabled</strong><small>external_retrieval_enabled=false</small></div>
      </div>
    </details>`;
}

function renderResult(data) {
  hideAll();
  renderDecision(data);
  renderAnswer(data);
  renderWhy(data);
  renderPolicy(data);
  renderEvidence(data);
  renderRuntime(data);
}

function renderCompare(compareData) {
  const rows = compareData.rows || [];
  if (!rows.length) return;
  const maxLatency = Math.max(...rows.map((row) => Number(row.latency_ms || 0)), 1);
  const body = rows
    .map((row) => `
      <tr>
        <td><b>${escapeHtml(row.strategy)}</b></td>
        <td>${escapeHtml(row.action)}</td>
        <td>${escapeHtml(row.retrieval)}</td>
        <td>${fmt(row.latency_ms)} ms</td>
        <td>${escapeHtml(row.answer_preview || "No answer generated.")}<br><small>${escapeHtml((row.warnings || []).join(" "))}</small></td>
      </tr>`)
    .join("");
  const bars = rows
    .map((row) => `
      <div class="bar-row">
        <span>${escapeHtml(row.strategy)}</span>
        <div class="bar"><i style="width:${Math.max(4, Number(row.latency_ms || 0) / maxLatency * 100)}%"></i></div>
        <span>${fmt(row.latency_ms)} ms</span>
      </div>`)
    .join("");
  compareCard.classList.remove("hidden");
  compareCard.innerHTML = `
    <p class="section-kicker">Compare Strategies</p>
    <h2>Single-question strategy comparison</h2>
    <p>${escapeHtml(compareData.note)}</p>
    <table class="comparison-table">
      <thead>
        <tr><th>Strategy</th><th>Action</th><th>Retrieval</th><th>Retrieval time</th><th>Answer</th></tr>
      </thead>
      <tbody>${body}</tbody>
    </table>
    <div class="latency-bars">${bars}</div>`;
}

async function postJson(url, payload) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "Request failed.");
  return data;
}

for (const button of document.querySelectorAll(".examples button")) {
  button.addEventListener("click", () => {
    document.getElementById("question").value = button.dataset.question || "";
    if (button.dataset.domain) document.getElementById("domain").value = button.dataset.domain;
  });
}

document.getElementById("mode").addEventListener("change", updateStrategyDescription);
updateStrategyDescription();

async function runQuestion(event) {
  event.preventDefault();
  const submit = form.querySelector("button[type='submit']");
  const payload = {
    question: document.getElementById("question").value,
    domain: document.getElementById("domain").value,
    mode: document.getElementById("mode").value,
    top_k: document.getElementById("top_k").value,
    generate: document.getElementById("generate").checked,
    use_saved: document.getElementById("use-saved").checked,
  };

  try {
    submit.disabled = true;
    setLoading();
    const data = await postJson("/ask", payload);
    renderResult(data);
    if (document.getElementById("compare-toggle").checked) {
      const compareData = await postJson("/compare", payload);
      renderCompare(compareData);
    }
  } catch (error) {
    hideAll();
    decisionCard.classList.remove("hidden");
    decisionCard.innerHTML = `
      <p class="section-kicker">Error</p>
      <h2 class="error">The request could not be completed</h2>
      <p class="error">${escapeHtml(error.message)}</p>`;
  } finally {
    submit.disabled = false;
  }
}

// Support form submission and direct button activation without duplicate requests.
form.addEventListener("submit", runQuestion);
form.querySelector("button[type=submit]").addEventListener("click", runQuestion);
