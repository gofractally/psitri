"use strict";
const all = (selector) => Array.from(document.querySelectorAll(selector));
const menu = document.querySelector(".menu-toggle");
menu?.addEventListener("click", () => {
  const expanded = menu.getAttribute("aria-expanded") === "true";
  menu.setAttribute("aria-expanded", String(!expanded));
  document.querySelector("#navigation").classList.toggle("is-open", !expanded);
});
all("#navigation a").forEach((a) =>
  a.addEventListener("click", () => {
    menu?.setAttribute("aria-expanded", "false");
    document.querySelector("#navigation")?.classList.remove("is-open");
  }),
);
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && menu?.getAttribute("aria-expanded") === "true") {
    menu.click();
    menu.focus();
  }
});

const treeStates = {
  snapshot: [
    "v1 · current",
    "Retain the root. Both versions share the same data.",
  ],
  write: [
    "v2 · current",
    "Copy the changed path. Readers on v1 keep their original view.",
  ],
  restore: [
    "v1 · restored",
    "Select the retained version. The old state is still there.",
  ],
};
all("[data-tree]").forEach((button) =>
  button.addEventListener("click", () => {
    const state = button.dataset.tree;
    document.querySelector(".tree-lab").dataset.state = state;
    document.querySelector("#root-version").textContent = treeStates[state][0];
    document.querySelector("#tree-caption").textContent = treeStates[state][1];
    all("[data-tree]").forEach((b) =>
      b.setAttribute("aria-pressed", String(b === button)),
    );
  }),
);

all("[data-engine]").forEach((button) =>
  button.addEventListener("click", () => {
    const code = document.querySelector("#example-code");
    code.textContent = code.textContent.replace(
      /psitri::(?:big|small)_engine/g,
      `psitri::${button.dataset.engine}_engine`,
    );
    all("[data-engine]").forEach((b) =>
      b.setAttribute("aria-pressed", String(b === button)),
    );
    document.querySelector("#copy-status").textContent = "";
  }),
);
document.querySelector("#copy-code")?.addEventListener("click", async () => {
  const status = document.querySelector("#copy-status");
  try {
    await navigator.clipboard.writeText(
      document.querySelector("#example-code").textContent,
    );
    status.textContent = "Copied";
  } catch {
    const selection = window.getSelection();
    const range = document.createRange();
    range.selectNodeContents(document.querySelector("#example-code"));
    selection.removeAllRanges();
    selection.addRange(range);
    status.textContent = "Selected — press Ctrl/Cmd+C";
  }
});

// Values and bars are generated from curated JSON at build time. No data fetch is
// needed, and the complete charts remain readable without JavaScript.
all("[data-comparison-chart]").forEach((chart) => {
  const within = (selector) => Array.from(chart.querySelectorAll(selector));
  const choices = within("[data-chart-choice]");
  const panels = within("[data-chart-panel]");
  const buttons = within("[data-chart-metric]");
  const research = chart.querySelector("[data-chart-research]");
  const experiments = choices.filter((c) => c.dataset.status === "research");
  let researchInitialized = false;
  let metric = "write";
  function render() {
    experiments.forEach((choice) => { choice.closest("label").hidden = !research.checked; });
    const selected = new Set(choices.filter((c) => c.checked &&
      (c.dataset.status !== "research" || research.checked)).map((c) => c.value));
    panels.forEach((panel) => {
      panel.hidden = selected.size === 0 || (buttons.length > 0 && panel.dataset.chartPanel !== metric);
    });
    const rows = within("[data-chart-engine]");
    rows.forEach((row) => { row.hidden = !selected.has(row.dataset.chartEngine); });
    const shown = rows.filter((row) => !row.hidden && !row.closest("[data-chart-panel]").hidden);
    const largest = Math.max(0, ...shown.map((row) => Number(row.dataset.value)));
    shown.forEach((row) => {
      row.querySelector(".bar-fill").style.setProperty("--bar-width", `${largest ? 100 * Number(row.dataset.value) / largest : 0}%`);
    });
    buttons.forEach((button) => button.setAttribute("aria-pressed", String(button.dataset.chartMetric === metric)));
    chart.querySelector("[data-chart-empty]").hidden = selected.size !== 0;
    const reference = selected.has(chart.dataset.baseline) ? "reference shown" : "reference hidden; ratios unchanged";
    chart.querySelector("[data-chart-status]").textContent = `${selected.size} engine${selected.size === 1 ? "" : "s"} selected · ${chart.dataset.baselineName} ${reference}`;
  }
  choices.forEach((choice) => choice.addEventListener("change", render));
  buttons.forEach((button) => button.addEventListener("click", () => {
    metric = button.dataset.chartMetric;
    render();
  }));
  research?.addEventListener("change", () => {
    if (research.checked && !researchInitialized) {
      experiments.forEach((c) => { c.checked = true; });
      researchInitialized = true;
    }
    render();
  });
  chart.querySelector("[data-chart-reset]").addEventListener("click", () => {
    choices.forEach((c) => { c.checked = c.dataset.default === "true"; });
    if (research) research.checked = false;
    researchInitialized = false;
    render();
  });
  chart.querySelector("[data-chart-clear]").addEventListener("click", () => {
    choices.forEach((c) => { c.checked = false; });
    render();
  });
  within("input, button").forEach((control) => { control.disabled = false; });
  render();
});

all("[data-api-comparison]").forEach((chart) => {
  const within = (selector) => Array.from(chart.querySelectorAll(selector));
  let architecture = chart.dataset.defaultArch;
  let metric = "write";
  function render() {
    const selected = new Set(within("[data-api-choice]").filter((c) => c.checked).map((c) => c.value));
    within("[data-api-platform]").forEach((p) => { p.hidden = p.dataset.apiPlatform !== architecture; });
    within("[data-api-metric-panel]").forEach((p) => { p.hidden = p.dataset.apiMetricPanel !== metric || selected.size === 0; });
    within("[data-api-pair]").forEach((p) => { p.hidden = !selected.has(p.dataset.apiPair); });
    within("[data-api-arch]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.apiArch === architecture)));
    within("[data-api-metric]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.apiMetric === metric)));
    chart.querySelector("[data-api-empty]").hidden = selected.size !== 0;
    const label = within("[data-api-arch]").find((b) => b.dataset.apiArch === architecture).textContent;
    chart.querySelector("[data-api-status]").textContent = `${label} · ${selected.size} API pair${selected.size === 1 ? "" : "s"} selected`;
  }
  within("[data-api-arch]").forEach((b) => b.addEventListener("click", () => { architecture = b.dataset.apiArch; render(); }));
  within("[data-api-metric]").forEach((b) => b.addEventListener("click", () => { metric = b.dataset.apiMetric; render(); }));
  within("[data-api-choice]").forEach((c) => c.addEventListener("change", render));
  within("input, button").forEach((c) => { c.disabled = false; });
  render();
});
