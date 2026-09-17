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

async function loadBenchmarks() {
  const chart = document.querySelector("#benchmark-bars");
  if (!chart) return;
  try {
    const response = await fetch("data/benchmarks.json");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    let metric = "write";
    const research = document.querySelector("#show-research");
    const metrics = data.bank.metrics;
    const format = new Intl.NumberFormat("en-US");
    function render() {
      const rows = data.bank.engines.filter(
        (e) => research.checked || e.status !== "research",
      );
      const spec = metrics[metric];
      const baseline = data.bank.engines.find((e) => e.id === "mdbx").values[
        metric
      ];
      const largest = Math.max(...rows.map((e) => e.values[metric]));
      document.querySelector("#metric-description").textContent =
        `${spec.label} · ${spec.direction === "higher" ? "higher" : "lower"} is better`;
      chart.replaceChildren();
      rows.forEach((engine) => {
        const value = engine.values[metric];
        const ratio = value / baseline;
        const worse = spec.direction === "lower" ? ratio > 1 : ratio < 1;
        const row = document.createElement("div");
        row.className = `bench-row ${engine.id === "mdbx" ? "baseline" : engine.status}`;
        const label = document.createElement("div");
        label.className = "bench-label";
        label.textContent = engine.name;
        const detail = document.createElement("small");
        detail.textContent = engine.detail;
        label.append(detail);
        const track = document.createElement("div");
        track.className = "bar-track";
        track.setAttribute("aria-hidden", "true");
        const fill = document.createElement("div");
        fill.className = "bar-fill";
        fill.style.setProperty(
          "--bar-width",
          `${((value / largest) * 100).toFixed(2)}%`,
        );
        track.append(fill);
        const number = document.createElement("div");
        number.className = "bench-value";
        const amount = document.createElement("span");
        amount.textContent = `${format.format(value)}${metric === "disk" ? " MB" : ""}`;
        const multiple = document.createElement("b");
        multiple.className = worse ? "worse" : "";
        multiple.textContent =
          engine.id === "mdbx" ? "1.00×" : `${ratio.toFixed(2)}×`;
        multiple.title = `${ratio.toFixed(2)} times the libmdbx ${spec.short}; ${spec.direction === "higher" ? "higher" : "lower"} is better`;
        number.append(amount, multiple);
        row.append(label, track, number);
        chart.append(row);
      });
      const big = data.bank.engines.find((e) => e.id === "sal_art_ref").values[
        metric
      ];
      const ratio = (big / baseline).toFixed(2);
      document.querySelector("#benchmark-insight").textContent =
        metric === "disk"
          ? `A visible tradeoff: ${ratio}× the disk high-water for Big in this measured configuration.`
          : `${ratio}× the ${spec.short} for Big in this measured configuration.`;
      document.querySelector("#benchmark-tradeoff").textContent =
        metric === "disk"
          ? "Lower is better here. This measures on-disk high-water after the workload, including allocation and retained copy-on-write storage; it is not settled live payload."
          : "The same run recorded 542 MB disk high-water for Big versus 64 MB for MDBX. Explore every metric above.";
      all("[data-metric]").forEach((b) =>
        b.setAttribute("aria-pressed", String(b.dataset.metric === metric)),
      );
    }
    all("[data-metric]").forEach((button) =>
      button.addEventListener("click", () => {
        metric = button.dataset.metric;
        render();
      }),
    );
    research.addEventListener("change", render);
    render();
  } catch {
    // The HTML contains real values and a methodology link, even offline.
    all("[data-metric], #show-research").forEach((control) => {
      control.disabled = true;
    });
    document.querySelector("#metric-description").textContent =
      "Static write-throughput summary · interactive data unavailable";
  }
}
loadBenchmarks();
