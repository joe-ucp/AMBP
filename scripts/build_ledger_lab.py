from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
OUT_DIR = ROOT / "ledger_lab"
OUT_FILE = OUT_DIR / "index.html"


FILES = {
    "same": RESULTS / "near_degenerate_multi_tube_packing_attack_summary.csv",
    "renewal": RESULTS / "renewal_cascade_shape_topsource_phi_env_act_joint_hammer_r0p1_summary.csv",
    "tail": RESULTS / "tail_denominator_ladder_audit_top2_top3_rows.csv",
    "arr": RESULTS / "arr_deficit_attribution_audit_c185_final81_summary.csv",
    "coherent": RESULTS / "coherent_viscous_residual_attribution_top2_c413.csv",
}


def require_real_csvs() -> None:
    missing = [str(path.relative_to(ROOT)) for path in FILES.values() if not path.exists()]
    if missing:
        raise SystemExit(
            "Ledger Lab needs real cached JHTDB-derived artifacts before it can be built.\n"
            + "\n".join(f"- {item}" for item in missing)
        )


def fnum(value: object, digits: int = 3) -> str:
    if pd.isna(value):
        return "missing"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(number) >= 1000:
        return f"{number:,.0f}"
    return f"{number:.{digits}g}"


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve()).replace("\\", "/")


def counts_text(counter: Counter[str]) -> str:
    return ", ".join(f"{key}={value}" for key, value in sorted(counter.items()))


COHERENT_STATUS_SHORT = {
    "renewal_deactivation_associated_residual": "renewal/deactivation",
    "sign_balanced_oscillation": "sign-balanced oscillation",
    "signed_damping": "signed damping",
}


def status_label(key: str) -> str:
    return COHERENT_STATUS_SHORT.get(key, key)


def counts_short(counter: Counter[str]) -> str:
    return ", ".join(f"{status_label(key)}={value}" for key, value in sorted(counter.items()))


def counts_lines(counter: Counter[str]) -> list[str]:
    return [f"{status_label(key)}: {value}" for key, value in sorted(counter.items())]


def evidence(label: str, value: object) -> dict[str, str]:
    return {"label": label, "value": fnum(value)}


def build_data() -> dict[str, object]:
    require_real_csvs()

    same = pd.read_csv(FILES["same"])
    renewal = pd.read_csv(FILES["renewal"])
    tail = pd.read_csv(FILES["tail"])
    arr = pd.read_csv(FILES["arr"])
    coherent = pd.read_csv(FILES["coherent"])

    same_candidates = same[
        (same["natural_tubes"].round() == 28)
        & (same["pre_quotient_tubes"].round() == 224)
        & (same["post_quotient_tubes"].round() == 28)
    ]
    same_row = same_candidates.iloc[0] if not same_candidates.empty else same.sort_values("pre_quotient_tubes").iloc[-1]

    renewal_stressed = renewal[renewal["M"].astype(float) > 1].copy()
    renewal_ref = renewal_stressed if not renewal_stressed.empty else renewal
    renewal_row = renewal_ref.sort_values("Fphys_star_available").iloc[0]

    tail_counts = Counter(str(x) for x in tail["best_denominator"].dropna())
    unresolved_d3 = int(tail["ratio_D3"].isna().sum()) if "ratio_D3" in tail else 0
    unresolved_d6 = int(tail["ratio_D6"].isna().sum()) if "ratio_D6" in tail else 0
    tail_worst = tail.sort_values("best_ratio").iloc[-1]

    arr_row = arr.iloc[0]
    coherent_counts = Counter(str(x) for x in coherent["door4_status"].dropna())
    coherent_worst = coherent.sort_values("coherent_residual_ratio").iloc[-1]

    channel_names = [
        "Source",
        "Boundary",
        "Tail",
        "Renewal",
        "Quotient",
        "Separation",
        "Strain",
        "Coherent residual",
        "Unresolved",
    ]

    def channel_map(entries: dict[str, dict[str, str]]) -> list[dict[str, str]]:
        rows = []
        for name in channel_names:
            base = {"name": name, "state": "idle", "value": ""}
            base.update(entries.get(name, {}))
            rows.append(base)
        return rows

    attacks = [
        {
            "id": "same-parent",
            "title": "Same-parent split",
            "type": "split",
            "caption": "Multiplicity alone is not amplification.",
            "sentence": "This attack does not create unpaid amplification; quotienting collapses painted labels back to the original physical count.",
            "counts": {
                "natural": int(round(float(same_row["natural_tubes"]))),
                "labeled": int(round(float(same_row["pre_quotient_tubes"]))),
                "post": int(round(float(same_row["post_quotient_tubes"]))),
                "eta": fnum(same_row["eta_post"]),
            },
            "channels": channel_map(
                {
                    "Source": {"state": "charged", "value": f"eta={fnum(same_row['eta_post'])}"},
                    "Quotient": {"state": "collapsed", "value": f"{fnum(same_row['quotient_merged_fraction'])} merged"},
                    "Unresolved": {"state": "idle", "value": "none for this attack"},
                }
            ),
            "evidence": [
                evidence("case", same_row["case"]),
                evidence("candidate", same_row["candidate_id"]),
                evidence("natural tubes", same_row["natural_tubes"]),
                evidence("pre-quotient tubes", same_row["pre_quotient_tubes"]),
                evidence("post-quotient tubes", same_row["post_quotient_tubes"]),
                evidence("eta post", same_row["eta_post"]),
                evidence("attack status", same_row["attack_status"]),
            ],
            "current": "All same-parent split rows merge back under the physical quotient.",
            "break": "Find a same-parent split with sustained post-quotient multiplicity and nonzero residual packing.",
            "unresolved": "No new unresolved row is introduced by label multiplicity in this cached audit.",
        },
        {
            "id": "renewal",
            "title": "Renewal cascade jitter",
            "type": "jitter",
            "caption": "If it does not quotient, it pays.",
            "sentence": "Cross-parent jitter keeps capture high, but the available physical charge remains positive instead of disappearing.",
            "counts": {
                "etaMin": fnum(renewal_ref["eta_post"].min()),
                "fphysMin": fnum(renewal_ref["Fphys_star_available"].min()),
                "growthMax": fnum(renewal_ref["post_growth_fraction"].max()),
                "c0thetaMax": fnum(renewal_ref["C0theta_proxy"].max()),
            },
            "channels": channel_map(
                {
                    "Source": {"state": "charged", "value": f"eta min {fnum(renewal_ref['eta_post'].min())}"},
                    "Renewal": {"state": "paid", "value": "cascade exposed"},
                    "Separation": {"state": "paid", "value": f"growth {fnum(renewal_ref['post_growth_fraction'].max())}"},
                    "Strain": {"state": "charged", "value": f"C0theta {fnum(renewal_ref['C0theta_proxy'].max())}"},
                    "Unresolved": {"state": "idle", "value": "no free jitter row"},
                }
            ),
            "evidence": [
                evidence("rows", len(renewal)),
                evidence("stressed rows", len(renewal_stressed)),
                evidence("min eta post", renewal_ref["eta_post"].min()),
                evidence("min Fphys", renewal_ref["Fphys_star_available"].min()),
                evidence("max post growth", renewal_ref["post_growth_fraction"].max()),
                evidence("max C0theta proxy", renewal_ref["C0theta_proxy"].max()),
                evidence("sample status", renewal_row["attack_status"]),
            ],
            "current": "High capture persists, but final physical charge remains visible in the cached stressed rows.",
            "break": "Produce high eta with vanishing final physical charge and no quotient redundancy.",
            "unresolved": "This remains a mechanism audit, not a continuum proof of every forward-routing constant.",
        },
        {
            "id": "tail",
            "title": "Tail denominator ladder",
            "type": "tail",
            "caption": "Paid rows and unresolved rows stay different.",
            "sentence": "Annular tail pressure routes to the best available D5 denominator while D3 and D6 remain explicitly unresolved.",
            "counts": {
                "rows": len(tail),
                "profile": counts_text(tail_counts),
                "bestMin": fnum(tail["best_ratio"].min()),
                "bestMax": fnum(tail["best_ratio"].max()),
                "unresolved": unresolved_d3 + unresolved_d6,
            },
            "channels": channel_map(
                {
                    "Source": {"state": "charged", "value": f"{len(tail)} rows"},
                    "Tail": {"state": "paid", "value": counts_text(tail_counts)},
                    "Unresolved": {"state": "unresolved", "value": f"D3={unresolved_d3}, D6={unresolved_d6}"},
                }
            ),
            "evidence": [
                evidence("rows", len(tail)),
                evidence("best denominator profile", counts_text(tail_counts)),
                evidence("min best ratio", tail["best_ratio"].min()),
                evidence("median best ratio", tail["best_ratio"].median()),
                evidence("max best ratio", tail["best_ratio"].max()),
                evidence("worst candidate", tail_worst["candidate_id"]),
            ],
            "current": "D5 is the best available denominator in the cached rows; D3 and D6 stay named rather than filled by proxy.",
            "break": "Find annular tail mass with no finite-overlap parent payment and no named denominator or unresolved row.",
            "unresolved": f"D3 is missing in {unresolved_d3} rows; D6 is missing in {unresolved_d6} rows.",
        },
        {
            "id": "arr",
            "title": "ARR correction",
            "type": "gauge",
            "caption": "A named repair moves the gauge.",
            "sentence": "Renewal exposure changes the ARR ratio from above one to below one in the cached c185 final81 row.",
            "counts": {
                "before": float(arr_row["ratio_current"]),
                "after": float(arr_row["ratio_with_renewal_exposure"]),
                "support": int(arr_row["support"]),
                "renewalFraction": fnum(arr_row["renewal_fraction"]),
            },
            "channels": channel_map(
                {
                    "Tail": {"state": "charged", "value": "ARR deficit"},
                    "Renewal": {"state": "paid", "value": "exposure correction"},
                    "Unresolved": {"state": "idle", "value": "none in this row"},
                }
            ),
            "evidence": [
                evidence("candidate", arr_row["candidate_id"]),
                evidence("support", arr_row["support"]),
                evidence("ratio current", arr_row["ratio_current"]),
                evidence("ratio with renewal", arr_row["ratio_with_renewal_exposure"]),
                evidence("renewal fraction", arr_row["renewal_fraction"]),
                evidence("renewal transitions", arr_row["renewal_transition_count"]),
            ],
            "current": "The ARR ratio moves from 1.022 to 0.967 after the renewal-exposure row is named.",
            "break": "Find a persistent ARR deficit that is not renewal, tail, boundary, coherent residual, or physical charge.",
            "unresolved": "The row is a finite-radius attribution result; it does not by itself prove the full ARR theorem.",
        },
        {
            "id": "coherent",
            "title": "Coherent residual",
            "type": "residual",
            "caption": "Residual energy does not vanish by magic.",
            "sentence": "Positive coherent residual rows route into signed damping, renewal/deactivation, or sign-balanced oscillation labels.",
            "counts": {
                "rows": len(coherent),
                "statuses": counts_short(coherent_counts),
                "statusLines": counts_lines(coherent_counts),
                "maxRatio": fnum(coherent["coherent_residual_ratio"].max()),
                "maxQ": fnum(coherent["Q"].max()),
            },
            "channels": channel_map(
                {
                    "Renewal": {"state": "paid", "value": "deactivation row"},
                    "Coherent residual": {"state": "charged", "value": counts_short(coherent_counts)},
                    "Unresolved": {"state": "idle", "value": "no unlabeled positive row"},
                }
            ),
            "evidence": [
                evidence("rows", len(coherent)),
                evidence("status profile", counts_text(coherent_counts)),
                evidence("max residual ratio", coherent["coherent_residual_ratio"].max()),
                evidence("max Q", coherent["Q"].max()),
                evidence("worst status", coherent_worst["door4_status"]),
                evidence("worst radius dx", coherent_worst["radius_dx"]),
            ],
            "current": "The observed positive residual rows keep their labels: sign balance, renewal/deactivation, or signed damping.",
            "break": "Find a positive coherent residual that is not sign-balanced, damping-paid, renewal-paid, or explicitly matrix-paid.",
            "unresolved": "The coherent row still needs theorem-level matrix routing constants.",
        },
    ]

    return {
        "builtFrom": {key: str(path.relative_to(ROOT)).replace("\\", "/") for key, path in FILES.items()},
        "attacks": attacks,
    }


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>The Ledger Lab</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #101215;
      --panel: #181c20;
      --panel-2: #20262b;
      --ink: #f3f5ed;
      --muted: #aeb8b4;
      --gold: #f1bd4b;
      --teal: #3db9a6;
      --red: #e06666;
      --green: #7ec36a;
      --blue: #74a7ff;
      --line: #313b40;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background:
        radial-gradient(circle at 12% 8%, rgba(61,185,166,0.14), transparent 26rem),
        linear-gradient(135deg, #101215 0%, #14181b 55%, #0d1114 100%);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }
    header {
      padding: 28px 32px 18px;
      border-bottom: 1px solid var(--line);
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 18px;
      align-items: end;
    }
    h1 {
      margin: 0 0 8px;
      font-size: 34px;
      line-height: 1.05;
      letter-spacing: 0;
    }
    .subtitle {
      margin: 0;
      max-width: 850px;
      color: var(--muted);
      line-height: 1.45;
      font-size: 15px;
    }
    .badge {
      border: 1px solid rgba(126,195,106,0.45);
      color: #dff5d6;
      background: rgba(126,195,106,0.12);
      padding: 9px 12px;
      font-size: 12px;
      text-transform: uppercase;
      white-space: nowrap;
    }
    main {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 360px;
      gap: 18px;
      padding: 18px;
      max-width: 1500px;
      margin: 0 auto;
    }
    .toolbar {
      grid-column: 1 / -1;
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
      min-width: 0;
    }
    .attack-button {
      appearance: none;
      border: 1px solid var(--line);
      color: var(--ink);
      background: #171b1f;
      padding: 10px 12px;
      cursor: pointer;
      font: inherit;
      font-size: 14px;
    }
    .attack-button[aria-pressed="true"] {
      border-color: rgba(241,189,75,0.8);
      background: rgba(241,189,75,0.16);
      color: #ffe4a0;
    }
    .stage-panel, .side-panel, .lower-panel {
      background: rgba(24, 28, 32, 0.94);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: 0 18px 50px rgba(0,0,0,0.25);
      min-width: 0;
    }
    .stage-panel {
      min-height: 600px;
      overflow: hidden;
      display: grid;
      grid-template-rows: auto minmax(420px, 1fr) auto;
    }
    .stage-head {
      padding: 18px 20px 10px;
      display: flex;
      justify-content: space-between;
      gap: 18px;
      align-items: start;
    }
    .stage-title {
      margin: 0;
      font-size: 22px;
      letter-spacing: 0;
    }
    .caption {
      margin: 6px 0 0;
      color: #ffe0a0;
      font-weight: 700;
    }
    .count-strip {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      justify-content: flex-end;
      max-width: 460px;
    }
    .metric-pill {
      border: 1px solid rgba(255,255,255,0.08);
      background: rgba(255,255,255,0.04);
      padding: 8px 10px;
      min-width: 98px;
    }
    .metric-pill span {
      display: block;
      color: var(--muted);
      font-size: 11px;
      text-transform: uppercase;
      margin-bottom: 2px;
    }
    .metric-pill strong {
      font-size: 18px;
      color: var(--ink);
      overflow-wrap: anywhere;
      line-height: 1.25;
      display: block;
    }
    .metric-pill--stacked {
      min-width: 132px;
    }
    .metric-pill--stacked strong {
      font-size: 13px;
      font-weight: 600;
    }
    .metric-pill--stacked strong + strong {
      margin-top: 3px;
    }
    canvas {
      width: 100%;
      max-width: 100%;
      height: 100%;
      display: block;
      background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(0,0,0,0.08));
    }
    .sentence {
      border-top: 1px solid var(--line);
      padding: 14px 20px;
      color: #f8e8bd;
      font-size: 16px;
      line-height: 1.4;
    }
    .side-panel {
      padding: 16px;
      display: grid;
      gap: 16px;
      align-content: start;
    }
    .panel-title {
      margin: 0 0 10px;
      font-size: 15px;
      text-transform: uppercase;
      color: var(--muted);
    }
    .ledger {
      display: grid;
      gap: 7px;
    }
    .ledger-row {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 8px;
      padding: 9px 10px;
      border: 1px solid var(--line);
      background: #15191d;
      border-left-width: 4px;
      overflow-wrap: anywhere;
      min-width: 0;
    }
    .ledger-row.paid { border-left-color: var(--green); }
    .ledger-row.charged { border-left-color: var(--blue); }
    .ledger-row.collapsed { border-left-color: var(--gold); }
    .ledger-row.unresolved { border-left-color: var(--red); }
    .ledger-row.idle { border-left-color: #555f66; opacity: 0.68; }
    .ledger-name { font-weight: 700; }
    .ledger-state {
      text-transform: uppercase;
      font-size: 11px;
      color: var(--muted);
      align-self: start;
    }
    .ledger-value {
      grid-column: 1 / -1;
      color: var(--muted);
      font-size: 12px;
      min-height: 16px;
      overflow-wrap: anywhere;
    }
    .lower-grid {
      grid-column: 1 / -1;
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(320px, 0.85fr);
      gap: 18px;
    }
    .lower-panel {
      padding: 16px;
    }
    .evidence-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
      gap: 10px;
    }
    .evidence-item {
      border: 1px solid var(--line);
      background: #14181c;
      padding: 10px;
    }
    .evidence-label {
      color: var(--muted);
      font-size: 11px;
      text-transform: uppercase;
      margin-bottom: 4px;
    }
    .evidence-value {
      font-size: 15px;
      overflow-wrap: anywhere;
    }
    .break-list {
      display: grid;
      gap: 10px;
    }
    .break-item {
      border-left: 3px solid var(--teal);
      background: #14181c;
      padding: 10px 12px;
      line-height: 1.42;
      overflow-wrap: anywhere;
      min-width: 0;
    }
    .break-item strong {
      display: block;
      margin-bottom: 4px;
      color: #d8fff6;
    }
    footer {
      color: var(--muted);
      font-size: 12px;
      padding: 0 18px 18px;
      max-width: 1500px;
      margin: 0 auto;
    }
    @media (max-width: 980px) {
      header { grid-template-columns: 1fr; }
      main { grid-template-columns: 1fr; }
      .lower-grid { grid-template-columns: 1fr; }
      .stage-head { display: block; }
      .count-strip { justify-content: flex-start; margin-top: 12px; }
    }
    @media (max-width: 720px) {
      header { padding: 24px 14px 16px; }
      main { padding: 10px; }
      h1 { font-size: 30px; }
      .attack-button { flex: 1 1 170px; }
      .stage-panel { min-height: 560px; }
      .metric-pill { flex: 1 1 42%; min-width: 0; }
      .stage-head { padding: 16px 14px 8px; }
      .sentence { padding: 13px 14px; font-size: 15px; }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>The Ledger Lab</h1>
      <p class="subtitle">Watching amplification try to escape. Gold packets move through real cached JHTDB-derived audit rows and light the ledger channels they are paid into, charged into, quotient-collapsed by, or still leave unresolved.</p>
    </div>
    <div class="badge">Real cached evidence only</div>
  </header>
  <main>
    <nav class="toolbar" id="attackButtons" aria-label="Attack selector"></nav>
    <section class="stage-panel" aria-label="Animated Ledger Flow">
      <div class="stage-head">
        <div>
          <h2 class="stage-title" id="attackTitle"></h2>
          <p class="caption" id="attackCaption"></p>
        </div>
        <div class="count-strip" id="metricStrip"></div>
      </div>
      <canvas id="ledgerCanvas" width="980" height="520"></canvas>
      <div class="sentence" id="attackSentence"></div>
    </section>
    <aside class="side-panel" aria-label="Payment Ledger">
      <section>
        <h3 class="panel-title">Payment Ledger</h3>
        <div class="ledger" id="ledgerRows"></div>
      </section>
      <section>
        <h3 class="panel-title">Data Sources</h3>
        <div class="break-list" id="sourceRows"></div>
      </section>
    </aside>
    <section class="lower-grid">
      <div class="lower-panel">
        <h3 class="panel-title">Evidence Panel</h3>
        <div class="evidence-grid" id="evidencePanel"></div>
      </div>
      <div class="lower-panel">
        <h3 class="panel-title">Break It Panel</h3>
        <div class="break-list" id="breakPanel"></div>
      </div>
    </section>
  </main>
  <footer>No synthetic rows are used by this visual. Missing real cached artifacts cause the builder to stop before generating the page.</footer>
  <script>
    const MODEL = __DATA__;
    const canvas = document.getElementById("ledgerCanvas");
    const ctx = canvas.getContext("2d");
    let active = MODEL.attacks[0];
    let start = performance.now();

    const channels = [
      ["Source", 760, 82],
      ["Boundary", 820, 132],
      ["Tail", 775, 188],
      ["Renewal", 842, 242],
      ["Quotient", 786, 300],
      ["Separation", 855, 354],
      ["Strain", 785, 410],
      ["Coherent residual", 845, 462],
      ["Unresolved", 700, 462],
    ];

    function attackById(id) {
      return MODEL.attacks.find((item) => item.id === id) || MODEL.attacks[0];
    }

    function setAttack(id) {
      active = attackById(id);
      start = performance.now();
      renderPanels();
    }

    function renderButtons() {
      const host = document.getElementById("attackButtons");
      host.innerHTML = "";
      MODEL.attacks.forEach((attack) => {
        const button = document.createElement("button");
        button.className = "attack-button";
        button.type = "button";
        button.textContent = attack.title;
        button.setAttribute("aria-pressed", attack.id === active.id ? "true" : "false");
        button.addEventListener("click", () => setAttack(attack.id));
        host.appendChild(button);
      });
    }

    function renderPanels() {
      renderButtons();
      document.getElementById("attackTitle").textContent = "Attack selected: " + active.title;
      document.getElementById("attackCaption").textContent = active.caption;
      document.getElementById("attackSentence").textContent = active.sentence;

      const metricStrip = document.getElementById("metricStrip");
      metricStrip.innerHTML = "";
      Object.entries(active.counts).forEach(([key, value]) => {
        if (key === "statusLines" || Array.isArray(value)) return;
        const item = document.createElement("div");
        if (key === "statuses" && Array.isArray(active.counts.statusLines)) {
          item.className = "metric-pill metric-pill--stacked";
          item.innerHTML = `<span>${labelize(key)}</span>` + active.counts.statusLines.map((line) => `<strong>${line}</strong>`).join("");
        } else {
          item.className = "metric-pill";
          item.innerHTML = `<span>${labelize(key)}</span><strong>${formatMetric(value)}</strong>`;
        }
        metricStrip.appendChild(item);
      });

      const ledger = document.getElementById("ledgerRows");
      ledger.innerHTML = "";
      active.channels.forEach((row) => {
        const item = document.createElement("div");
        item.className = `ledger-row ${row.state}`;
        item.innerHTML = `<div class="ledger-name">${row.name}</div><div class="ledger-state">${row.state}</div><div class="ledger-value">${row.value || "&nbsp;"}</div>`;
        ledger.appendChild(item);
      });

      const evidence = document.getElementById("evidencePanel");
      evidence.innerHTML = "";
      active.evidence.forEach((row) => {
        const item = document.createElement("div");
        item.className = "evidence-item";
        item.innerHTML = `<div class="evidence-label">${row.label}</div><div class="evidence-value">${row.value}</div>`;
        evidence.appendChild(item);
      });

      const breakPanel = document.getElementById("breakPanel");
      breakPanel.innerHTML = "";
      [
        ["Current read", active.current],
        ["How to break it", active.break],
        ["Unresolved remainder", active.unresolved],
      ].forEach(([label, text]) => {
        const item = document.createElement("div");
        item.className = "break-item";
        item.innerHTML = `<strong>${label}</strong>${text}`;
        breakPanel.appendChild(item);
      });

      const sourceRows = document.getElementById("sourceRows");
      sourceRows.innerHTML = "";
      Object.entries(MODEL.builtFrom).forEach(([label, path]) => {
        const item = document.createElement("div");
        item.className = "break-item";
        item.innerHTML = `<strong>${label}</strong>${path}`;
        sourceRows.appendChild(item);
      });
    }

    function labelize(key) {
      return key.replace(/([A-Z])/g, " $1").replace(/^./, (c) => c.toUpperCase());
    }

    function formatMetric(value) {
      if (typeof value === "number") {
        if (Math.abs(value) >= 1000) return value.toLocaleString(undefined, { maximumFractionDigits: 0 });
        return value.toLocaleString(undefined, { maximumSignificantDigits: 4 });
      }
      return String(value);
    }

    function fit() {
      const rect = canvas.getBoundingClientRect();
      const ratio = window.devicePixelRatio || 1;
      canvas.width = Math.max(1, Math.floor(rect.width * ratio));
      canvas.height = Math.max(1, Math.floor(rect.height * ratio));
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    }

    window.addEventListener("resize", fit);
    fit();
    renderPanels();
    requestAnimationFrame(draw);

    function draw(now) {
      const w = canvas.clientWidth;
      const h = canvas.clientHeight;
      const t = ((now - start) / 1000) % 8;
      ctx.clearRect(0, 0, w, h);
      drawGrid(w, h);
      drawBase(w, h);
      if (active.type === "split") drawSplit(w, h, t);
      if (active.type === "jitter") drawJitter(w, h, t);
      if (active.type === "tail") drawTail(w, h, t);
      if (active.type === "gauge") drawGauge(w, h, t);
      if (active.type === "residual") drawResidual(w, h, t);
      requestAnimationFrame(draw);
    }

    function drawGrid(w, h) {
      ctx.save();
      ctx.strokeStyle = "rgba(255,255,255,0.035)";
      ctx.lineWidth = 1;
      for (let x = 0; x < w; x += 36) {
        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
      }
      for (let y = 0; y < h; y += 36) {
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
      }
      ctx.restore();
    }

    function drawBase(w, h) {
      const sx = 90, sy = h * 0.5;
      const cx = w * 0.43, cy = h * 0.5;
      drawNode(sx, sy, 80, "Source", "#3db9a6");
      drawNode(cx, cy, 118, "Scale-critical amplification", "#f1bd4b");
      drawArrow(sx + 80, sy, cx - 118, cy, "rgba(241,189,75,0.45)");
      channels.forEach(([name, px, py]) => {
        const x = Math.min(w - 120, px / 980 * w);
        const y = py / 520 * h;
        const row = active.channels.find((item) => item.name === name);
        const color = stateColor(row ? row.state : "idle");
        drawNode(x, y, name.length > 12 ? 82 : 64, name, color, row && row.state !== "idle");
        if (row && row.state !== "idle") {
          drawArrow(cx + 118, cy, x - 64, y, color + "88");
        }
      });
    }

    function drawSplit(w, h, t) {
      const cx = w * 0.43, cy = h * 0.5;
      const qx = Math.min(w - 170, 786 / 980 * w), qy = 300 / 520 * h;
      const phase = Math.floor(t / 2.6);
      const natural = active.counts.natural;
      const labeled = active.counts.labeled;
      const post = active.counts.post;
      const shown = phase === 0 ? 8 : phase === 1 ? 28 : 8;
      const label = phase === 0 ? `${natural} natural tubes` : phase === 1 ? `${labeled} painted labels` : `${post} post-quotient tubes`;
      for (let i = 0; i < shown; i++) {
        const a = (i / shown) * Math.PI * 2 + t;
        const radius = phase === 1 ? 108 + 18 * Math.sin(t * 2 + i) : 54;
        const x = cx + Math.cos(a) * radius;
        const y = cy + Math.sin(a) * radius * 0.58;
        drawPacket(x, y, phase === 1 ? 5 : 8, phase === 1 ? "#f1bd4b" : "#ffd980");
      }
      drawArrow(cx + 95, cy, qx - 70, qy, "rgba(241,189,75,0.8)");
      drawText(label, cx, cy - 148, 18, "#ffe4a0", "center");
      drawText(`${natural} -> ${labeled} -> ${post}`, cx, cy + 146, 28, "#f3f5ed", "center");
      if (phase === 2) drawText("Quotient gate strips label paint", qx, qy + 86, 16, "#ffe4a0", "center");
      movingPackets([[90, h * 0.5], [cx, cy], [qx, qy]], t, 9);
    }

    function drawJitter(w, h, t) {
      const cx = w * 0.43, cy = h * 0.5;
      drawText(`eta min ${active.counts.etaMin}; min Fphys ${active.counts.fphysMin}`, cx, cy - 148, 18, "#ffe4a0", "center");
      const targets = ["Renewal", "Separation", "Strain", "Source"];
      targets.forEach((name, idx) => {
        const [_, px, py] = channels.find((item) => item[0] === name);
        const x = Math.min(w - 120, px / 980 * w);
        const y = py / 520 * h;
        movingPackets([[90, h * 0.5], [cx + Math.sin(t + idx) * 28, cy + Math.cos(t * 1.3 + idx) * 20], [x, y]], t + idx * 0.8, 5);
      });
      for (let i = 0; i < 18; i++) {
        const x = cx + Math.sin(t * 2 + i) * 140 + Math.cos(i) * 18;
        const y = cy + Math.cos(t * 2.2 + i * 0.7) * 80;
        drawPacket(x, y, 4, "#74a7ff");
      }
    }

    function drawTail(w, h, t) {
      const cx = w * 0.43, cy = h * 0.5;
      ctx.save();
      ctx.strokeStyle = "rgba(116,167,255,0.28)";
      ctx.lineWidth = 2;
      for (let r = 70; r <= 170; r += 35) {
        ctx.beginPath();
        ctx.ellipse(cx, cy, r * 1.45, r * 0.78, 0, 0, Math.PI * 2);
        ctx.stroke();
      }
      ctx.restore();
      drawText(`Best denominator: ${active.counts.profile}`, cx, cy - 155, 18, "#ffe4a0", "center");
      drawText(`D3/D6 unresolved rows: ${active.counts.unresolved}`, cx, cy + 158, 18, "#ffb5a8", "center");
      const tailTarget = channels.find((item) => item[0] === "Tail");
      const unresolvedTarget = channels.find((item) => item[0] === "Unresolved");
      [tailTarget, unresolvedTarget].forEach((target, idx) => {
        const x = Math.min(w - 120, target[1] / 980 * w);
        const y = target[2] / 520 * h;
        movingPackets([[90, h * 0.5], [cx, cy], [x, y]], t + idx * 1.6, idx ? 4 : 8);
      });
    }

    function drawGauge(w, h, t) {
      const cx = w * 0.43, cy = h * 0.5;
      const before = active.counts.before;
      const after = active.counts.after;
      const p = Math.min(1, (t % 5) / 3.5);
      const eased = p * p * (3 - 2 * p);
      const value = before + (after - before) * eased;
      drawText("ARR ratio", cx, cy - 150, 18, "#ffe4a0", "center");
      drawText(value.toFixed(3), cx, cy + 12, 76, value < 1 ? "#7ec36a" : "#e06666", "center");
      ctx.save();
      ctx.strokeStyle = "rgba(255,255,255,0.22)";
      ctx.lineWidth = 18;
      ctx.beginPath();
      ctx.arc(cx, cy + 12, 125, Math.PI * 0.85, Math.PI * 2.15);
      ctx.stroke();
      ctx.strokeStyle = value < 1 ? "#7ec36a" : "#e06666";
      ctx.beginPath();
      ctx.arc(cx, cy + 12, 125, Math.PI * 0.85, Math.PI * (0.85 + 1.3 * eased));
      ctx.stroke();
      ctx.restore();
      const renewal = channels.find((item) => item[0] === "Renewal");
      movingPackets([[90, h * 0.5], [cx, cy], [Math.min(w - 120, renewal[1] / 980 * w), renewal[2] / 520 * h]], t, 7);
      drawText(`${before.toFixed(3)} -> ${after.toFixed(3)} after renewal exposure`, cx, cy + 172, 18, "#f3f5ed", "center");
    }

    function drawResidual(w, h, t) {
      const cx = w * 0.43, cy = h * 0.5;
      const lines = Array.isArray(active.counts.statusLines)
        ? active.counts.statusLines
        : String(active.counts.statuses || "").split(/,\s*/).filter(Boolean);
      const titleY = cy - 168 - Math.max(0, lines.length - 1) * 8;
      drawText("Residual status profile", cx, titleY, 16, "#ffe4a0", "center");
      lines.forEach((line, i) => {
        drawText(line, cx, titleY + 26 + i * 18, 13, "#ffe4a0", "center");
      });
      for (let i = 0; i < 26; i++) {
        const r = 34 + (i % 5) * 16 + Math.sin(t * 2 + i) * 5;
        const a = t * 0.9 + i * 0.68;
        drawPacket(cx + Math.cos(a) * r * 1.5, cy + Math.sin(a) * r, 4 + (i % 3), i % 2 ? "#74a7ff" : "#3db9a6");
      }
      ["Coherent residual", "Renewal"].forEach((name, idx) => {
        const target = channels.find((item) => item[0] === name);
        movingPackets([[90, h * 0.5], [cx, cy], [Math.min(w - 120, target[1] / 980 * w), target[2] / 520 * h]], t + idx * 1.2, 5);
      });
    }

    function movingPackets(points, t, n) {
      for (let i = 0; i < n; i++) {
        const p = ((t * 0.22 + i / n) % 1);
        const pos = bezier(points[0], points[1], points[2], p);
        drawPacket(pos[0], pos[1], 5 + (i % 3), "#f1bd4b");
      }
    }

    function bezier(a, b, c, p) {
      const x = (1 - p) * (1 - p) * a[0] + 2 * (1 - p) * p * b[0] + p * p * c[0];
      const y = (1 - p) * (1 - p) * a[1] + 2 * (1 - p) * p * b[1] + p * p * c[1];
      return [x, y];
    }

    function drawNode(x, y, r, label, color, activeNode = true) {
      ctx.save();
      ctx.globalAlpha = activeNode ? 1 : 0.55;
      ctx.fillStyle = colorToFill(color);
      ctx.strokeStyle = color;
      ctx.lineWidth = activeNode ? 2 : 1;
      ctx.beginPath();
      ctx.roundRect(x - r, y - 22, r * 2, 44, 8);
      ctx.fill();
      ctx.stroke();
      drawText(label, x, y + 5, label.length > 18 ? 11 : 12, "#f3f5ed", "center");
      ctx.restore();
    }

    function drawArrow(x1, y1, x2, y2, color) {
      ctx.save();
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      const mid = (x1 + x2) / 2;
      ctx.bezierCurveTo(mid, y1, mid, y2, x2, y2);
      ctx.stroke();
      ctx.restore();
    }

    function drawPacket(x, y, r, color) {
      ctx.save();
      ctx.fillStyle = color;
      ctx.shadowColor = color;
      ctx.shadowBlur = 14;
      ctx.beginPath();
      ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    }

    function drawText(text, x, y, size, color, align) {
      ctx.save();
      ctx.fillStyle = color;
      ctx.font = `700 ${size}px Inter, system-ui, sans-serif`;
      ctx.textAlign = align || "left";
      ctx.textBaseline = "middle";
      ctx.fillText(text, x, y);
      ctx.restore();
    }

    function stateColor(state) {
      if (state === "paid") return "#7ec36a";
      if (state === "charged") return "#74a7ff";
      if (state === "collapsed") return "#f1bd4b";
      if (state === "unresolved") return "#e06666";
      return "#606970";
    }

    function colorToFill(color) {
      if (color === "#7ec36a") return "rgba(126,195,106,0.13)";
      if (color === "#74a7ff") return "rgba(116,167,255,0.13)";
      if (color === "#f1bd4b") return "rgba(241,189,75,0.15)";
      if (color === "#e06666") return "rgba(224,102,102,0.13)";
      if (color === "#3db9a6") return "rgba(61,185,166,0.13)";
      return "rgba(255,255,255,0.05)";
    }
  </script>
</body>
</html>
"""


def main() -> int:
    data = build_data()
    html = HTML_TEMPLATE.replace("__DATA__", json.dumps(data, ensure_ascii=True))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(html, encoding="utf-8")
    print(f"wrote {rel(OUT_FILE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
