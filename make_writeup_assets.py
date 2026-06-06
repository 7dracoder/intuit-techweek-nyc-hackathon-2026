#!/usr/bin/env python3
"""
Generate figures and tables for Deliverable D from a completed pipeline run.

Reads ``submission/assets/metrics.json`` (written by ``solution.py``) and emits,
into ``submission/assets/``:

    dag.png             - the fitted Structural Causal Model DAG (Deliverable C)
    calibration.png     - validation reliability diagram (predicted vs observed)
    coverage.png        - per-PD-bin interval coverage of the realized rate
    coverage_table.md   - the same coverage data as a markdown table
    results_summary.md   - headline metrics table for the writeup

This script is OFFLINE tooling for the writeup only. It imports nothing from the
scored pipeline beyond the metrics file and matplotlib; it never touches the
submission CSVs. Run ``python solution.py`` first, then this.

    pip install -r requirements.txt   # includes matplotlib
    python solution.py
    python make_writeup_assets.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "submission" / "assets"
METRICS_PATH = ASSETS / "metrics.json"


def load_metrics() -> dict:
    if not METRICS_PATH.exists():
        raise SystemExit(
            f"metrics file not found at {METRICS_PATH}.\n"
            "Run `python solution.py` first to generate it."
        )
    with open(METRICS_PATH) as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# DAG figure
# --------------------------------------------------------------------------- #

def _topo_layers(edges: dict) -> list[list[str]]:
    """Assign each node a layer = longest path from any root, for tidy columns."""
    nodes = set(edges) | {p for ps in edges.values() for p in ps}
    depth = {n: 0 for n in nodes}
    # Relax depths (DAG => converges).
    for _ in range(len(nodes)):
        changed = False
        for child, parents in edges.items():
            for p in parents:
                if depth[child] < depth[p] + 1:
                    depth[child] = depth[p] + 1
                    changed = True
        if not changed:
            break
    layers: dict[int, list[str]] = {}
    for n in nodes:
        layers.setdefault(depth[n], []).append(n)
    return [sorted(layers[k]) for k in sorted(layers)]


def _short(name: str) -> str:
    return name.replace("observed_", "obs_").replace("_count", "_cnt")


def render_dag(metrics: dict, out: Path) -> None:
    edges = metrics.get("scm_edges", {})
    if not edges:
        print("[assets] no scm_edges in metrics; skipping DAG.")
        return
    fitted = set(metrics.get("C", {}).get("equations", list(edges.keys())))
    layers = _topo_layers(edges)

    pos: dict[str, tuple[float, float]] = {}
    for ci, layer in enumerate(layers):
        n = len(layer)
        for ri, node in enumerate(layer):
            y = (n - 1) / 2.0 - ri
            pos[node] = (ci * 3.2, y * 1.6)

    fig, ax = plt.subplots(figsize=(13, 8))
    # Edges first (so boxes sit on top).
    for child, parents in edges.items():
        for p in parents:
            if p in pos and child in pos:
                x0, y0 = pos[p]
                x1, y1 = pos[child]
                ax.add_patch(FancyArrowPatch(
                    (x0 + 1.05, y0), (x1 - 1.05, y1),
                    arrowstyle="-|>", mutation_scale=11,
                    color="#7a7a7a", lw=0.9, alpha=0.75,
                    connectionstyle="arc3,rad=0.05",
                ))
    # Nodes.
    for node, (x, y) in pos.items():
        is_fitted = node in fitted
        is_exog = node not in edges
        face = "#cfe8ff" if is_fitted else ("#eeeeee" if is_exog else "#fff2cc")
        edge = "#1f6fb2" if is_fitted else "#999999"
        box = FancyBboxPatch(
            (x - 1.05, y - 0.42), 2.1, 0.84,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            linewidth=1.3, edgecolor=edge, facecolor=face,
        )
        ax.add_patch(box)
        ax.text(x, y, _short(node), ha="center", va="center",
                fontsize=7.0, wrap=True)

    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    ax.set_xlim(min(xs) - 1.6, max(xs) + 1.6)
    ax.set_ylim(min(ys) - 1.2, max(ys) + 1.2)
    ax.axis("off")
    ax.set_title(
        f"Structural Causal Model — {len(fitted)} fitted structural equations\n"
        "blue = fitted child node, grey = exogenous parent",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[assets] wrote {out}")


# --------------------------------------------------------------------------- #
# Calibration figure
# --------------------------------------------------------------------------- #

def render_calibration(metrics: dict, out: Path, n_bins: int = 10) -> None:
    A = metrics.get("A", {})
    pdv = np.asarray(A.get("val_pd", []), dtype=float)
    yv = np.asarray(A.get("val_y", []), dtype=float)
    if pdv.size == 0:
        print("[assets] no val_pd in metrics; skipping calibration.")
        return
    order = np.argsort(pdv)
    pdv, yv = pdv[order], yv[order]
    edges = np.linspace(0, len(pdv), n_bins + 1).astype(int)
    xs, ys, ns = [], [], []
    for i in range(n_bins):
        a, b = edges[i], edges[i + 1]
        if b > a:
            xs.append(pdv[a:b].mean())
            ys.append(yv[a:b].mean())
            ns.append(b - a)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "--", color="#999999", label="perfect calibration")
    ax.plot(xs, ys, "o-", color="#1f6fb2", label="model (decile bins)")
    ax.set_xlabel("Mean predicted PD")
    ax.set_ylabel("Observed default rate")
    auc = A.get("auc_final")
    brier = A.get("brier_final")
    sub = []
    if auc is not None:
        sub.append(f"AUC={auc:.4f}")
    if brier is not None:
        sub.append(f"Brier={brier:.4f}")
    ax.set_title("Validation reliability diagram"
                 + (f"\n{'  '.join(sub)}" if sub else ""))
    lim = max(0.5, float(max(xs + ys)) * 1.1)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.legend(loc="upper left")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[assets] wrote {out}")


# --------------------------------------------------------------------------- #
# Coverage figure + table
# --------------------------------------------------------------------------- #

def compute_coverage(metrics: dict, n_bins: int = 10):
    A = metrics.get("A", {})
    pdv = np.asarray(A.get("val_pd", []), dtype=float)
    yv = np.asarray(A.get("val_y", []), dtype=float)
    lo = np.asarray(A.get("val_lo", []), dtype=float)
    hi = np.asarray(A.get("val_hi", []), dtype=float)
    if pdv.size == 0 or lo.size != pdv.size:
        return None
    order = np.argsort(pdv)
    pdv, yv, lo, hi = pdv[order], yv[order], lo[order], hi[order]
    edges = np.linspace(0, len(pdv), n_bins + 1).astype(int)
    rows = []
    for i in range(n_bins):
        a, b = edges[i], edges[i + 1]
        if b <= a:
            continue
        obs = yv[a:b].mean()
        lo_m, hi_m = lo[a:b].mean(), hi[a:b].mean()
        rows.append({
            "bin": i + 1, "n": int(b - a),
            "pred": float(pdv[a:b].mean()), "obs": float(obs),
            "lo": float(lo_m), "hi": float(hi_m),
            "width": float(hi_m - lo_m),
            "covered": bool(lo_m - 1e-9 <= obs <= hi_m + 1e-9),
        })
    return rows


def render_coverage(metrics: dict, out_png: Path, out_md: Path, n_bins: int = 10) -> None:
    rows = compute_coverage(metrics, n_bins)
    if not rows:
        print("[assets] no interval data in metrics; skipping coverage.")
        return
    cov = np.mean([r["covered"] for r in rows])
    width = np.mean([r["width"] for r in rows])

    fig, ax = plt.subplots(figsize=(7, 5))
    x = [r["pred"] for r in rows]
    obs = [r["obs"] for r in rows]
    lo = [r["lo"] for r in rows]
    hi = [r["hi"] for r in rows]
    yerr = [np.array(obs) - np.array(lo), np.array(hi) - np.array(obs)]
    ax.errorbar(x, obs, yerr=yerr, fmt="o", color="#1f6fb2",
                ecolor="#9bbfe0", elinewidth=8, capsize=0, alpha=0.9,
                label="observed rate (band = 90% interval)")
    ax.plot([0, max(x) * 1.1], [0, max(x) * 1.1], "--", color="#999999",
            label="perfect calibration")
    ax.set_xlabel("Mean predicted PD (decile bin)")
    ax.set_ylabel("Observed default rate")
    ax.set_title(f"Per-bin interval coverage — {cov:.0%} of bins covered, "
                 f"mean width {width:.2f}")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"[assets] wrote {out_png}")

    lines = [
        "| Bin | n | Pred PD | Observed | Lo90 | Hi90 | Width | Covered |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['bin']} | {r['n']} | {r['pred']:.3f} | {r['obs']:.3f} | "
            f"{r['lo']:.3f} | {r['hi']:.3f} | {r['width']:.3f} | "
            f"{'yes' if r['covered'] else 'no'} |"
        )
    lines.append("")
    lines.append(f"**Bin-wise coverage: {cov:.0%}  ·  mean interval width: {width:.3f}**")
    out_md.write_text("\n".join(lines) + "\n")
    print(f"[assets] wrote {out_md}")


# --------------------------------------------------------------------------- #
# Results summary
# --------------------------------------------------------------------------- #

def render_summary(metrics: dict, out: Path) -> None:
    A = metrics.get("A", {})
    B = metrics.get("B", {})
    C = metrics.get("C", {})
    lines = ["# Run results summary", "", "| Metric | Value |", "|---|---|"]

    def add(label, val):
        lines.append(f"| {label} | {val} |")

    add("Features (incl. engineered)", f"{A.get('n_features','?')} ({A.get('n_engineered','?')} engineered)")
    add("Validation AUC (no IPW)", f"{A.get('auc_noipw','?'):.4f}" if A.get('auc_noipw') is not None else "?")
    if A.get("auc_ipw") is not None:
        add("Validation AUC (IPW)", f"{A['auc_ipw']:.4f}")
    add("IPW reject inference kept", A.get("ipw_kept"))
    add("Final AUC / Brier", f"{A.get('auc_final','?'):.4f} / {A.get('brier_final','?'):.4f}"
        if A.get('auc_final') is not None else "?")
    add("Mean effective recovery (LGD)", f"{A.get('mean_recovery','?'):.3f}" if A.get('mean_recovery') is not None else "?")
    add("Profit-simulated PD threshold", f"{A.get('profit_threshold','?'):.3f}" if A.get('profit_threshold') is not None else "?")
    add("Approval rate", f"{A.get('approval_rate', 0)*100:.1f}%")
    add("B person-period rows", f"{B.get('n_person_period','?'):,}" if isinstance(B.get('n_person_period'), int) else B.get('n_person_period','?'))
    add("B hazard models", B.get("n_hazard_models", "?"))
    add("C structural equations", C.get("n_equations", "?"))
    n = max(1, C.get("n_queries", 1))
    add("C directional split (up/down/flat)",
        f"{C.get('n_up',0)/n:.0%} / {C.get('n_down',0)/n:.0%} / {C.get('n_flat',0)/n:.0%}")
    out.write_text("\n".join(lines) + "\n")
    print(f"[assets] wrote {out}")


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    metrics = load_metrics()
    render_dag(metrics, ASSETS / "dag.png")
    render_calibration(metrics, ASSETS / "calibration.png")
    render_coverage(metrics, ASSETS / "coverage.png", ASSETS / "coverage_table.md")
    render_summary(metrics, ASSETS / "results_summary.md")
    print("[assets] done.")


if __name__ == "__main__":
    main()
