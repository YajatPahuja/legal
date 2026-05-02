"""
plot_results.py

Generates publication-quality bar charts comparing BM25, Dense, Hybrid,
and Hybrid+Reranker across all IR metrics for both AILA tasks.

Run from project root:
    python evaluation/plot_results.py

Output: evaluation/figures/  (one combined figure + two per-task figures)

Install deps first if needed:
    pip install matplotlib seaborn
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns

BASE_DIR   = Path(__file__).resolve().parent.parent
PROC_DIR   = BASE_DIR / "data" / "processed"
FIGURE_DIR = BASE_DIR / "evaluation" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# ── Load results ──────────────────────────────────────────────────────────────

def load(fname):
    return json.loads((PROC_DIR / fname).read_text())

bm25     = load("bm25_results.json")
dense    = load("dense_results.json")
hybrid   = load("hybrid_results.json")
reranked = load("reranked_results.json")

MODELS = ["BM25", "Dense\n(InLegalBERT)", "Hybrid\n(BM25+Dense)", "Hybrid\n+Reranker"]
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
METRICS = ["MAP", "NDCG@10", "MRR", "P@5", "P@10"]

def get_metrics(result, task):
    m = result[task]["metrics"]
    return [m[k] for k in METRICS]

# ── Style ─────────────────────────────────────────────────────────────────────

sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "figure.dpi":       150,
})

# ── Helper ────────────────────────────────────────────────────────────────────

def grouped_bar(ax, data_rows, metric_labels, model_labels, colors, title, ylabel="Score"):
    n_metrics = len(metric_labels)
    n_models  = len(model_labels)
    x         = np.arange(n_metrics)
    width     = 0.18
    offsets   = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * width

    bars_all = []
    for i, (row, color, label) in enumerate(zip(data_rows, colors, model_labels)):
        bars = ax.bar(x + offsets[i], row, width, color=color, alpha=0.88,
                      edgecolor="white", linewidth=0.6, label=label)
        bars_all.append(bars)
        for bar, val in zip(bars, row):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.001,
                f"{val:.4f}",
                ha="center", va="bottom", fontsize=6.5, rotation=90,
                color="#333333",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_ylim(0, max(max(r) for r in data_rows) * 1.35)
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)


# ── Figure 1: Combined (2 subplots side-by-side) ─────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("AILA 2019 Retrieval — Model Comparison", fontsize=15, fontweight="bold", y=1.01)

for ax, task_key, task_title in [
    (axes[0], "task1_cases",    "Task 1: Prior Case Retrieval"),
    (axes[1], "task2_statutes", "Task 2: Statute Retrieval"),
]:
    data_rows = [
        get_metrics(bm25,     task_key),
        get_metrics(dense,    task_key),
        get_metrics(hybrid,   task_key),
        get_metrics(reranked, task_key),
    ]
    grouped_bar(ax, data_rows, METRICS, MODELS, COLORS, task_title)

handles = [mpatches.Patch(color=c, alpha=0.88, label=m.replace("\n", " "))
           for c, m in zip(COLORS, MODELS)]
fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
           fontsize=10, bbox_to_anchor=(0.5, -0.06))

plt.tight_layout()
out = FIGURE_DIR / "all_models_comparison.png"
fig.savefig(out, bbox_inches="tight", dpi=200)
print(f"Saved → {out}")
plt.close(fig)


# ── Figure 2 & 3: Per-task standalone figures ─────────────────────────────────

for task_key, task_title, fname in [
    ("task1_cases",    "Task 1: Prior Case Retrieval\n(AILA 2019)",    "task1_cases.png"),
    ("task2_statutes", "Task 2: Statute Retrieval\n(AILA 2019)",        "task2_statutes.png"),
]:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    data_rows = [
        get_metrics(bm25,     task_key),
        get_metrics(dense,    task_key),
        get_metrics(hybrid,   task_key),
        get_metrics(reranked, task_key),
    ]
    grouped_bar(ax, data_rows, METRICS, MODELS, COLORS, task_title)
    handles = [mpatches.Patch(color=c, alpha=0.88, label=m.replace("\n", " "))
               for c, m in zip(COLORS, MODELS)]
    ax.legend(handles=handles, loc="upper right", frameon=True, fontsize=9,
              framealpha=0.9, edgecolor="#cccccc")
    plt.tight_layout()
    out = FIGURE_DIR / fname
    fig.savefig(out, bbox_inches="tight", dpi=200)
    print(f"Saved → {out}")
    plt.close(fig)


# ── Figure 4: Metric-wise improvement over BM25 baseline ─────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Relative Improvement over BM25 Baseline (%)", fontsize=14,
             fontweight="bold", y=1.01)

compare_models  = ["Dense\n(InLegalBERT)", "Hybrid\n(BM25+Dense)", "Hybrid\n+Reranker"]
compare_results = [dense, hybrid, reranked]
compare_colors  = COLORS[1:]

for ax, task_key, task_title in [
    (axes[0], "task1_cases",    "Task 1: Prior Case Retrieval"),
    (axes[1], "task2_statutes", "Task 2: Statute Retrieval"),
]:
    base = get_metrics(bm25, task_key)
    n_metrics = len(METRICS)
    n_models  = len(compare_models)
    x         = np.arange(n_metrics)
    width     = 0.22
    offsets   = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * width

    all_vals = []
    for i, (res, color, label) in enumerate(zip(compare_results, compare_colors, compare_models)):
        row = get_metrics(res, task_key)
        rel = [(r - b) / b * 100 if b > 0 else 0 for r, b in zip(row, base)]
        all_vals.extend(rel)
        bars = ax.bar(x + offsets[i], rel, width, color=color, alpha=0.88,
                      edgecolor="white", linewidth=0.6, label=label)
        for bar, val in zip(bars, rel):
            ypos = bar.get_height() + (1 if val >= 0 else -5)
            ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                    f"{val:+.1f}%", ha="center", va="bottom",
                    fontsize=6.5, rotation=90, color="#333333")

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(METRICS, fontsize=11)
    ax.set_ylabel("Relative change vs BM25 (%)", fontsize=10)
    ax.set_title(task_title, fontsize=12, fontweight="bold")
    margin = max(abs(v) for v in all_vals) * 0.35
    ax.set_ylim(min(0, min(all_vals)) - margin, max(all_vals) + margin + 10)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

handles = [mpatches.Patch(color=c, alpha=0.88, label=m.replace("\n", " "))
           for c, m in zip(compare_colors, compare_models)]
fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
           fontsize=10, bbox_to_anchor=(0.5, -0.06))
plt.tight_layout()
out = FIGURE_DIR / "relative_improvement.png"
fig.savefig(out, bbox_inches="tight", dpi=200)
print(f"Saved → {out}")
plt.close(fig)


# ── Figure 5: Summary table heatmap ──────────────────────────────────────────

import pandas as pd

for task_key, task_title, fname in [
    ("task1_cases",    "Task 1: Cases",    "heatmap_task1.png"),
    ("task2_statutes", "Task 2: Statutes", "heatmap_task2.png"),
]:
    data = {
        "BM25":            get_metrics(bm25,     task_key),
        "Dense":           get_metrics(dense,    task_key),
        "Hybrid":          get_metrics(hybrid,   task_key),
        "Hybrid+Reranker": get_metrics(reranked, task_key),
    }
    df = pd.DataFrame(data, index=METRICS)

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(df, annot=True, fmt=".4f", cmap="YlOrRd", ax=ax,
                linewidths=0.5, linecolor="#dddddd",
                annot_kws={"size": 10}, cbar_kws={"shrink": 0.8})
    ax.set_title(f"Metric Heatmap — {task_title}", fontsize=13, fontweight="bold", pad=12)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha="right", fontsize=10)
    plt.tight_layout()
    out = FIGURE_DIR / fname
    fig.savefig(out, bbox_inches="tight", dpi=200)
    print(f"Saved → {out}")
    plt.close(fig)

print("\nAll figures saved to evaluation/figures/")
