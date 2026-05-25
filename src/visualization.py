"""
visualization.py — plotting helpers.

Two modes:
  1. per_experiment_plot()  — the 3×2 panel from the notebook, one per run.
  2. generate_final_graphs() — cross-experiment comparison plots over all results.
"""

from __future__ import annotations
import json, os
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# ---------------------------------------------------------------------------
# Single-experiment plots  (mirrors notebook cell 26)
# ---------------------------------------------------------------------------

def per_experiment_plot(
    history: Dict,
    baseline_history: Optional[Dict],
    lang_fracs: List[List[float]],
    lang_codes: List[str],
    exp_id: str,
    figs_dir: str = "figs",
) -> str:
    """Draw the 3×2 panel for one experiment and save to figs_dir. Returns path."""
    os.makedirs(figs_dir, exist_ok=True)

    num_rounds = len(history.get("local_train_losses", []))
    num_clients = len(lang_fracs)
    rounds = np.arange(1, num_rounds + 1)
    local_steps = 1  # placeholder for x-axis alignment; overridden below if baseline exists
    if baseline_history and baseline_history.get("step"):
        # infer from baseline steps vs rounds
        local_steps = baseline_history["step"][-1] // num_rounds if num_rounds else 1

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle(exp_id, fontsize=11, y=1.01)

    # ---- 1: Global validation loss ----
    ax = axes[0, 0]
    if history.get("eval_rounds"):
        ax.plot(history["eval_rounds"], history["global_val_loss"],
                "b-o", ms=4, lw=1.5, label="FL — global val")
    if baseline_history and baseline_history.get("step"):
        bl_rounds = [s / local_steps for s in baseline_history["step"]]
        ax.plot(bl_rounds, baseline_history["val_loss"],
                "r--", lw=1.5, label="Centralized baseline — val")
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Validation loss")
    ax.set_title("Global Validation Loss")
    ax.legend(); ax.grid(True, alpha=0.3)

    # ---- 2: Local training loss (mean ± range) ----
    ax = axes[0, 1]
    if history.get("local_train_losses"):
        local_train_arr = np.array(history["local_train_losses"])
        ax.fill_between(rounds, local_train_arr.min(1), local_train_arr.max(1),
                        alpha=0.15, color="green", label="Client range")
        ax.plot(rounds, local_train_arr.mean(1), "g-", lw=1.5, label="Mean across clients")
    if baseline_history and baseline_history.get("step"):
        bl_rounds = [s / local_steps for s in baseline_history["step"]]
        ax.plot(bl_rounds, baseline_history["train_loss"],
                "r--", lw=1.2, label="Centralized baseline — train")
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Training loss")
    ax.set_title("Local Training Loss")
    ax.legend(); ax.grid(True, alpha=0.3)

    # ---- 3: Per-client local val loss ----
    ax = axes[1, 0]
    if history.get("local_val_losses"):
        local_val_arr = np.array(history["local_val_losses"])
        for ci in range(num_clients):
            dom_lang = lang_codes[int(np.argmax(lang_fracs[ci]))]
            ax.plot(rounds, local_val_arr[:, ci],
                    label=f"C{ci} ({dom_lang})", alpha=0.8, lw=1.2)
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Validation loss")
    ax.set_title("Per-Client Local Validation Loss")
    ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.3)

    # ---- 4: Gradient divergence or comm cost ----
    ax = axes[1, 1]
    if history.get("grad_divergence"):
        r_div = np.arange(1, len(history["grad_divergence"]) + 1)
        ax.plot(r_div, history["grad_divergence"], "m-o", ms=4, lw=1.5)
        ax.set_xlabel("Communication round")
        ax.set_ylabel("1 − mean cosine similarity")
        ax.set_title("Client Update Divergence")
        ax.grid(True, alpha=0.3)
    elif history.get("cum_comm_mb"):
        r_comm = np.arange(1, len(history["cum_comm_mb"]) + 1)
        ax.plot(r_comm, history["cum_comm_mb"], "c-", lw=1.5)
        ax.set_xlabel("Communication round")
        ax.set_ylabel("Cumulative MB transmitted")
        ax.set_title("Communication Cost")
        ax.grid(True, alpha=0.3)

    # ---- 5: Per-client gradient norms ----
    ax = axes[2, 0]
    for ci in range(num_clients):
        norms = history.get(f"client_grad_norms_{ci}", [])
        if norms:
            dom_lang = lang_codes[int(np.argmax(lang_fracs[ci]))]
            ax.plot(range(1, len(norms) + 1), norms,
                    label=f"C{ci} ({dom_lang})", alpha=0.8, lw=1.2)
    ax.axhline(y=1.0, color="gray", linestyle=":", lw=1, label="Clip threshold")
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Gradient norm (pre-clip)")
    ax.set_title("Per-Client Gradient Norm (mean over local steps)")
    ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.3)

    # ---- 6: Empty / future use ----
    axes[2, 1].axis("off")

    plt.tight_layout()
    ts   = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = os.path.join(figs_dir, f"{exp_id}_{ts}.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Cross-experiment comparison plots
# ---------------------------------------------------------------------------

def _load_results(results_dir: str) -> List[Dict]:
    """Load all completed experiment result dicts from results_dir."""
    records = []
    for exp_id in sorted(os.listdir(results_dir)):
        path = os.path.join(results_dir, exp_id, "history.json")
        if os.path.isfile(path):
            with open(path) as f:
                records.append(json.load(f))
    return records


def _final_val_loss(record: Dict) -> Optional[float]:
    h = record.get("history", {})
    gvl = h.get("global_val_loss", [])
    return gvl[-1] if gvl else None


def generate_final_graphs(results_dir: str, figs_dir: str = "figs") -> None:
    """
    Load all completed experiments and produce cross-experiment comparison figures.

    Saves multiple PNG files to figs_dir:
      - comparison_by_aggregator.png
      - comparison_by_optimizer.png
      - comparison_by_distribution.png
      - comparison_by_num_clients.png
      - comparison_by_regularization.png
      - heatmap_final_val_loss.png
    """
    os.makedirs(figs_dir, exist_ok=True)
    records = _load_results(results_dir)
    if not records:
        print("[viz] No completed experiments found in", results_dir)
        return
    print(f"[viz] Loaded {len(records)} completed experiments.")

    # Build summary table
    rows = []
    for r in records:
        cfg = r.get("config", {})
        fvl = _final_val_loss(r)
        if fvl is None:
            continue
        rows.append({
            "exp_id":       cfg.get("aggregator", "?") + "/" + cfg.get("optimizer", "?"),
            "num_clients":  cfg.get("num_clients", "?"),
            "distribution": cfg.get("data_distribution", "?"),
            "alpha":        cfg.get("dirichlet_alpha", 0.5),
            "aggregator":   cfg.get("aggregator", "?"),
            "optimizer":    cfg.get("optimizer", "?"),
            "regularization": cfg.get("client_regularization") or "none",
            "final_val_loss": fvl,
            "history":      r.get("history", {}),
            "baseline":     r.get("baseline_history"),
        })

    if not rows:
        print("[viz] No valid results to plot.")
        return

    # ---- Helper: grouped bar chart ----
    def grouped_bar(group_key: str, title: str, filename: str):
        from collections import defaultdict
        groups = defaultdict(list)
        for row in rows:
            groups[str(row[group_key])].append(row["final_val_loss"])
        keys = sorted(groups.keys())
        means = [np.mean(groups[k]) for k in keys]
        stds  = [np.std(groups[k])  for k in keys]

        fig, ax = plt.subplots(figsize=(max(6, len(keys) * 1.2), 4))
        colors = cm.tab10(np.linspace(0, 1, len(keys)))
        bars = ax.bar(keys, means, yerr=stds, capsize=4, color=colors, alpha=0.85)
        ax.set_xlabel(group_key.replace("_", " ").title())
        ax.set_ylabel("Final global val loss (mean ± std)")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
        # Annotate mean values
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                    f"{mean:.3f}", ha="center", va="bottom", fontsize=8)
        plt.tight_layout()
        path = os.path.join(figs_dir, filename)
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"[viz] Saved {path}")

    grouped_bar("aggregator",    "Final Val Loss by Aggregator",        "comparison_by_aggregator.png")
    grouped_bar("optimizer",     "Final Val Loss by Optimizer",         "comparison_by_optimizer.png")
    grouped_bar("distribution",  "Final Val Loss by Data Distribution", "comparison_by_distribution.png")
    grouped_bar("num_clients",   "Final Val Loss by Number of Clients", "comparison_by_num_clients.png")
    grouped_bar("regularization","Final Val Loss by Client Regularization", "comparison_by_regularization.png")

    # ---- Learning curves: aggregator × optimizer ----
    _plot_learning_curves(rows, figs_dir)

    # ---- Heatmap: aggregator vs optimizer ----
    _plot_heatmap(rows, figs_dir)

    print(f"[viz] All comparison graphs saved to {figs_dir}/")


def _plot_learning_curves(rows: List[Dict], figs_dir: str):
    """Global val loss curves grouped by (aggregator, optimizer)."""
    from collections import defaultdict
    groups = defaultdict(list)
    for row in rows:
        key = f"{row['aggregator']}+{row['optimizer']}"
        gvl = row["history"].get("global_val_loss", [])
        if gvl:
            groups[key].append(gvl)

    if not groups:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = cm.tab20(np.linspace(0, 1, len(groups)))
    for (key, curves), color in zip(sorted(groups.items()), colors):
        max_len  = max(len(c) for c in curves)
        padded   = [c + [c[-1]] * (max_len - len(c)) for c in curves]
        arr      = np.array(padded)
        mean_c   = arr.mean(0)
        std_c    = arr.std(0)
        x = np.arange(1, max_len + 1)
        ax.plot(x, mean_c, label=key, color=color, lw=1.5)
        ax.fill_between(x, mean_c - std_c, mean_c + std_c, color=color, alpha=0.1)

    ax.set_xlabel("Communication round")
    ax.set_ylabel("Global validation loss")
    ax.set_title("Global Val Loss Curves by (Aggregator + Optimizer)\nmean ± std across all distribution/client configs")
    ax.legend(fontsize=7, ncol=3)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(figs_dir, "learning_curves_agg_opt.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] Saved {path}")


def _plot_heatmap(rows: List[Dict], figs_dir: str):
    """Heatmap: mean final val loss, axes = aggregator × optimizer."""
    import pandas as pd
    from collections import defaultdict

    data = defaultdict(lambda: defaultdict(list))
    for row in rows:
        data[row["aggregator"]][row["optimizer"]].append(row["final_val_loss"])

    aggregators = sorted(data.keys())
    optimizers  = sorted({row["optimizer"] for row in rows})

    matrix = np.full((len(aggregators), len(optimizers)), np.nan)
    for i, agg in enumerate(aggregators):
        for j, opt in enumerate(optimizers):
            vals = data[agg].get(opt, [])
            if vals:
                matrix[i, j] = np.mean(vals)

    fig, ax = plt.subplots(figsize=(max(5, len(optimizers) * 1.5),
                                    max(3, len(aggregators) * 1.2)))
    im = ax.imshow(matrix, cmap="RdYlGn_r", aspect="auto")
    ax.set_xticks(range(len(optimizers)))
    ax.set_xticklabels(optimizers)
    ax.set_yticks(range(len(aggregators)))
    ax.set_yticklabels(aggregators)
    ax.set_xlabel("Optimizer")
    ax.set_ylabel("Aggregator")
    ax.set_title("Mean Final Val Loss\n(lower is better)")
    plt.colorbar(im, ax=ax)
    for i in range(len(aggregators)):
        for j in range(len(optimizers)):
            if not np.isnan(matrix[i, j]):
                ax.text(j, i, f"{matrix[i,j]:.3f}",
                        ha="center", va="center", fontsize=9, color="black")
    plt.tight_layout()
    path = os.path.join(figs_dir, "heatmap_final_val_loss.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] Saved {path}")
