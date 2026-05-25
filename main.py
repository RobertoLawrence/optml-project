"""
main.py — entry point for the federated learning experiment sweep.

Usage
-----
# Run all 120 experiments, resuming where you left off (default):
    python main.py

# Start fresh, overwriting any past results:
    python main.py --overwrite

# Only run a specific experiment by its ID:
    python main.py --exp-id c4_multi_language_fedavg_none_adamw

# Generate comparison graphs from completed results:
    python main.py --generate-graphs

# Enable wandb logging:
    python main.py --wandb --wandb-project optml-fedlearning

# Full example:
    python main.py --overwrite --wandb --wandb-project optml-fedlearning --num-rounds 20

Environment variables
---------------------
  HF_TOKEN    — HuggingFace token (optional, for higher download rate limits)
  WANDB_API_KEY — wandb API key (or pass --wandb-key)
"""

from __future__ import annotations
import argparse, gc, json, os, sys, time, traceback
from pathlib import Path

# ---- make sure src/ is importable when running from project root ----
sys.path.insert(0, str(Path(__file__).parent))

from src.config import (
    DataConfig, ModelConfig, TrainingConfig, ExperimentConfig,
    generate_grid, filter_pending,
)
from src.data import DataManager
from src.training import run_fl_experiment, setup_device
from src.visualization import per_experiment_plot, generate_final_graphs


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent
CACHE_DIR    = PROJECT_ROOT / "cache"
RESULTS_DIR  = PROJECT_ROOT / "results"
FIGS_DIR     = PROJECT_ROOT / "figs"


# ---------------------------------------------------------------------------
# Result persistence
# ---------------------------------------------------------------------------

def save_result(result: dict, exp_id: str) -> Path:
    exp_dir = RESULTS_DIR / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    path = exp_dir / "history.json"

    # Convert numpy types to plain Python for JSON serialisation
    def _clean(obj):
        import numpy as np
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_clean(v) for v in obj]
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(path, "w") as f:
        json.dump(_clean(result), f, indent=2)
    return path


def result_exists(exp_id: str) -> bool:
    return (RESULTS_DIR / exp_id / "history.json").exists()


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Federated learning experiment sweep over 120 parameter combinations."
    )

    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--overwrite", action="store_true",
        help="Re-run all experiments, overwriting existing results.",
    )
    mode.add_argument(
        "--generate-graphs", action="store_true",
        help="Skip training; load completed results and produce comparison figures.",
    )

    p.add_argument(
        "--exp-id", type=str, default=None,
        help="Run only the experiment with this ID (e.g. c4_multi_language_fedavg_none_adamw).",
    )

    # FL hyper-parameters (override grid defaults)
    p.add_argument("--num-rounds",  type=int, default=20,
                   help="Number of FL communication rounds (default: 20).")
    p.add_argument("--local-steps", type=int, default=50,
                   help="Local optimisation steps per client per round (default: 50).")
    p.add_argument("--no-baseline", action="store_true",
                   help="Skip the centralized baseline training.")

    # Data
    p.add_argument("--hf-token", type=str,
                   default=os.environ.get("HF_TOKEN", ""),
                   help="HuggingFace token (can also be set via HF_TOKEN env var).")
    p.add_argument("--cache-dir", type=str, default=str(CACHE_DIR),
                   help="Directory for cached tokenizer and token tensors.")
    p.add_argument("--results-dir", type=str, default=str(RESULTS_DIR),
                   help="Directory where per-experiment results are stored.")
    p.add_argument("--figs-dir", type=str, default=str(FIGS_DIR),
                   help="Directory where figures are saved.")

    # wandb
    p.add_argument("--wandb", action="store_true",
                   help="Enable Weights & Biases logging.")
    p.add_argument("--wandb-project", type=str, default="optml-fedlearning",
                   help="wandb project name.")
    p.add_argument("--wandb-entity", type=str, default=None,
                   help="wandb entity (team/username).")
    p.add_argument("--wandb-key", type=str,
                   default=os.environ.get("WANDB_API_KEY", ""),
                   help="wandb API key (can also be set via WANDB_API_KEY env var).")

    return p.parse_args()


# ---------------------------------------------------------------------------
# wandb helpers
# ---------------------------------------------------------------------------

def init_wandb(args, exp_config: ExperimentConfig):
    import wandb
    if args.wandb_key:
        wandb.login(key=args.wandb_key)
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=exp_config.exp_id(),
        config=exp_config.to_dict(),
        reinit=True,
    )
    return run


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    results_dir = args.results_dir
    figs_dir    = args.figs_dir
    cache_dir   = args.cache_dir
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figs_dir, exist_ok=True)

    # ---- Graph-only mode ----
    if args.generate_graphs:
        print("[main] Graph generation mode — loading completed results...")
        generate_final_graphs(results_dir=results_dir, figs_dir=figs_dir)
        print("[main] Done.")
        return

    # ---- Build experiment grid ----
    grid = generate_grid(
        num_rounds=args.num_rounds,
        local_steps=args.local_steps,
        run_baseline=not args.no_baseline,
    )

    if args.exp_id:
        grid = [e for e in grid if e.exp_id() == args.exp_id]
        if not grid:
            print(f"[main] ERROR: no experiment with id={args.exp_id!r}")
            print("[main] Valid IDs:")
            for e in generate_grid():
                print(f"  {e.exp_id()}")
            sys.exit(1)

    if not args.overwrite:
        pending = filter_pending(grid, results_dir)
        skipped = len(grid) - len(pending)
        if skipped:
            print(f"[main] Resuming: {skipped} experiments already completed, "
                  f"{len(pending)} remaining.")
        grid = pending

    if not grid:
        print("[main] Nothing to run — all experiments complete.")
        print("[main] Use --overwrite to re-run, or --generate-graphs to plot.")
        return

    print(f"[main] Running {len(grid)} experiment(s).")

    # ---- Static configs ----
    data_config  = DataConfig()
    model_config = ModelConfig()
    train_config = TrainingConfig()

    # ---- Prepare data (download + tokenize, cached) ----
    dm = DataManager(
        config=data_config,
        cache_dir=cache_dir,
        hf_token=args.hf_token or None,
    )
    dm.prepare()

    tok = dm.get_tokenizer()
    train_config.actual_vocab_size = tok.get_vocab_size()  # type: ignore[attr-defined]
    lang_train_tensors, lang_val_tensors = dm.get_tensors()
    lang_codes = data_config.language_codes()

    print(f"[main] Vocabulary size: {train_config.actual_vocab_size}")
    for lc in lang_codes:
        print(f"  {lc}: {len(lang_train_tensors[lc]):>10,} train  "
              f"{len(lang_val_tensors[lc]):>8,} val tokens")

    # ---- Device ----
    device, ac_type, ac_dtype = setup_device()
    print(f"[main] Device: {device}"
          + (f" ({device_info()})" if device.type == "cuda" else ""))

    # ================================================================
    # Experiment loop
    # ================================================================
    total        = len(grid)
    completed    = 0
    failed_ids   = []
    t_sweep_start = time.time()

    for idx, exp_config in enumerate(grid):
        exp_id = exp_config.exp_id()

        if not args.overwrite and result_exists(exp_id):
            print(f"[{idx+1}/{total}] SKIP (already done): {exp_id}")
            completed += 1
            continue

        print(f"\n[{idx+1}/{total}] START: {exp_id}")
        t_exp = time.time()

        wandb_run = None
        if args.wandb:
            try:
                wandb_run = init_wandb(args, exp_config)
            except Exception as e:
                print(f"  [wandb] WARNING: could not init wandb run: {e}")

        try:
            result = run_fl_experiment(
                exp_config=exp_config,
                model_config=model_config,
                train_config=train_config,
                lang_train_tensors=lang_train_tensors,
                lang_val_tensors=lang_val_tensors,
                lang_codes=lang_codes,
                device=device,
                ac_type=ac_type,
                ac_dtype=ac_dtype,
                wandb_run=wandb_run,
            )

            # Save result to disk
            result_path = save_result(result, exp_id)
            print(f"  Result saved → {result_path}")

            # Per-experiment plot
            try:
                fig_path = per_experiment_plot(
                    history=result["history"],
                    baseline_history=result.get("baseline_history"),
                    lang_fracs=result["lang_fracs"],
                    lang_codes=result["lang_codes"],
                    exp_id=exp_id,
                    figs_dir=figs_dir,
                )
                print(f"  Figure saved → {fig_path}")
            except Exception as plot_err:
                print(f"  [viz] WARNING: plot failed: {plot_err}")

            completed += 1
            elapsed = time.time() - t_exp
            total_elapsed = time.time() - t_sweep_start
            remaining = (total - completed)
            eta = (total_elapsed / completed * remaining) if completed else 0
            print(
                f"  [{idx+1}/{total}] DONE in {elapsed:.0f}s | "
                f"completed={completed}/{total} | ETA ~{eta/60:.0f} min"
            )

        except Exception as exc:
            print(f"  [ERROR] Experiment {exp_id} FAILED:")
            traceback.print_exc()
            failed_ids.append(exp_id)
            # Save error marker so we know it was attempted
            err_dir = Path(results_dir) / exp_id
            err_dir.mkdir(parents=True, exist_ok=True)
            (err_dir / "error.txt").write_text(
                f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}"
            )

        finally:
            if wandb_run is not None:
                try:
                    wandb_run.finish()
                except Exception:
                    pass
            gc.collect()

    # ================================================================
    # Summary
    # ================================================================
    total_time = time.time() - t_sweep_start
    print(f"\n{'='*60}")
    print(f"Sweep complete: {completed}/{total} experiments done in "
          f"{total_time/3600:.1f}h")
    if failed_ids:
        print(f"Failed ({len(failed_ids)}):")
        for fid in failed_ids:
            print(f"  {fid}")

    # Auto-generate comparison graphs when full sweep is done
    all_exp_ids = {e.exp_id() for e in generate_grid(
        num_rounds=args.num_rounds,
        local_steps=args.local_steps,
        run_baseline=not args.no_baseline,
    )}
    done_exp_ids = {
        d for d in os.listdir(results_dir)
        if (Path(results_dir) / d / "history.json").exists()
    }
    if all_exp_ids.issubset(done_exp_ids):
        print("\n[main] All 120 experiments complete — generating final graphs...")
        generate_final_graphs(results_dir=results_dir, figs_dir=figs_dir)
    else:
        remaining = all_exp_ids - done_exp_ids
        print(f"\n[main] {len(remaining)} experiments still pending — "
              f"run with --generate-graphs once all are done.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def device_info() -> str:
    import torch
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name()
        mem  = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return f"{name}, {mem:.1f} GB"
    return "CPU"


if __name__ == "__main__":
    main()
