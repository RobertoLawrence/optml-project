"""
training.py — FL training loop and centralized baseline.

run_fl_experiment() is the main entry point called by main.py for each
parameter combination.  It returns a history dict which is also what gets
saved to disk and logged to wandb.
"""

from __future__ import annotations
import gc, time
from collections import defaultdict
from contextlib import nullcontext
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from src.config import ExperimentConfig, ModelConfig, TrainingConfig
from src.model import GPT, GPTConfig, build_model_config
from src.data import make_dataloader, create_client_datasets
from src.aggregation import fedavg_aggregate, FedAdam


# ---------------------------------------------------------------------------
# Device / autocast helpers
# ---------------------------------------------------------------------------

def setup_device() -> Tuple[torch.device, str, Optional[torch.dtype]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        ac_type = "cuda"
        try:
            with torch.amp.autocast(device_type=ac_type, dtype=torch.bfloat16):
                _ = torch.ones(1, device=device) @ torch.ones(1, device=device)
            ac_dtype = torch.bfloat16
        except RuntimeError:
            ac_dtype = torch.float16
    elif device.type == "cpu":
        ac_type, ac_dtype = "cpu", torch.bfloat16
    else:
        ac_type, ac_dtype = None, None
    return device, ac_type, ac_dtype


def autocast_context(ac_type, ac_dtype):
    if ac_type is None or ac_dtype is None:
        return nullcontext()
    return torch.amp.autocast(device_type=ac_type, dtype=ac_dtype)


# ---------------------------------------------------------------------------
# Optimizer factory
# ---------------------------------------------------------------------------

def make_optimizer(model: nn.Module, optimizer: str, lr: float, weight_decay: float):
    if optimizer == "adamw":
        return torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95)
        )
    elif optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.95))
    elif optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer!r}")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    data_tensor: torch.Tensor,
    device: torch.device,
    ac_type,
    ac_dtype,
    num_batches: int = 20,
    batch_size: int = 16,
    seq_len: int = 1024,
) -> float:
    model.eval()
    loader = make_dataloader(data_tensor, batch_size, seq_len, device)
    total  = 0.0
    for _ in range(num_batches):
        x, y = next(loader)
        with autocast_context(ac_type, ac_dtype):
            total += model(x, y).item()
    model.train()
    return total / num_batches


# ---------------------------------------------------------------------------
# LR schedule (used for baseline)
# ---------------------------------------------------------------------------

def get_lr_multiplier(
    step: int,
    total: int,
    warmup_ratio: float,
    warmdown_ratio: float,
    final_lr_frac: float,
) -> float:
    p = step / total
    if p < warmup_ratio:
        return p / warmup_ratio if warmup_ratio > 0 else 1.0
    elif p < 1.0 - warmdown_ratio:
        return 1.0
    else:
        return final_lr_frac + (1.0 - final_lr_frac) * (1.0 - p) / warmdown_ratio


# ---------------------------------------------------------------------------
# Params → flat vector (for gradient divergence tracking)
# ---------------------------------------------------------------------------

def params_to_vec(model_or_sd) -> torch.Tensor:
    if isinstance(model_or_sd, dict):
        tensors = [
            v.float().cpu().flatten()
            for v in model_or_sd.values() if torch.is_floating_point(v)
        ]
    else:
        tensors = [p.detach().float().cpu().flatten() for p in model_or_sd.parameters()]
    return torch.cat(tensors)


# ---------------------------------------------------------------------------
# Centralized baseline
# ---------------------------------------------------------------------------

def run_baseline(
    gpt_config: GPTConfig,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    exp_config: ExperimentConfig,
    train_config: TrainingConfig,
    device: torch.device,
    ac_type,
    ac_dtype,
    wandb_run=None,
) -> Dict:
    """
    Train a centralized model with the same total compute as the FL run
    (baseline_steps = local_steps × num_rounds).
    """
    baseline_steps = exp_config.baseline_steps()
    lr = train_config.lr_for(exp_config.optimizer)

    torch.manual_seed(train_config.__dict__.get("seed", 42))
    bl_model = GPT(gpt_config).to(device)
    bl_model.init_weights()

    if exp_config.run_baseline and train_config.__dict__.get("use_compile", True):
        bl_model_eval = torch.compile(bl_model)
    else:
        bl_model_eval = bl_model

    bl_opt    = make_optimizer(bl_model, exp_config.optimizer, lr, train_config.weight_decay)
    bl_loader = make_dataloader(
        train_data, train_config.device_batch_size, gpt_config.sequence_len, device
    )

    baseline_history: Dict = defaultdict(list)
    smooth_loss = 0.0
    log_every   = max(baseline_steps // 20, 1)

    print(f"  [baseline] {baseline_steps} steps, optimizer={exp_config.optimizer}, lr={lr}")
    t_bl = time.time()
    for step in range(baseline_steps):
        lrm = get_lr_multiplier(
            step, baseline_steps,
            train_config.warmup_ratio,
            train_config.warmdown_ratio,
            train_config.final_lr_frac,
        )
        for g in bl_opt.param_groups:
            g["lr"] = lr * lrm

        x, y = next(bl_loader)
        with autocast_context(ac_type, ac_dtype):
            loss = bl_model_eval(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(bl_model.parameters(), 1.0)
        bl_opt.step()
        bl_opt.zero_grad(set_to_none=True)

        smooth_loss = 0.9 * smooth_loss + 0.1 * loss.item()

        if (step + 1) % log_every == 0:
            val_loss = evaluate_loss(
                bl_model_eval, val_data, device, ac_type, ac_dtype,
                num_batches=train_config.eval_batches,
                batch_size=train_config.device_batch_size,
                seq_len=gpt_config.sequence_len,
            )
            corrected_train = smooth_loss / (1 - 0.9 ** (step + 1))
            baseline_history["step"].append(step + 1)
            baseline_history["train_loss"].append(corrected_train)
            baseline_history["val_loss"].append(val_loss)
            print(
                f"    step {step+1:5d}/{baseline_steps} | "
                f"train={corrected_train:.4f} | val={val_loss:.4f} | "
                f"lrm={lrm:.3f} | {time.time()-t_bl:.0f}s"
            )
            if wandb_run is not None:
                wandb_run.log({
                    "baseline/train_loss": corrected_train,
                    "baseline/val_loss":   val_loss,
                    "baseline/step":       step + 1,
                })

    print(f"  [baseline] done in {time.time()-t_bl:.0f}s")

    # Cleanup
    del bl_model, bl_model_eval, bl_opt, bl_loader
    gc.collect()
    try:
        torch._dynamo.reset()
    except Exception:
        pass
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return dict(baseline_history)


# ---------------------------------------------------------------------------
# FL training loop
# ---------------------------------------------------------------------------

def run_fl_experiment(
    exp_config: ExperimentConfig,
    model_config: ModelConfig,
    train_config: TrainingConfig,
    lang_train_tensors: Dict[str, torch.Tensor],
    lang_val_tensors:   Dict[str, torch.Tensor],
    lang_codes: List[str],
    device: torch.device,
    ac_type,
    ac_dtype,
    wandb_run=None,
) -> Dict:
    """
    Run one complete FL experiment (FL training loop + optional baseline).

    Returns a dict with keys: history, baseline_history, lang_fracs, config.
    This dict is serialised to results/{exp_id}/history.json by main.py.
    """
    # ---- derive model dimensions ----
    gpt_cfg = build_model_config(
        depth=model_config.depth,
        aspect_ratio=model_config.aspect_ratio,
        head_dim=model_config.head_dim,
        max_seq_len=model_config.max_seq_len,
        actual_vocab_size=lang_train_tensors[lang_codes[0]].max().item() + 1
        if False else 0,  # placeholder — overridden below
    )
    # Actually compute vocab size from the tokenizer's actual size (passed via train_config)
    actual_vocab_size = getattr(train_config, "actual_vocab_size", 8192)
    gpt_cfg = build_model_config(
        depth=model_config.depth,
        aspect_ratio=model_config.aspect_ratio,
        head_dim=model_config.head_dim,
        max_seq_len=model_config.max_seq_len,
        actual_vocab_size=actual_vocab_size,
    )

    fl_bs = exp_config.fl_batch_size(train_config.device_batch_size)
    lr    = train_config.lr_for(exp_config.optimizer)

    print(f"\n{'='*70}")
    print(f"  Experiment: {exp_config.exp_id()}")
    print(f"  model: depth={model_config.depth}, dim={gpt_cfg.n_embd}, "
          f"params={_fmt_params(gpt_cfg)}")
    print(f"  FL: {exp_config.num_clients} clients × {exp_config.num_rounds} rounds "
          f"× {exp_config.local_steps} local steps")
    print(f"  dist={exp_config.data_distribution}  agg={exp_config.aggregator}  "
          f"reg={exp_config.client_regularization}  opt={exp_config.optimizer}  lr={lr}")
    print(f"  fl_batch_size={fl_bs}")
    print(f"{'='*70}\n")

    # ---- build client datasets ----
    (client_train_tensors,
     client_val_tensors,
     global_val_tensor,
     lang_fracs) = create_client_datasets(
        lang_train_tensors=lang_train_tensors,
        lang_val_tensors=lang_val_tensors,
        num_clients=exp_config.num_clients,
        data_distribution=exp_config.data_distribution,
        dirichlet_alpha=exp_config.dirichlet_alpha,
        tokens_per_client=exp_config.tokens_per_client,
        seed=getattr(model_config, "seed", 42),
    )

    # Flat train/val for baseline
    train_data = torch.cat(list(lang_train_tensors.values()))
    val_data   = torch.cat(list(lang_val_tensors.values()))

    # ---- build global model ----
    torch.manual_seed(getattr(model_config, "seed", 42))
    global_model = GPT(gpt_cfg).to(device)
    global_model.init_weights()

    use_compile = getattr(model_config, "use_compile", True)
    eval_model    = torch.compile(global_model) if use_compile else global_model
    client_base   = GPT(gpt_cfg).to(device)
    compiled_client = torch.compile(client_base) if use_compile else client_base

    server_opt = None
    if exp_config.aggregator == "fedadam":
        server_opt = FedAdam(global_model, lr=exp_config.fedadam_lr)

    n_params = global_model.num_params()
    history: Dict = defaultdict(list)
    t_fl_start = time.time()

    # ================================================================
    # Main FL loop
    # ================================================================
    for round_idx in range(exp_config.num_rounds):
        t_round = time.time()

        global_sd_cpu = {k: v.detach().cpu() for k, v in global_model.state_dict().items()}
        global_vec    = params_to_vec(global_sd_cpu) if exp_config.track_grad_divergence else None

        client_state_dicts:  List = []
        client_weights:      List = []
        round_train_losses:  List = []
        round_val_losses:    List = []
        client_deltas:       List = []

        # ---- per-client local training ----
        for ci in range(exp_config.num_clients):
            client_base.load_state_dict(global_sd_cpu)
            compiled_client.train()
            opt    = make_optimizer(compiled_client, exp_config.optimizer, lr,
                                    train_config.weight_decay)
            loader = make_dataloader(
                client_train_tensors[ci], fl_bs, gpt_cfg.sequence_len, device
            )

            step_losses     = []
            step_grad_norms = []

            global_params_for_prox = None
            if exp_config.client_regularization == "fedprox":
                global_params_for_prox = [
                    p.detach().clone() for p in client_base.parameters()
                ]

            for _ in range(exp_config.local_steps):
                x, y = next(loader)
                with autocast_context(ac_type, ac_dtype):
                    loss = compiled_client(x, y)
                    if exp_config.client_regularization == "fedprox":
                        prox = sum(
                            ((p - p0) ** 2).sum()
                            for p, p0 in zip(compiled_client.parameters(),
                                             global_params_for_prox)
                        )
                        loss = loss + (exp_config.fedprox_mu / 2) * prox

                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    compiled_client.parameters(), 1.0
                )
                opt.step()
                opt.zero_grad(set_to_none=True)
                step_losses.append(loss.item())
                step_grad_norms.append(grad_norm.item())

            round_train_losses.append(float(np.mean(step_losses)))
            round_val_losses.append(
                evaluate_loss(
                    client_base, client_val_tensors[ci], device, ac_type, ac_dtype,
                    num_batches=exp_config.eval_batches_fl,
                    batch_size=fl_bs,
                    seq_len=gpt_cfg.sequence_len,
                )
            )
            history[f"client_grad_norms_{ci}"].append(float(np.mean(step_grad_norms)))
            client_weights.append(len(client_train_tensors[ci]))

            client_sd_cpu = {k: v.detach().cpu() for k, v in client_base.state_dict().items()}
            client_state_dicts.append(client_sd_cpu)

            if exp_config.track_grad_divergence:
                client_deltas.append(params_to_vec(client_sd_cpu) - global_vec)

            del opt, loader
            if device.type == "cuda":
                torch.cuda.empty_cache()

        # ---- aggregation ----
        if exp_config.aggregator == "fedavg":
            fedavg_aggregate(global_model, client_state_dicts, client_weights)
        elif exp_config.aggregator == "fedadam":
            server_opt.step(client_state_dicts, client_weights)

        # ---- global eval ----
        if (round_idx + 1) % exp_config.eval_every_n_rounds == 0:
            gval = evaluate_loss(
                eval_model, global_val_tensor, device, ac_type, ac_dtype,
                num_batches=exp_config.eval_batches_fl,
                batch_size=train_config.device_batch_size,
                seq_len=gpt_cfg.sequence_len,
            )
            history["eval_rounds"].append(round_idx + 1)
            history["global_val_loss"].append(gval)

        # ---- optional metrics ----
        if exp_config.track_grad_divergence and client_deltas:
            mean_delta = torch.stack(client_deltas).mean(0)
            cos_sims = [
                torch.nn.functional.cosine_similarity(
                    d.unsqueeze(0), mean_delta.unsqueeze(0)
                ).item()
                for d in client_deltas
            ]
            history["grad_divergence"].append(1.0 - float(np.mean(cos_sims)))

        if exp_config.track_comm_cost:
            round_cost = 2 * n_params * 4 * exp_config.num_clients / 1024 ** 2
            prev       = history["cum_comm_mb"][-1] if history["cum_comm_mb"] else 0.0
            history["cum_comm_mb"].append(prev + round_cost)

        history["local_train_losses"].append(round_train_losses)
        history["local_val_losses"].append(round_val_losses)
        history["round_times"].append(time.time() - t_round)

        gval_str = (
            f"{history['global_val_loss'][-1]:.4f}"
            if history["global_val_loss"] else "—"
        )
        div_str = (
            f"  div={history['grad_divergence'][-1]:.3f}"
            if history["grad_divergence"] else ""
        )
        print(
            f"  Round {round_idx+1:3d}/{exp_config.num_rounds} | "
            f"global_val={gval_str} | "
            f"local_train={np.mean(round_train_losses):.4f} | "
            f"local_val={np.mean(round_val_losses):.4f}"
            f"{div_str} | {history['round_times'][-1]:.1f}s"
        )

        if wandb_run is not None:
            log_dict = {
                "fl/round":             round_idx + 1,
                "fl/local_train_loss":  float(np.mean(round_train_losses)),
                "fl/local_val_loss":    float(np.mean(round_val_losses)),
            }
            if history["global_val_loss"]:
                log_dict["fl/global_val_loss"] = history["global_val_loss"][-1]
            if history["grad_divergence"]:
                log_dict["fl/grad_divergence"] = history["grad_divergence"][-1]
            wandb_run.log(log_dict, step=round_idx + 1)

    total_fl_time = time.time() - t_fl_start
    print(
        f"\n  FL done — {total_fl_time:.0f}s total, "
        f"{total_fl_time/exp_config.num_rounds:.1f}s/round"
    )

    # ---- GPU cleanup after FL ----
    del global_model, eval_model, client_base, compiled_client
    if server_opt is not None:
        del server_opt
    del client_state_dicts, client_deltas, global_sd_cpu
    gc.collect()
    try:
        torch._dynamo.reset()
    except Exception:
        pass
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    # ---- optional centralized baseline ----
    baseline_history = None
    if exp_config.run_baseline:
        print("  Running centralized baseline...")
        baseline_history = run_baseline(
            gpt_cfg, train_data, val_data,
            exp_config, train_config,
            device, ac_type, ac_dtype,
            wandb_run=wandb_run,
        )

    return {
        "history":          dict(history),
        "baseline_history": baseline_history,
        "lang_fracs":       lang_fracs.tolist(),
        "lang_codes":       lang_codes,
        "total_fl_time":    total_fl_time,
        "config":           exp_config.to_dict(),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_params(gpt_cfg: GPTConfig) -> str:
    from src.model import GPT
    m = GPT(gpt_cfg)
    n = m.num_params()
    del m
    if n >= 1e6:
        return f"{n/1e6:.1f}M"
    return f"{n/1e3:.1f}K"
