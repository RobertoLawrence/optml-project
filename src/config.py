"""
config.py — all configuration dataclasses and the experiment grid generator.

The 120-experiment grid comes from the cell-27 table in fedLearning.ipynb:
  NUM_CLIENTS        : [4, 32]
  DATA_DISTRIBUTION  : multi_language | single_language | dirichlet(0.1) | dirichlet(0.5) | dirichlet(2.0)
  AGGREGATOR         : fedavg | fedadam
  CLIENT_REGULARIZATION : None | fedprox
  OPTIMIZER          : adamw | adam | sgd
  => 2 × 5 × 2 × 2 × 3 = 120
"""

from __future__ import annotations
import hashlib, json
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Static configs (shared across all experiments)
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    num_train_per_lang: int = 20_000
    num_val_per_lang: int   = 2_000
    # (lang_code, num_train, num_val) — num_train/val overridden at runtime
    languages: List[Tuple[str, int, int]] = field(default_factory=lambda: [
        ("fra_Latn", 20_000, 2_000),
        ("spa_Latn", 20_000, 2_000),
        ("ita_Latn", 20_000, 2_000),
        ("por_Latn", 20_000, 2_000),
    ])
    vocab_size: int = 8192
    default_iid_single_language: str = "fra_Latn"

    def language_codes(self) -> List[str]:
        return [lc for lc, _, _ in self.languages]


@dataclass
class ModelConfig:
    depth: int       = 8
    aspect_ratio: int = 64
    head_dim: int    = 64
    max_seq_len: int = 1024
    use_compile: bool = True
    seed: int = 42


@dataclass
class TrainingConfig:
    """Fixed training hyper-parameters (not swept)."""
    device_batch_size: int = 128
    total_batch_size: int  = 2**17
    eval_interval: int     = 50
    eval_batches: int      = 10
    weight_decay: float    = 0.1
    warmup_ratio: float    = 0.05
    warmdown_ratio: float  = 0.3
    final_lr_frac: float   = 0.1
    # LR per optimizer
    lr_adamw: float = 3e-4
    lr_adam: float  = 3e-4
    lr_sgd: float   = 0.05

    def lr_for(self, optimizer: str) -> float:
        return {"adamw": self.lr_adamw, "adam": self.lr_adam, "sgd": self.lr_sgd}[optimizer]


# ---------------------------------------------------------------------------
# Per-experiment config (the swept axes)
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    # ---- swept axes ----
    num_clients: int               = 4
    data_distribution: str         = "multi_language"   # multi_language | single_language | dirichlet
    dirichlet_alpha: float         = 0.5
    aggregator: str                = "fedavg"            # fedavg | fedadam
    client_regularization: Optional[str] = None          # None | fedprox
    optimizer: str                 = "adamw"             # adamw | adam | sgd

    # ---- fixed FL params (can be overridden from CLI) ----
    num_rounds: int       = 20
    local_steps: int      = 50
    fedprox_mu: float     = 0.01
    run_baseline: bool    = True
    eval_every_n_rounds: int = 1
    eval_batches_fl: int  = 20
    track_grad_divergence: bool = True
    track_comm_cost: bool       = True
    tokens_per_client: Optional[int] = None

    # ---- server-side adam LR ----
    fedadam_lr: float = 0.01

    def fl_batch_size(self, device_batch_size: int) -> int:
        """Per-client batch size; at least 1 to avoid zero-size batches."""
        return max(1, device_batch_size // self.num_clients)

    def baseline_steps(self) -> int:
        return self.local_steps * self.num_rounds

    # ---- identity / persistence ----

    def exp_id(self) -> str:
        """
        Human-readable stable identifier, e.g.
        'c4_multi_language_fedavg_none_adamw'
        'c32_dirichlet0.1_fedadam_fedprox_sgd'
        """
        dist = self.data_distribution
        if dist == "dirichlet":
            dist = f"dirichlet{self.dirichlet_alpha}"
        reg = self.client_regularization or "none"
        return f"c{self.num_clients}_{dist}_{self.aggregator}_{reg}_{self.optimizer}"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentConfig":
        return cls(**d)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, s: str) -> "ExperimentConfig":
        return cls.from_dict(json.loads(s))


# ---------------------------------------------------------------------------
# Experiment grid
# ---------------------------------------------------------------------------

def generate_grid(
    num_rounds: int = 20,
    local_steps: int = 50,
    run_baseline: bool = True,
) -> List[ExperimentConfig]:
    """
    Return all 120 ExperimentConfig instances for the full sweep.

    Distribution axis encodes alpha inside the string when relevant:
      multi_language, single_language, dirichlet (alpha 0.1 / 0.5 / 2.0)
    """
    dist_axis = [
        ("multi_language",  0.5),
        ("single_language", 0.5),
        ("dirichlet",       0.1),
        ("dirichlet",       0.5),
        ("dirichlet",       2.0),
    ]
    experiments: List[ExperimentConfig] = []
    for num_clients in [4, 32]:
        for dist, alpha in dist_axis:
            for aggregator in ["fedavg", "fedadam"]:
                for reg in [None, "fedprox"]:
                    for opt in ["adamw", "adam", "sgd"]:
                        experiments.append(ExperimentConfig(
                            num_clients=num_clients,
                            data_distribution=dist,
                            dirichlet_alpha=alpha,
                            aggregator=aggregator,
                            client_regularization=reg,
                            optimizer=opt,
                            num_rounds=num_rounds,
                            local_steps=local_steps,
                            run_baseline=run_baseline,
                        ))
    return experiments


def filter_pending(
    experiments: List[ExperimentConfig],
    results_dir: str,
) -> List[ExperimentConfig]:
    """Return only experiments whose result file does not yet exist."""
    import os
    pending = []
    for exp in experiments:
        result_path = os.path.join(results_dir, exp.exp_id(), "history.json")
        if not os.path.exists(result_path):
            pending.append(exp)
    return pending
