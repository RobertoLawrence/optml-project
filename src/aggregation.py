"""
aggregation.py — server-side aggregation algorithms.

FedAvg  : plain weighted average of client parameters.
FedAdam : adaptive server optimizer (Adam on the pseudo-gradient).
"""

from __future__ import annotations
from typing import Dict, List

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# FedAvg
# ---------------------------------------------------------------------------

def fedavg_aggregate(
    global_model: nn.Module,
    client_state_dicts: List[Dict[str, torch.Tensor]],
    client_weights: List[float],
) -> None:
    """Weighted average of CPU client state_dicts into global_model in-place."""
    total     = sum(client_weights)
    global_sd = global_model.state_dict()
    for key in global_sd:
        agg = sum(
            sd[key].float() * (w / total)
            for sd, w in zip(client_state_dicts, client_weights)
        ).to(dtype=global_sd[key].dtype, device=global_sd[key].device)
        global_sd[key] = agg
    global_model.load_state_dict(global_sd)


# ---------------------------------------------------------------------------
# FedAdam
# ---------------------------------------------------------------------------

class FedAdam:
    """
    Adaptive server optimizer for federated learning.

    The pseudo-gradient is the difference between the weighted-average of
    client parameters and the current global model.  Adam moment estimates
    are maintained server-side across rounds.

    lr    : server-side learning rate (default 0.01 — separate from local LR)
    beta1 : first-moment decay
    beta2 : second-moment decay
    eps   : numerical stability term (larger than standard Adam's 1e-8
            to stabilise the heterogeneous FL setting)
    """

    def __init__(
        self,
        model: nn.Module,
        lr:    float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps:   float = 1e-3,
    ):
        self.model = model
        self.lr    = lr
        self.b1    = beta1
        self.b2    = beta2
        self.eps   = eps
        self.m = {k: torch.zeros_like(v) for k, v in model.state_dict().items()}
        self.v = {k: torch.zeros_like(v) for k, v in model.state_dict().items()}
        self.t = 0

    def step(
        self,
        client_state_dicts: List[Dict[str, torch.Tensor]],
        client_weights: List[float],
    ) -> None:
        self.t += 1
        total     = sum(client_weights)
        global_sd = self.model.state_dict()

        for key in global_sd:
            avg = sum(
                sd[key].float().to(global_sd[key].device) * (w / total)
                for sd, w in zip(client_state_dicts, client_weights)
            )
            g = avg - global_sd[key].float()

            self.m[key] = self.b1 * self.m[key] + (1 - self.b1) * g
            self.v[key] = self.b2 * self.v[key] + (1 - self.b2) * g ** 2

            m_hat = self.m[key] / (1 - self.b1 ** self.t)
            v_hat = self.v[key] / (1 - self.b2 ** self.t)

            global_sd[key] = (
                global_sd[key].float() + self.lr * m_hat / (v_hat.sqrt() + self.eps)
            ).to(global_sd[key].dtype)

        self.model.load_state_dict(global_sd)
