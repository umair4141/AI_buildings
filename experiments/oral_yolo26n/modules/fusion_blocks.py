"""Neck fusion blocks for the first two experiments.

Exp-1 GqROFConcat: Gaussian q-rung orthopair gate (learnable q).
Exp-2 PlainGateConcat: same gate structure from plain projected features (no q).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _soft_sigma(raw: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    return F.softplus(raw) + eps


class _BaseDualGate(nn.Module):
    """Shared residual dual-stream gate used by both experiments."""

    def __init__(self, c1: int, c2: int | None = None):
        super().__init__()
        # Ultralytics Concat-style modules often receive c1 as sum of channels after cat.
        # Here we expect a list/tuple of two feature maps with equal channels.
        self.c = int(c1) if c2 is None else int(c2)
        # If parse_model passes summed channels (2C), recover per-stream channels.
        if c2 is None and c1 % 2 == 0:
            self.c = c1 // 2
        self.proj_a = nn.Conv2d(self.c, self.c, 1, bias=False)
        self.proj_b = nn.Conv2d(self.c, self.c, 1, bias=False)
        self.out = nn.Conv2d(2 * self.c, 2 * self.c, 1, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))

    def gate_from_evidence(self, ea: torch.Tensor, eb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def forward(self, x):
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise ValueError(f"{self.__class__.__name__} expects 2 feature maps, got {type(x)}")
        fa, fb = x
        if fa.shape[1] != fb.shape[1]:
            raise ValueError(f"Channel mismatch: {fa.shape} vs {fb.shape}")

        ea = self.proj_a(fa)
        eb = self.proj_b(fb)
        ga, gb = self.gate_from_evidence(ea, eb)

        fa_g = fa * ga
        fb_g = fb * gb
        fused = torch.cat((fa_g, fb_g), dim=1)
        base = torch.cat((fa, fb), dim=1)
        return base + self.gamma * self.out(fused)


class GqROFConcat(_BaseDualGate):
    """Exp-1: q-rung Gaussian membership gate with learnable q."""

    def __init__(self, c1: int, c2: int | None = None):
        super().__init__(c1, c2)
        # Learnable Gaussian centers / widths for membership and non-membership
        self.c_mu = nn.Parameter(torch.zeros(1, self.c, 1, 1))
        self.c_nu = nn.Parameter(torch.zeros(1, self.c, 1, 1))
        self.s_mu = nn.Parameter(torch.ones(1, self.c, 1, 1))
        self.s_nu = nn.Parameter(torch.ones(1, self.c, 1, 1))
        # q = 1 + 5*sigmoid(rho); init near q=2 => sigmoid(rho)=0.2 => rho≈-1.386
        self.rho = nn.Parameter(torch.tensor(-1.3863))

    def _orthopair(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = 1.0 + 5.0 * torch.sigmoid(self.rho)
        sig_mu = _soft_sigma(self.s_mu)
        sig_nu = _soft_sigma(self.s_nu)
        mu = torch.exp(-((x - self.c_mu) ** 2) / (2.0 * sig_mu**2))
        mu = mu.clamp(1e-6, 1.0)
        # Constrained non-membership so mu^q + nu^q <= 1
        nu = (1.0 - mu.pow(q)).clamp(min=0.0).pow(1.0 / q) * torch.exp(
            -((x - self.c_nu) ** 2) / (2.0 * sig_nu**2)
        )
        nu = nu.clamp(0.0, 1.0)
        s = (mu.pow(q) - nu.pow(q) + 1.0) * 0.5
        return mu, nu, s

    def gate_from_evidence(self, ea: torch.Tensor, eb: torch.Tensor):
        _, _, sa = self._orthopair(ea)
        _, _, sb = self._orthopair(eb)
        # Map score in (0,1) -> positive gate around 1
        ga = 0.5 + sa
        gb = 0.5 + sb
        return ga, gb


class PlainGateConcat(_BaseDualGate):
    """Exp-2 control A: gate from plain projected features (no q-rung)."""

    def __init__(self, c1: int, c2: int | None = None):
        super().__init__(c1, c2)
        self.gate_a = nn.Sequential(nn.Conv2d(self.c, self.c, 1, bias=True), nn.Sigmoid())
        self.gate_b = nn.Sequential(nn.Conv2d(self.c, self.c, 1, bias=True), nn.Sigmoid())

    def gate_from_evidence(self, ea: torch.Tensor, eb: torch.Tensor):
        return self.gate_a(ea), self.gate_b(eb)
