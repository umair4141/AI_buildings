"""Concat-compatible cross-stream gated fusion for YOLO26 neck merges."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _soft_sigma(raw: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    return F.softplus(raw) + eps


class _CrossStreamGate(nn.Module):
    """Drop-in Concat replacement with unequal-channel support and 2*sigmoid gates."""

    def __init__(self, dimension: int = 1):
        super().__init__()
        self.d = dimension
        self._inited = False
        self.logit_w = nn.Parameter(torch.tensor(0.0))

    def _lazy_init(self, fa: torch.Tensor, fb: torch.Tensor) -> None:
        ca, cb = fa.shape[1], fb.shape[1]
        self.ev_a = nn.Conv2d(ca, 1, kernel_size=1, bias=True)
        self.ev_b = nn.Conv2d(cb, 1, kernel_size=1, bias=True)
        self.cha = nn.Conv2d(1, ca, kernel_size=1, bias=True)
        self.chb = nn.Conv2d(1, cb, kernel_size=1, bias=True)
        for m in (self.ev_a, self.ev_b, self.cha, self.chb):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)
        self._init_extra(ca, cb)
        self.to(fa.device)
        self._inited = True

    def _init_extra(self, ca: int, cb: int) -> None:
        pass

    def fused_evidence(self, fa: torch.Tensor, fb: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x):
        if not isinstance(x, (list, tuple)) or len(x) < 2:
            raise ValueError(f"{self.__class__.__name__} expects 2 feature maps")
        fa, fb = x[0], x[1]
        if not self._inited:
            self._lazy_init(fa, fb)
        s = self.fused_evidence(fa, fb)
        ga = 2.0 * torch.sigmoid(self.cha(s))
        gb = 2.0 * torch.sigmoid(self.chb(s))
        return torch.cat((fa * ga, fb * gb), self.d)


class GqROFConcat(_CrossStreamGate):
    """Cross-stream Gaussian q-rung aggregation + matched 2*sigmoid gate."""

    def _init_extra(self, ca: int, cb: int) -> None:
        self.c_mu = nn.Parameter(torch.zeros(1, 1, 1, 1))
        self.c_nu = nn.Parameter(torch.zeros(1, 1, 1, 1))
        self.s_mu = nn.Parameter(torch.ones(1, 1, 1, 1))
        self.s_nu = nn.Parameter(torch.ones(1, 1, 1, 1))
        self.rho = nn.Parameter(torch.tensor(-1.3863))

    def _orthopair(self, e: torch.Tensor):
        q = 1.0 + 5.0 * torch.sigmoid(self.rho)
        mu = torch.exp(-((e - self.c_mu) ** 2) / (2.0 * _soft_sigma(self.s_mu) ** 2)).clamp(1e-6, 1.0)
        nu = (1.0 - mu.pow(q)).clamp(min=0.0).pow(1.0 / q) * torch.exp(
            -((e - self.c_nu) ** 2) / (2.0 * _soft_sigma(self.s_nu) ** 2)
        )
        nu = nu.clamp(0.0, 1.0)
        s = 0.5 * (mu.pow(q) - nu.pow(q) + 1.0)
        return mu, nu, s

    def fused_evidence(self, fa: torch.Tensor, fb: torch.Tensor) -> torch.Tensor:
        ea, eb = self.ev_a(fa), self.ev_b(fb)
        mu_a, nu_a, _ = self._orthopair(ea)
        mu_b, nu_b, _ = self._orthopair(eb)
        w = torch.sigmoid(self.logit_w)
        mu = w * mu_a + (1.0 - w) * mu_b
        nu = w * nu_a + (1.0 - w) * nu_b
        q = 1.0 + 5.0 * torch.sigmoid(self.rho)
        norm = (mu.pow(q) + nu.pow(q)).clamp(min=1e-6)
        scale = torch.clamp(norm, max=1.0) / norm
        mu = (mu * scale.pow(1.0 / q)).clamp(1e-6, 1.0)
        nu = (nu * scale.pow(1.0 / q)).clamp(0.0, 1.0)
        return 0.5 * (mu.pow(q) - nu.pow(q) + 1.0)


class PlainGateConcat(_CrossStreamGate):
    """Control A: cross-stream mix + 2*sigmoid, no q-rung."""

    def fused_evidence(self, fa: torch.Tensor, fb: torch.Tensor) -> torch.Tensor:
        ea, eb = self.ev_a(fa), self.ev_b(fb)
        w = torch.sigmoid(self.logit_w)
        return torch.sigmoid(w * ea + (1.0 - w) * eb)


def copy_ultralytics_meta(dst: nn.Module, src: nn.Module) -> nn.Module:
    """Preserve parse_model metadata so DetectionModel._predict_once keeps working."""
    for key in ("f", "i", "type", "np"):
        if hasattr(src, key):
            setattr(dst, key, getattr(src, key))
    return dst
