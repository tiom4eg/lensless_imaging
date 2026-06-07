# Monakhova et al., "Learned reconstructions for practical mask-based lensless imaging" (arXiv:1908.11502)
# + multi-GPU support

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from src.model import ops


def isoftplus(y):
    return torch.log(torch.expm1(torch.as_tensor(y, dtype=torch.float32).clamp(min=1e-10)))


class ADMM(nn.Module):

    def __init__(
        self,
        n_iters=100,
        learnable=False,
        mu1=1e-4,
        mu2=1e-4,
        mu3=1e-4,
        tau=2e-4,
        pad_factor=2,
        fft_friendly=True,
        gradient_checkpointing=True,
    ):
        super().__init__()
        self.n_iters = n_iters
        self.learnable = learnable
        self.pad_factor = pad_factor
        self.fft_friendly = fft_friendly
        # recompute each iteration in the backward pass to bound the memory of unrolled graph
        self.gradient_checkpointing = gradient_checkpointing
        raw_mu1 = torch.full((n_iters,), isoftplus(mu1).item())
        raw_mu2 = torch.full((n_iters,), isoftplus(mu2).item())
        raw_mu3 = torch.full((n_iters,), isoftplus(mu3).item())
        raw_tau = torch.full((n_iters,), isoftplus(tau).item())
        if learnable:
            self.raw_mu1 = nn.Parameter(raw_mu1)
            self.raw_mu2 = nn.Parameter(raw_mu2)
            self.raw_mu3 = nn.Parameter(raw_mu3)
            self.raw_tau = nn.Parameter(raw_tau)
        else:
            self.register_buffer("raw_mu1", raw_mu1, persistent=False)
            self.register_buffer("raw_mu2", raw_mu2, persistent=False)
            self.register_buffer("raw_mu3", raw_mu3, persistent=False)
            self.register_buffer("raw_tau", raw_tau, persistent=False)

    def _step(self, i, x, nu, u, w, lam, eta, rho, otf, hth, psi_freq, crop_mask, ctb, full_hw):
        mu1, mu2, tau, mu3 = F.softplus(self.raw_mu1[i]), F.softplus(self.raw_mu2[i]), F.softplus(self.raw_tau[i]), F.softplus(self.raw_mu3[i])
        Mx = ops.conv(x, otf)
        Psix = ops.grad(x)
        u = ops.soft_threshold(Psix + eta / mu2, tau / mu2)
        nu = (ctb + mu1 * Mx + lam) / (crop_mask + mu1)
        w = torch.clamp(x + rho / mu3, min=0.0)
        rhs = ops.conv_adjoint(mu1 * nu - lam, otf) + ops.grad_adjoint(mu2 * u - eta) + (mu3 * w - rho)
        x = torch.fft.irfft2(torch.fft.rfft2(rhs, dim=(-2, -1)) / (mu1 * hth + mu2 * psi_freq + mu3), s=full_hw, dim=(-2, -1))
        Mx = ops.conv(x, otf)
        Psix = ops.grad(x)
        lam = lam + mu1 * (Mx - nu)
        eta = eta + mu2 * (Psix - u)
        rho = rho + mu3 * (x - w)
        return x, nu, u, w, lam, eta, rho

    def forward(self, b, psf):
        sensor_hw = b.shape[-2:]
        full_hw = ops.get_full_shape(sensor_hw, self.pad_factor, self.fft_friendly)
        otf = ops.psf_to_otf(psf, full_hw)
        hth = ops.otf_power(otf)
        psi_freq = ops.grad_power(full_hw, device=b.device, dtype=b.dtype)
        crop_mask = ops.pad(torch.ones_like(b), full_hw)
        ctb = ops.pad(b, full_hw)

        x = torch.zeros_like(ctb)
        nu = torch.zeros_like(ctb)
        u = torch.zeros_like(ops.grad(x))
        w = torch.zeros_like(ctb)
        lam = torch.zeros_like(ctb)
        eta = torch.zeros_like(u)
        rho = torch.zeros_like(ctb)

        use_ckpt = self.gradient_checkpointing and self.learnable and self.training and torch.is_grad_enabled()
        for i in range(self.n_iters):
            state = (x, nu, u, w, lam, eta, rho)
            consts = (otf, hth, psi_freq, crop_mask, ctb, full_hw)
            if use_ckpt:
                x, nu, u, w, lam, eta, rho = checkpoint(self._step, i, *state, *consts, use_reentrant=False)
            else:
                x, nu, u, w, lam, eta, rho = self._step(i, *state, *consts)

        return ops.crop(x, sensor_hw)
