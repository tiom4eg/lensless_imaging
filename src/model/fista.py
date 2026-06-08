# Beck & Teboulle, "A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems" (https://www.tau.ac.il/~becka/FISTA.pdf)

import torch
import torch.nn as nn

from src.model import ops


class FISTA(nn.Module):

    def __init__(self, n_iters=100, tau=1e-4, pad_factor=2, fft_friendly=True):
        super().__init__()
        self.n_iters = n_iters
        self.tau = tau
        self.pad_factor = pad_factor
        self.fft_friendly = fft_friendly

    def forward(self, lensless, psf, **batch):
        return {"recon": self.reconstruct(lensless, psf)}

    def reconstruct(self, b, psf):
        sensor_hw = b.shape[-2:]
        full_hw = ops.get_full_shape(sensor_hw, self.pad_factor, self.fft_friendly)
        otf = ops.psf_to_otf(psf, full_hw)
        hth = ops.otf_power(otf)
        lip = hth.amax(dim=(-2, -1), keepdim=True).clamp(min=1e-12)  # (B, C, 1, 1)
        ctb = ops.pad(b, full_hw)

        x = torch.zeros_like(ctb)
        y = x.clone()
        t = 1.0
        thresh = self.tau / lip
        for _ in range(self.n_iters):
            My = ops.conv(y, otf)
            resid = ops.pad(ops.crop(My, sensor_hw), full_hw) - ctb
            grad = ops.conv_adjoint(resid, otf)
            z = y - grad / lip
            x_new = torch.clamp(z - thresh, min=0.0)
            t_new = 0.5 * (1.0 + (1.0 + 4.0 * t * t) ** 0.5)
            y = x_new + ((t - 1.0) / t_new) * (x_new - x)
            x, t = x_new, t_new

        return ops.crop(x, sensor_hw)
