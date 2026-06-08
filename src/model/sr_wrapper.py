# Pretrained restoration network (from deepinv)

import torch
from deepinv.models import DRUNet
import torch.nn as nn

from src.model.admm import ADMM
from src.utils.image_utils import normalize_minmax


class ADMMPlusSR(nn.Module):

    def __init__(self, admm_iters=100, sigma=0.03, pad_factor=2, pretrained="download"):
        super().__init__()
        self.admm = ADMM(n_iters=admm_iters, learnable=False, pad_factor=pad_factor)
        self.sigma = float(sigma)
        self.sr = DRUNet(in_channels=3, out_channels=3, pretrained=pretrained)
        self.sr.eval()
        for p in self.sr.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, lensless, psf, **batch):
        recon = self.admm(lensless, psf)
        recon = normalize_minmax(recon).clamp(0.0, 1.0)
        refined = self.sr(recon, self.sigma)
        return {"recon": refined.clamp(0.0, 1.0)}
