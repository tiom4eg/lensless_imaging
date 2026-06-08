# Unified lensless reconstructor: optional pre-processor, (Le-)ADMM core, optional post-processor

import torch.nn as nn

from src.model.admm import ADMM
from src.model.drunet import DRUNet


class LenslessReconstructor(nn.Module):

    def __init__(
        self,
        admm_iters=5,
        admm_learnable=True,
        use_pre=False,
        use_post=False,
        proc_base=32,
        proc_blocks=2,
        mu1=1e-4,
        mu2=1e-4,
        mu3=1e-4,
        tau=2e-4,
        pad_factor=2,
    ):
        super().__init__()
        self.admm = ADMM(
            n_iters=admm_iters,
            learnable=admm_learnable,
            mu1=mu1,
            mu2=mu2,
            mu3=mu3,
            tau=tau,
            pad_factor=pad_factor,
        )
        self.pre = DRUNet(3, 3, base=proc_base, n_blocks=proc_blocks, residual=True) if use_pre else None
        self.post = DRUNet(3, 3, base=proc_base, n_blocks=proc_blocks, residual=True) if use_post else None

    def forward(self, lensless, psf, **batch):
        meas = lensless
        if self.pre is not None:
            meas = self.pre(meas)
        recon = self.admm(meas, psf)
        if self.post is not None:
            recon = self.post(recon)
        return {"recon": recon}
