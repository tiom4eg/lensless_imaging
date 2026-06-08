import torch
import torch.nn as nn

from src.utils.image_utils import normalize_minmax, roi_crop


class LenslessLoss(nn.Module):
    # MSE + LPIPS(VGG) on the ROI
    def __init__(self, mse_weight=1.0, lpips_weight=1.0, lpips_net="vgg"):
        super().__init__()
        self.mse_weight = float(mse_weight)
        self.lpips_weight = float(lpips_weight)
        self.use_lpips = self.lpips_weight > 0
        if self.use_lpips:
            import lpips
            self.lpips = lpips.LPIPS(net=lpips_net, verbose=False)
            for p in self.lpips.parameters():
                p.requires_grad_(False)

    def forward(self, recon, lensed, **batch):
        pred = normalize_minmax(roi_crop(recon)).clamp(0.0, 1.0)
        target = roi_crop(lensed).clamp(0.0, 1.0)

        mse = torch.mean((pred - target) ** 2)
        out = {"mse": mse}
        loss = self.mse_weight * mse
        if self.use_lpips:
            lp = self.lpips(pred * 2 - 1, target * 2 - 1).mean()
            out["lpips_loss"] = lp
            loss = loss + self.lpips_weight * lp
        out["loss"] = loss
        return out
