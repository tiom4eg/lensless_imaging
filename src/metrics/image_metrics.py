import torch
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure

from src.metrics.base_metric import BaseMetric
from src.utils.image_utils import normalize_minmax, roi_crop


def _resolve_device(device):
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def prepare_pair(recon, lensed):
    pred = normalize_minmax(roi_crop(recon)).clamp(0.0, 1.0)
    target = roi_crop(lensed).clamp(0.0, 1.0)
    return pred, target


class ImageMetric(BaseMetric):
    def __init__(self, device="cpu", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = _resolve_device(device)

    def __call__(self, recon, lensed, **batch):
        pred, target = prepare_pair(recon.detach(), lensed.detach())
        return self._score(pred.to(self.device), target.to(self.device))

    def _score(self, pred, target):
        raise NotImplementedError


class PSNRMetric(ImageMetric):
    def _score(self, pred, target):
        return float(peak_signal_noise_ratio(pred, target, data_range=1.0, dim=(1, 2, 3)).item())


class SSIMMetric(ImageMetric):
    def _score(self, pred, target):
        return float(structural_similarity_index_measure(pred, target, data_range=1.0).item())


class MSEMetric(ImageMetric):
    def _score(self, pred, target):
        return float(torch.mean((pred - target) ** 2).item())


class LPIPSMetric(ImageMetric):

    def __init__(self, net="vgg", device="cpu", *args, **kwargs):
        super().__init__(device=device, *args, **kwargs)
        import lpips
        self.model = lpips.LPIPS(net=net, verbose=False).to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def _score(self, pred, target):
        # lpips expects inputs in [-1, 1]
        d = self.model(pred * 2 - 1, target * 2 - 1)
        return float(d.mean().item())
