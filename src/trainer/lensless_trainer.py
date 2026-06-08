import io

import matplotlib.pyplot as plt
import torch
from PIL import Image

from src.trainer.base_trainer import BaseTrainer
from src.utils.image_utils import normalize_minmax, roi_crop


class LenslessTrainer(BaseTrainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_grad_norm = self.cfg_trainer.get("max_grad_norm", None)
        self.log_image_n = int(self.cfg_trainer.get("log_image_n", 4))

    def process_batch(self, batch, metrics):
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)

        outputs = self.model(**batch)
        batch.update(outputs)

        loss_out = self.criterion(**batch)
        batch.update(loss_out)

        if self.is_train:
            self.optimizer.zero_grad(set_to_none=True)
            if self.accelerator is not None:
                self.accelerator.backward(batch["loss"])
            else:
                batch["loss"].backward()
            if self.max_grad_norm is not None:
                if self.accelerator is not None:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        metric_funcs = self.metrics["train"] if self.is_train else self.metrics["inference"]
        for loss_name in self.config.writer.loss_names:
            if loss_name in batch:
                value = batch[loss_name]
                metrics.update(loss_name, float(value.detach().item()) if torch.is_tensor(value) else float(value))
        for met in metric_funcs:
            metrics.update(met.name, met(**batch))
        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        if self.writer is None or not self.is_main:
            return
        n = min(self.log_image_n, batch["recon"].shape[0])
        lensless = batch["lensless"].detach().cpu()
        recon = normalize_minmax(roi_crop(batch["recon"].detach())).clamp(0, 1).cpu()
        lensed = roi_crop(batch["lensed"]).detach().cpu() if "lensed" in batch else None
        for i in range(n):
            img = _triptych(
                lensless[i],
                None if lensed is None else lensed[i],
                recon[i],
            )
            self.writer.add_image(f"sample/{i}", img)


def _triptych(lensless, lensed_roi, recon_roi):
    # Render lensless,  ground-truth ROI and reconstruction ROI as single image
    plt.switch_backend("agg")
    cols = 3 if lensed_roi is not None else 2
    fig, axes = plt.subplots(1, cols, figsize=(4 * cols, 3))
    axes[0].imshow(lensless.permute(1, 2, 0).clamp(0, 1).numpy())
    axes[0].set_title("lensless")
    idx = 1
    if lensed_roi is not None:
        axes[idx].imshow(lensed_roi.permute(1, 2, 0).clamp(0, 1).numpy())
        axes[idx].set_title("ground truth")
        idx += 1
    axes[idx].imshow(recon_roi.permute(1, 2, 0).numpy())
    axes[idx].set_title("reconstruction")
    for a in axes:
        a.axis("off")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()
