import cv2
import numpy as np
import torch
from tqdm.auto import tqdm

from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer
from src.utils.image_utils import normalize_minmax


class LenslessInferencer(BaseTrainer):

    def __init__(
        self,
        model,
        config,
        device,
        dataloaders,
        save_path,
        metrics=None,
        batch_transforms=None,
        skip_model_load=False,
    ):
        assert (
            skip_model_load or config.inferencer.get("from_pretrained") is not None
        ), "Provide a checkpoint via inferencer.from_pretrained or set skip_model_load=True"

        self.config = config
        self.cfg_trainer = self.config.inferencer
        self.accelerator = None
        self.device = device
        self.model = model
        self.batch_transforms = batch_transforms
        self.evaluation_dataloaders = dict(dataloaders)
        self.save_path = save_path
        self.metrics = metrics
        self.save_images = bool(self.cfg_trainer.get("save_images", True))
        if self.metrics is not None:
            self.evaluation_metrics = MetricTracker(*[m.name for m in self.metrics["inference"]], writer=None)
        else:
            self.evaluation_metrics = None

        if not skip_model_load:
            self._from_pretrained(config.inferencer.get("from_pretrained"))

    def run_inference(self):
        out = {}
        for part, loader in self.evaluation_dataloaders.items():
            out[part] = self._inference_part(part, loader)
        return out

    def process_batch(self, batch_idx, batch, metrics, part):
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)
        with torch.no_grad():
            batch.update(self.model(**batch))

        if metrics is not None and "lensed" in batch:
            for met in self.metrics["inference"]:
                metrics.update(met.name, met(**batch))

        if self.save_path is not None and self.save_images:
            self._save_recon(batch, part)
        return batch

    def _save_recon(self, batch, part):
        recon = normalize_minmax(batch["recon"].detach()).clamp(0, 1).cpu()
        ids = batch["id"]
        for i in range(recon.shape[0]):
            img = (recon[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(self.save_path / part / f"{ids[i]}.png"), img)

    def _inference_part(self, part, dataloader):
        self.is_train = False
        self.model.eval()
        if self.evaluation_metrics is not None:
            self.evaluation_metrics.reset()
        if self.save_path is not None:
            (self.save_path / part).mkdir(exist_ok=True, parents=True)
        with torch.no_grad():
            for batch_idx, batch in tqdm(enumerate(dataloader), desc=part, total=len(dataloader)):
                self.process_batch(batch_idx=batch_idx, batch=batch, metrics=self.evaluation_metrics, part=part)
        return self.evaluation_metrics.result() if self.evaluation_metrics is not None else {}

    def move_batch_to_device(self, batch):
        for key in self.cfg_trainer.device_tensors:
            if key in batch:
                batch[key] = batch[key].to(self.device)
        return batch
