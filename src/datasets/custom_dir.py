# CustomDirDataset: reconstruct from an arbitrary directory of captures.
# Expected layout (lensed is optional):
# root/
# ├── lensless/ImageID.png
# ├── masks/ImageID.npy
# └── lensed/ImageID.png

import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.datasets.digicam import process_lensed, process_lensless, psf_from_mask

logger = logging.getLogger(__name__)


def _load_png(path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Failed to read image {path}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


class CustomDirDataset(Dataset):

    def __init__(self, data_dir, limit=None, psf_cache_dir=None):
        self.root = Path(data_dir)
        self.lensless_dir = self.root / "lensless"
        self.masks_dir = self.root / "masks"
        self.lensed_dir = self.root / "lensed"
        if not self.lensless_dir.is_dir():
            raise FileNotFoundError(f"Missing 'lensless' subdir in {self.root}")
        if not self.masks_dir.is_dir():
            raise FileNotFoundError(f"Missing 'masks' subdir in {self.root}")
        self.has_lensed = self.lensed_dir.is_dir()

        ids = sorted(p.stem for p in self.lensless_dir.glob("*.png"))
        if not ids:
            raise RuntimeError(f"No .png files in {self.lensless_dir}")
        if limit is not None:
            ids = ids[:limit]
        self.ids = ids

        self.psf_cache_dir = Path(psf_cache_dir) if psf_cache_dir else self.root / "psf_cache"
        self.psf_cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"CustomDir: {len(self.ids)} samples, lensed={self.has_lensed}")

    def _load_psf(self, image_id):
        cache = self.psf_cache_dir / f"psf_{image_id}.pt"
        if cache.exists():
            return torch.load(cache, weights_only=True)
        mask = np.load(self.masks_dir / f"{image_id}.npy")
        psf = psf_from_mask(mask)
        torch.save(psf, cache)
        return psf

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        image_id = self.ids[idx]
        lensless = process_lensless(_load_png(self.lensless_dir / f"{image_id}.png"))
        psf = self._load_psf(image_id)
        out = {"lensless": lensless, "psf": psf, "id": image_id}
        if self.has_lensed:
            lensed_path = self.lensed_dir / f"{image_id}.png"
            if lensed_path.exists():
                out["lensed"] = process_lensed(_load_png(lensed_path), lensless)
        return out
