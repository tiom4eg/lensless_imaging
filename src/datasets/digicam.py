import logging
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download, list_repo_files
from torch.utils.data import Dataset

from lensless_helpers.preprocessor import (
    convert_image_to_float,
    force_rgb,
    get_cropped_lensed,
)
from lensless_helpers.psf import simulate_psf_from_mask
from src.utils.io_utils import ROOT_PATH

logger = logging.getLogger(__name__)

REPO_ID = "bezzam/DigiCam-Mirflickr-MultiMask-10K"


def _resolve_url(filename):
    return f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/{filename}"


def process_lensless(measurement):
    # See preprocessor.get_dataset_object in helpers
    arr = convert_image_to_float(force_rgb(np.array(measurement)))
    t = torch.from_numpy(arr)
    t = torch.rot90(t, k=2, dims=(-3, -2))
    return t.permute(2, 0, 1).contiguous().float()


def process_lensed(measurement, lensless_chw):
    arr = convert_image_to_float(force_rgb(np.array(measurement)))
    canvas_hw3 = np.zeros((lensless_chw.shape[-2], lensless_chw.shape[-1], 3), dtype=np.float32)
    placed = get_cropped_lensed(arr, canvas_hw3)
    return torch.from_numpy(placed).permute(2, 0, 1).contiguous().float()


def psf_from_mask(mask_vals):
    psf = simulate_psf_from_mask(mask_vals)  # (1, H, W, 3)
    return psf.squeeze(0).permute(2, 0, 1).contiguous().float()


class DigiCamDataset(Dataset):

    def __init__(self, split="train", limit=None, psf_cache_dir=None, shuffle=False):
        files = [f for f in list_repo_files(REPO_ID, repo_type="dataset") if f.endswith(".parquet")]
        split_files = sorted(f for f in files if Path(f).name.startswith(f"{split}-"))
        if not split_files:
            raise FileNotFoundError(f"No parquet files for split '{split}' in {REPO_ID}")

        data = load_dataset("parquet", data_files={split: [_resolve_url(f) for f in split_files]})[split]
        if shuffle:
            data = data.shuffle(seed=42)
        if limit is not None:
            data = data.select(range(min(limit, len(data))))
        self.data = data
        self.split = split

        self.psf_cache_dir = Path(psf_cache_dir) if psf_cache_dir else ROOT_PATH / "data" / "psf_cache"
        self.psf_cache_dir.mkdir(parents=True, exist_ok=True)

        labels = sorted(set(int(x) for x in self.data["mask_label"]))
        logger.info(f"DigiCam[{split}]: {len(self.data)} samples, {len(labels)} unique masks")
        self.psf_bank = {label: self._load_psf(label) for label in labels}

    def _load_psf(self, label):
        cache = self.psf_cache_dir / f"psf_{label}.pt"
        if cache.exists():
            return torch.load(cache, weights_only=True)
        mask_path = hf_hub_download(repo_id=REPO_ID, filename=f"masks/mask_{label}.npy", repo_type="dataset")
        psf = psf_from_mask(np.load(mask_path))
        torch.save(psf, cache)
        return psf

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        lensless = process_lensless(item["lensless"])
        lensed = process_lensed(item["lensed"], lensless)
        psf = self.psf_bank[int(item["mask_label"])]
        return {"lensless": lensless, "lensed": lensed, "psf": psf, "id": f"{self.split}_{idx}"}
