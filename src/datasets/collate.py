import torch


def collate_fn(dataset_items):
    batch = {
        "lensless": torch.stack([it["lensless"] for it in dataset_items], dim=0),
        "psf": torch.stack([it["psf"] for it in dataset_items], dim=0),
        "id": [it["id"] for it in dataset_items],
    }
    if all("lensed" in it for it in dataset_items):
        batch["lensed"] = torch.stack([it["lensed"] for it in dataset_items], dim=0)
    return batch
