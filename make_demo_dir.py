# build a CustomDirDataset directory from the DigiCam test split
# python scripts/make_demo_dir.py --n 10 --out data/custom_demo

import argparse
from pathlib import Path

import numpy as np
from datasets import load_dataset
from huggingface_hub import hf_hub_download

REPO_ID = "bezzam/DigiCam-Mirflickr-MultiMask-10K"


def main():
    p = argparse.ArgumentParser(description="Export a CustomDir sample from DigiCam")
    p.add_argument("--split", default="test")
    p.add_argument("--shard", default="test-00000-of-00002.parquet")
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--out", default="data/custom_demo")
    p.add_argument("--with-lensed", action="store_true", default=True)
    args = p.parse_args()

    url = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/data/{args.shard}"
    ds = load_dataset("parquet", data_files={args.split: url})[args.split]

    out = Path(args.out)
    (out / "lensless").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(parents=True, exist_ok=True)
    if args.with_lensed:
        (out / "lensed").mkdir(parents=True, exist_ok=True)

    for i in range(min(args.n, len(ds))):
        item = ds[i]
        image_id = f"img_{i}"
        item["lensless"].save(out / "lensless" / f"{image_id}.png")
        if args.with_lensed:
            item["lensed"].save(out / "lensed" / f"{image_id}.png")
        mask_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=f"masks/mask_{item['mask_label']}.npy",
            repo_type="dataset",
        )
        np.save(out / "masks" / f"{image_id}.npy", np.load(mask_path))
        print(f"  wrote {image_id} (mask {item['mask_label']})")

    print(f"\nCustomDir written to {out}/  ({min(args.n, len(ds))} samples)")


if __name__ == "__main__":
    main()
