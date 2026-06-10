# python calculate_metrics.py --gt_dir path/to/lensed --pred_dir path/to/recon

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from src.datasets.digicam import process_lensed
from src.metrics import LPIPSMetric, MSEMetric, PSNRMetric, SSIMMetric


def _load_rgb(path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Failed to read {path}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def _pred_tensor(path):
    arr = _load_rgb(path).astype(np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, -1)
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def main():
    parser = argparse.ArgumentParser(description="ROI reconstruction metrics")
    parser.add_argument("--gt_dir", required=True, help="directory with ground-truth images")
    parser.add_argument("--pred_dir", required=True, help="directory with reconstructions")
    parser.add_argument("--device", default="cpu", help="device for LPIPS/metrics")
    parser.add_argument("--lpips_net", default="vgg")
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    gt_dir, pred_dir = Path(args.gt_dir), Path(args.pred_dir)
    preds = {p.stem: p for p in pred_dir.glob("*.png")}
    if not preds:
        raise RuntimeError(f"No .png reconstructions in {pred_dir}")

    def find_gt(stem):
        for ext in (".png", ".jpg", ".jpeg"):
            cand = gt_dir / f"{stem}{ext}"
            if cand.exists():
                return cand
        return None

    pairs = [(s, p, find_gt(s)) for s, p in sorted(preds.items())]
    matched = [(s, p, g) for s, p, g in pairs if g is not None]
    if not matched:
        raise RuntimeError(f"No id matched between {pred_dir} and {gt_dir}. Ground truth may be absent while metrics require it.")
    print(f"Matched {len(matched)}/{len(pairs)} reconstructions with ground truth.")

    metrics = [
        PSNRMetric(name="PSNR", device=args.device),
        SSIMMetric(name="SSIM", device=args.device),
        MSEMetric(name="MSE", device=args.device),
        LPIPSMetric(name="LPIPS", net=args.lpips_net, device=args.device),
    ]
    totals = {m.name: 0.0 for m in metrics}
    n = 0

    for start in range(0, len(matched), args.batch_size):
        chunk = matched[start:start + args.batch_size]
        preds_t, gts_t = [], []
        for _, pred_path, gt_path in chunk:
            pred = _pred_tensor(pred_path)
            gt = process_lensed(_load_rgb(gt_path), pred)
            preds_t.append(pred)
            gts_t.append(gt)
        recon = torch.stack(preds_t)
        lensed = torch.stack(gts_t)
        for m in metrics:
            totals[m.name] += m(recon=recon, lensed=lensed) * len(chunk)
        n += len(chunk)

    print(f"\nROI metrics over {n} images:")
    for name, total in totals.items():
        print(f"    {name:6s}: {total / n:.4f}")


if __name__ == "__main__":
    main()
