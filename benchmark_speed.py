# Usage: python benchmark_speed.py --device cuda --batch_size 1

import argparse
import time

import torch
from hydra import compose, initialize
from hydra.utils import instantiate

DEFAULT_MODELS = ["admm100", "leadmm20", "modular_prepost", "modular_pre", "modular_post", "fista"]


def time_model(model, lensless, psf, n_warmup, n_iters, device):
    model.eval()
    with torch.no_grad():
        for _ in range(n_warmup):
            model(lensless=lensless, psf=psf)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n_iters):
            model(lensless=lensless, psf=psf)
        if device == "cuda":
            torch.cuda.synchronize()
        dt = (time.time() - t0) / n_iters
    return dt


def main():
    parser = argparse.ArgumentParser(description="Reconstruction speed benchmark")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--height", type=int, default=380)
    parser.add_argument("--width", type=int, default=507)
    parser.add_argument("--n_warmup", type=int, default=2)
    parser.add_argument("--n_iters", type=int, default=10)
    args = parser.parse_args()

    device = args.device
    bs, h, w = args.batch_size, args.height, args.width
    lensless = torch.rand(bs, 3, h, w, device=device)
    psf = torch.rand(bs, 3, h, w, device=device)
    psf = psf / psf.flatten(1).norm(dim=1).view(bs, 1, 1, 1)

    models = args.models.split(",")
    print(f"Benchmark on {device}, batch={bs}, size={h}x{w}\n")
    print(f"{'method':18s} {'ms/image':>10s} {'img/s':>8s}")
    with initialize(version_base=None, config_path="src/configs"):
        for name in models:
            cfg = compose(config_name="inference", overrides=[f"model={name}"])
            model = instantiate(cfg.model).to(device)
            dt = time_model(model, lensless, psf, args.n_warmup, args.n_iters, device)
            per_img = dt / bs
            print(f"{name:18s} {per_img * 1000:10.1f} {1.0 / per_img:8.1f}")
            del model
            if device == "cuda":
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
