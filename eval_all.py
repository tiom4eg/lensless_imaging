import argparse
import json
import os
import sys
import time

# run as `python scripts/eval_all.py`, sys.path[0] is scripts/, so the repo root isn't importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

METHODS = [
    ("admm100", None),
    ("fista", None),
    ("admm100_sr", None),
    ("leadmm20", "leadmm20"),
    ("modular_pre", "modular_pre"),
    ("modular_post", "modular_post"),
    ("modular_prepost", "modular_prepost_ft"),
]


def resolve_checkpoint(hf_user, run_name, token):
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi(token=token)
    prefix = f"lensless-{run_name}_"
    cands = [m for m in api.list_models(author=hf_user, search=prefix) if m.id.split("/")[-1].startswith(prefix)]
    if not cands:
        raise RuntimeError(f"No HF repo for run_name '{run_name}' under {hf_user}")
    cands.sort(key=lambda m: m.id, reverse=True)
    repo = cands[0].id
    print(f"  checkpoint: {repo}/model_best.pth")
    return hf_hub_download(repo_id=repo, filename="model_best.pth", token=token)


def build_inferencer(model_cfg, limit, ckpt_path):
    from src.datasets.data_utils import get_dataloaders
    from src.trainer import LenslessInferencer
    from src.utils.io_utils import ROOT_PATH

    GlobalHydra.instance().clear()
    with initialize(version_base=None, config_path="../src/configs"):
        cfg = compose(
            config_name="inference",
            overrides=[f"model={model_cfg}", f"+datasets.test.limit={limit}"],
        )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    OmegaConf.set_struct(cfg, False)
    cfg.inferencer.save_images = False
    cfg.inferencer.from_pretrained = ckpt_path
    OmegaConf.set_struct(cfg, True)

    dataloaders, batch_transforms = get_dataloaders(cfg, device)
    model = instantiate(cfg.model).to(device)
    metrics = instantiate(cfg.metrics)
    inferencer = LenslessInferencer(
        model=model,
        config=cfg,
        device=device,
        dataloaders=dataloaders,
        save_path=None,
        metrics=metrics,
        batch_transforms=batch_transforms,
        skip_model_load=(ckpt_path is None),
    )
    return inferencer, model, device


def measure_speed(model, device, h=380, w=507, n=5):
    model.eval()
    x = torch.rand(1, 3, h, w, device=device)
    psf = torch.rand(1, 3, h, w, device=device)
    psf = psf / psf.flatten(1).norm(dim=1).view(1, 1, 1, 1)
    with torch.no_grad():
        for _ in range(2):
            model(lensless=x, psf=psf)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n):
            model(lensless=x, psf=psf)
        if device == "cuda":
            torch.cuda.synchronize()
    return (time.time() - t0) / n


def main():
    p = argparse.ArgumentParser(description="Evaluate all methods")
    p.add_argument("--limit", type=int, default=512, help="test samples for metrics")
    p.add_argument("--hf-user", default="MHDCSM")
    p.add_argument("--methods", default="", help="comma list to restrict (default all)")
    p.add_argument("--out", default="eval_results.json")
    args = p.parse_args()

    import os

    token = os.environ.get("HF_TOKEN") or os.environ.get("HF_WRITE_TOKEN")
    wanted = set(args.methods.split(",")) if args.methods else None

    results = {}
    for model_cfg, run_name in METHODS:
        if wanted and model_cfg not in wanted:
            continue
        print(f"\n=== {model_cfg} ===", flush=True)
        try:
            ckpt = resolve_checkpoint(args.hf_user, run_name, token) if run_name else None
            inferencer, model, device = build_inferencer(model_cfg, args.limit, ckpt)
            logs = inferencer.run_inference()
            metrics = logs.get("test", {})
            speed = measure_speed(model, device)
            metrics["sec_per_image"] = speed
            results[model_cfg] = {k: float(v) for k, v in metrics.items()}
            print(f"  {model_cfg}: " + ", ".join(f"{k}={v:.4f}" for k, v in results[model_cfg].items()))
        except Exception as exc:
            print(f"  FAILED {model_cfg}: {exc}")
            results[model_cfg] = {"error": str(exc)}
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print("\n\n========== SUMMARY ==========")
    cols = ["PSNR", "SSIM", "MSE", "LPIPS", "sec_per_image"]
    print(f"{'method':18s}" + "".join(f"{c:>14s}" for c in cols))
    for m, r in results.items():
        if "error" in r:
            print(f"{m:18s}  ERROR: {r['error'][:60]}")
            continue
        print(f"{m:18s}" + "".join(f"{r.get(c, float('nan')):>14.4f}" for c in cols))


if __name__ == "__main__":
    main()
