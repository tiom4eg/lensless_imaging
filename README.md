# Lensless Computational Imaging

Reconstruction of images from mask-based lensless captures ([DigiCam-Mirflickr-MultiMask-10K](https://huggingface.co/datasets/bezzam/DigiCam-Mirflickr-MultiMask-10K)) implemented from scratch using several methods with their comparison. Main method uses unrolled ADMM and modular learned reconstruction.

## Implemented methods

All reconstruction math is written from scratch. A single configurable model `LenslessReconstructor` covers ADMM family, using DRUNet-s as processors.

| Config | Method | Trainable | Notes |
| --- | --- | --- | --- |
| `admm100`         | ADMM-100                      | no  | fixed $\mu$ = 1e-4, $\tau$ = 2e-4, 100 iterations |
| `leadmm20`        | Unrolled LeADMM-20            | yes | per-iteration $\mu_1, \mu_2, \mu_3, \tau$ |
| `modular_prepost` | Modular Le-ADMM-5 (pre+post)  | yes | ~8M processors ($\approx$ 4M each) |
| `modular_pre`     | Modular Le-ADMM-5 (pre only)  | yes | ~8M pre-processor |
| `modular_post`    | Modular Le-ADMM-5 (post only) | yes | ~8M post-processor |
| `fista`           | FISTA                         | no  | non-ADMM accelerated proximal gradient |
| `admm100_sr`      | ADMM-100 + pretrained DRUNet  | no  | general-purpose restoration prior |

Metrics (PSNR, SSIM, MSE, LPIPS-VGG) and the training loss are computed on the ROI on the min-max-normalised reconstruction, since lensless recovery is scale-ambiguous.

## Repository layout

```text
.
├── train.py                # training entry point (Accelerate)
├── inference.py            # reconstruct a dataset, save PNGs, report metrics
├── calculate_metrics.py    # standalone ROI metrics: gt dir vs recon dir
├── benchmark_speed.py      # per-method reconstruction speed
├── demo.ipynb              # Colab demo for a custom dataset URL
├── lensless_helpers/       # provided PSF-simulation helpers
└── src/
    ├── configs/            # Hydra configs (model/*, datasets/*, ...)
    ├── datasets/           # DigiCam + CustomDir datasets, PSF cache
    ├── model/              # operators, ADMM, DRUNet, modular, FISTA, SR
    ├── loss/  metrics/     # ROI loss + PSNR/SSIM/MSE/LPIPS
    ├── trainer/            # trainer + inferencer
    ├── logger/             # Comet/HF logging
    └── utils/              # ROI/IO utils
```

## Installation

Create an environment with Python 3.10+ and install the dependencies:

```bash
git clone -q https://github.com/tiom4eg/lensless_imaging.git <REPO_DIR>
cd <REPO_DIR>
pip install -q -r requirements.txt
```

`requirements.txt` includes the PSF-simulation helpers (`waveprop`, `slm_controller`, `perlin_numpy`) from git.
In case of acquiring wrong `torch`/`torchaudio` installations follow through [PyTorch installation options](https://pytorch.org/get-started/locally/) to pick the desired build.

## Data

The dataset is pulled automatically from the HuggingFace Hub on first use (`bezzam/DigiCam-Mirflickr-MultiMask-10K`): train split for training and test split for evaluation. There are 100 distinct masks, the PSF for each is simulated once and cached under `data/psf_cache/`.

## Checkpoints

Final checkpoints are auto-uploaded to HuggingFace during training, one repo per run named `MHDCSM/lensless-<run_name>_<timestamp>`.
The trained models used in final evaluation:

| Model | HuggingFace repo |
| --- | --- |
| Modular pre+post (best, + fine-tune) | `MHDCSM/lensless-modular_prepost_ft_20260609_190232` |
| Modular pre+post (equal-budget, no fine-tune) | `MHDCSM/lensless-modular_prepost_20260609_003122` |
| Modular post | `MHDCSM/lensless-modular_post_20260610_035523` |
| Modular pre | `MHDCSM/lensless-modular_pre_20260610_100629` |
| Le-ADMM-20 | `MHDCSM/lensless-leadmm20_20260609_193725` |

```python
from huggingface_hub import hf_hub_download
path = hf_hub_download(
    repo_id="MHDCSM/lensless-modular_prepost_ft_20260609_190232",
    filename="model_best.pth",
    local_dir="saved/modular_prepost",
)
```

## Training

```bash
# single GPU
python train.py model=modular_prepost writer.run_name=modular_prepost

# 2x T4 on Kaggle
accelerate launch --multi_gpu --num_processes=2 train.py model=modular_prepost writer.run_name=modular_prepost dataloader.batch_size=8
```

Training monitors `val_PSNR` and saves `saved/<run_name>/model_best.pth`.

## Inference and evaluation

```bash
# reconstruct the DigiCam test split, save PNGs, print ROI metrics
python inference.py model=modular_prepost inferencer.from_pretrained=saved/modular_prepost/model_best.pth inferencer.save_path=modular_prepost_test

# reconstruct an arbitrary CustomDir (lensless/ + masks/ [+ lensed/])
python inference.py datasets=custom_dir model=modular_prepost datasets.test.data_dir=/path/to/custom inferencer.from_pretrained=saved/modular_prepost/model_best.pth

# standalone metrics: gt dir vs recon dir
python calculate_metrics.py --gt_dir /path/to/custom/lensed --pred_dir data/modular_prepost_test/test --device cuda
```

Reconstructions are written as `<id>.png` (matched to the input id).

You can use `benchmark_speed.py` to measure reconstruction speed of different methods:

```bash
python benchmark_speed.py --device cuda --batch_size 1
```

## Demo

[`demo.ipynb`](demo.ipynb) is a Colab notebook for easy inference: it clones the repo, installs dependencies, downloads the checkpoint, takes a Google-Drive `.zip` URL of a `CustomDir` dataset, runs `inference.py`, visualises original vs lensless vs reconstruction and runs `calculate_metrics.py` when ground truth is present.

`make_demo_dir.py` builds a small `CustomDir` sample from the test split for local testing.

## Report

See [REPORT.md](REPORT.md) for the description of each method, qualitative and quantitative comparison, and overall analysis.

## Experiment tracking

Training logs are tracked with [Comet ML project](https://www.comet.com/tiom4eg/lensless-imaging/).

To enable Comet logging locally when training, set:

```bash
export COMET_API_KEY=...      # your Comet api key
export COMET_WORKSPACE=...    # your workspace
export HF_WRITE_TOKEN=...     # optional for checkpoint auto-saving
```

Offline logging: `python train.py writer.mode=offline`.
