import warnings

import hydra
import torch
from hydra.utils import instantiate

from src.datasets.data_utils import get_dataloaders
from src.trainer import LenslessInferencer
from src.utils.init_utils import set_random_seed
from src.utils.io_utils import ROOT_PATH

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="inference")
def main(config):
    """
    Reconstruct a dataset with a trained model and (optionally) score it.
    Saves one <id>.png reconstruction per input and prints ROI metrics when ground truth is available.
    """
    set_random_seed(config.inferencer.seed)

    if config.inferencer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.inferencer.device

    dataloaders, batch_transforms = get_dataloaders(config, device)
    model = instantiate(config.model).to(device)
    metrics = instantiate(config.metrics)

    save_path = ROOT_PATH / "data" / config.inferencer.save_path
    save_path.mkdir(exist_ok=True, parents=True)

    inferencer = LenslessInferencer(
        model=model,
        config=config,
        device=device,
        dataloaders=dataloaders,
        save_path=save_path,
        metrics=metrics,
        batch_transforms=batch_transforms,
        skip_model_load=False,
    )
    logs = inferencer.run_inference()

    for part in logs:
        for key, value in logs[part].items():
            print(f"    {part}_{key:15s}: {value}")


if __name__ == "__main__":
    main()
