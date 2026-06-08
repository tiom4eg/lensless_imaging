import logging
import warnings
from datetime import timedelta

import hydra
from accelerate import Accelerator, DistributedDataParallelKwargs, InitProcessGroupKwargs
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.datasets.data_utils import get_dataloaders
from src.trainer import LenslessTrainer
from src.utils.init_utils import set_random_seed, setup_saving_and_logging

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="lensless")
def main(config):
    set_random_seed(config.trainer.seed)

    ddp_kwargs = DistributedDataParallelKwargs(broadcast_buffers=False, find_unused_parameters=True)
    # raise the NCCL process-group timeout well above the 600s default: transient collective stall on a Kaggle node otherwise trips the watchdog and tears the whole multi-GPU run down
    init_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=3600))
    accelerator = Accelerator(
        mixed_precision=config.trainer.get("mixed_precision", "no"),
        gradient_accumulation_steps=config.trainer.get("grad_accum", 1),
        kwargs_handlers=[ddp_kwargs, init_kwargs],
    )
    device = accelerator.device
    is_main = accelerator.is_main_process

    project_config = OmegaConf.to_container(config, resolve=True)
    if is_main:
        logger = setup_saving_and_logging(config)
        try:
            writer = instantiate(config.writer, logger, project_config)
        except Exception as exc:
            logger.error(f"Online logging init failed ({exc}), falling back to offline.")
            OmegaConf.set_struct(config, False)
            config.writer.mode = "offline"
            OmegaConf.set_struct(config, True)
            writer = instantiate(config.writer, logger, project_config)
    else:
        logger = logging.getLogger("train")
        writer = None
    accelerator.wait_for_everyone()
    # build datasets on the main process first so caches are warm
    with accelerator.main_process_first():
        dataloaders, batch_transforms = get_dataloaders(config, device)

    model = instantiate(config.model)
    loss_function = instantiate(config.loss_function)
    metrics = instantiate(config.metrics)
    if is_main:
        logger.info(model)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Trainable parameters: {n_params / 1e6:.2f}M")

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = instantiate(config.optimizer, params=trainable_params)
    # fill cosine T_max from the total number of optimizer steps if not given
    if "T_max" in config.lr_scheduler and config.lr_scheduler.T_max is None:
        OmegaConf.set_struct(config, False)
        config.lr_scheduler.T_max = int(config.trainer.n_epochs) * int(config.trainer.epoch_len)
        OmegaConf.set_struct(config, True)
    lr_scheduler = instantiate(config.lr_scheduler, optimizer=optimizer)

    hf_uploader = None
    if is_main and config.get("hf_uploader") is not None:
        try:
            hf_uploader = instantiate(config.hf_uploader, logger=logger)
        except Exception as exc:
            logger.error(f"HF uploader init failed: {exc}. Continuing without it.")

    train_loader = dataloaders["train"]
    val_loader = dataloaders.get("val")
    model, loss_function, optimizer, lr_scheduler, train_loader, val_loader = accelerator.prepare(model, loss_function, optimizer, lr_scheduler, train_loader, val_loader)
    dataloaders = {"train": train_loader}
    if val_loader is not None:
        dataloaders["val"] = val_loader

    trainer = LenslessTrainer(
        model=model,
        criterion=loss_function,
        metrics=metrics,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        config=config,
        device=device,
        dataloaders=dataloaders,
        epoch_len=config.trainer.get("epoch_len"),
        logger=logger,
        writer=writer,
        batch_transforms=batch_transforms,
        skip_oom=config.trainer.get("skip_oom", True),
        accelerator=accelerator,
        hf_uploader=hf_uploader,
    )
    trainer.train()


if __name__ == "__main__":
    main()
