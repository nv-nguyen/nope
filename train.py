import logging
import os
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
import torch.nn as nn
from src.utils.weight import load_checkpoint
import pytorch_lightning as pl
import multiprocessing
from src.utils.dataloader import concat_dataloader
from torch.utils.data import ConcatDataset

pl.seed_everything(2022)
# set level logging
logging.basicConfig(level=logging.INFO)


@hydra.main(version_base=None, config_path="configs", config_name="train")
def train(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_path = hydra_cfg["runtime"]["output_dir"]
    os.makedirs(cfg.callback.checkpoint.dirpath, exist_ok=True)
    logging.info(
        f"Training script. The outputs of hydra will be stored in: {output_path}"
    )
    logging.info(f"Checkpoints will be stored in: {cfg.callback.checkpoint.dirpath}")

    # Delayed imports to get faster parsing
    from hydra.utils import instantiate

    logging.info("Initializing logger, callbacks and trainer")
    os.environ["WANDB_API_KEY"] = cfg.user.wandb_api_key
    if cfg.machine.dryrun:
        os.environ["WANDB_MODE"] = "offline"
    logging.info(f"Wandb logger initialized at {cfg.save_dir}")

    if cfg.machine.name == "slurm":
        cfg.machine.trainer.devices = int(os.environ["SLURM_GPUS_ON_NODE"])
        cfg.machine.trainer.num_nodes = int(os.environ["SLURM_NNODES"])
    trainer = instantiate(cfg.machine.trainer)
    logging.info(f"Trainer initialized")

    model = instantiate(cfg.model)
    logging.info(f"Model initialized")
    if cfg.model.u_net.pretrained_path is not None and cfg.use_pretrained:
        if "ldm" in cfg.model.u_net._target_:
            load_checkpoint(
                model.u_net,
                cfg.model.u_net.pretrained_path,
                checkpoint_key="state_dict",
                prefix="model.diffusion_model.",
            )
        else:
            load_checkpoint(model.u_net, cfg.model.u_net.pretrained_path)
            logging.info(f"{cfg.model.u_net.pretrained_path} loaded!")
    if "template" in cfg.model.u_net.encoder._target_:
        load_checkpoint(model.u_net.encoder, cfg.model.u_net.encoder.pretrained_path, checkpoint_key="state_dict",
                prefix="",)
    # override the number of workers
    # cfg.machine.num_workers = multiprocessing.cpu_count()
    
    train_dataloaders = {}
    for data_name in cfg.train_data_name:
        config_dataloader = cfg.data[data_name]
        # make sure that we select correct split
        if data_name == "shapeNet":
            config_dataloader.split = "training"
        elif data_name == "tless":
            config_dataloader.seen = True
        # instantiating dataloader
        if data_name == "bop_texture":
            list_datasets = []
            for data_name_i in config_dataloader:
                config_dataloader_i = config_dataloader[data_name_i].dataloader
                # get split
                splits = [
                    split
                    for split in os.listdir(config_dataloader_i.root_dir)
                    if os.path.isdir(os.path.join(config_dataloader_i.root_dir, split))
                ]
                splits = [
                    split
                    for split in splits
                    if split.startswith("train") or split.startswith("val")
                ]
                assert len(splits) == 1, f"Found {splits} train splits for {data_name}"
                split = splits[0]

                config_dataloader_i.split = split
                config_dataloader_i.reset_metaData = False
                config_dataloader_i.use_augmentation = False
                config_dataloader_i.use_random_geometric = False

                dataset_i = instantiate(config_dataloader_i)
                list_datasets.append(dataset_i)
            train_dataset = ConcatDataset(list_datasets)
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=cfg.machine.batch_size,
                num_workers=cfg.machine.num_workers,
                shuffle=True,
            )
        else:
            train_dataloader = DataLoader(
                instantiate(config_dataloader),
                batch_size=cfg.machine.batch_size,
                num_workers=cfg.machine.num_workers,
                shuffle=True,
            )
        logging.info(
            f"Loading train dataloader with {data_name}, size {len(train_dataloader)} done!"
        )
        logging.info("---"*10)
        train_dataloaders[data_name] = train_dataloader
    for dataset_name in train_dataloaders:
        logging.info(
            f"Training sets: {dataset_name}, size {len(train_dataloaders[data_name])}"
        )
    train_dataloaders = concat_dataloader(train_dataloaders)

    val_dataloaders = {}
    for data_name in cfg.test_data_name:
        config_dataloader = cfg.data[data_name]
        if data_name == "shapeNet":
            config_dataloader.split = "unseen_training"
        elif data_name == "tless":
            config_dataloader.seen = False
        val_dataloader = DataLoader(
            instantiate(config_dataloader),
            batch_size=cfg.machine.batch_size,
            num_workers=cfg.machine.num_workers,
            shuffle=False,
        )
        val_dataloaders[data_name] = val_dataloader
        logging.info(
            f"Loading validation dataloader with {data_name}, size {len(val_dataloader)} done!"
        )
    val_dataloaders = concat_dataloader(val_dataloaders)
    logging.info("Fitting the model..")
    trainer.fit(
        model,
        train_dataloaders=train_dataloaders,
        val_dataloaders=val_dataloaders,
        ckpt_path=cfg.model.checkpoint_path
        if cfg.model.checkpoint_path is not None and cfg.use_pretrained
        else None,
    )
    logging.info(f"Fitting done")


if __name__ == "__main__":
    train()
