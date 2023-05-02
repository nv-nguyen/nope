from pytorch_lightning.trainer.supporters import CombinedLoader
import logging

import torch


def concat_dataloader(list_dataloaders, mode="max_size_cycle", names=None):
    logging.info(f"Concatenating dataloaders {type(list_dataloaders)}")
    if isinstance(list_dataloaders, dict):
        combined_loader = CombinedLoader(list_dataloaders, mode)
        return combined_loader
    else:
        if names is None:
            names = [f"{i}" for i in range(len(list_dataloaders))]
        list_dataloaders = {
            name: loader for name, loader in zip(names, list_dataloaders)
        }
        combined_loader = CombinedLoader(list_dataloaders, mode=mode)
        return combined_loader
