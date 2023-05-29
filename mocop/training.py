import json
import os
import shutil
from typing import Dict, Union

import hydra
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader, Dataset

from dataset import _split_data
from utils import utils


def build_dataloaders(
    dataset: Dataset,
    batch_size: int,
    splits: Dict[str, str] = None,
    **kwargs,
) -> Dict[str, DataLoader]:
    split_dataset = _split_data(dataset, splits=splits)
    dataloaders = {}
    for split, ds in split_dataset.items():
        dataloaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=split in ["train", "val"],
            drop_last=split in ["train", "val"],
            **kwargs,
        )
    return dataloaders


def train(cfg: Union[Dict, DictConfig]) -> nn.Module:
    if isinstance(cfg, dict):
        cfg = OmegaConf.create(cfg)
    print(OmegaConf.to_yaml(cfg))

    utils.set_seed(cfg.seed)

    model = hydra.utils.instantiate(cfg.model)
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())
    model.set_optimizer(optimizer)

    if hasattr(cfg, "scheduler"):
        scheduler = hydra.utils.instantiate(cfg.scheduler, optimizer=optimizer)
        model.set_scheduler(scheduler, OmegaConf.to_container(cfg.scheduler_config))

    dataloaders = hydra.utils.call(cfg.dataloaders)
    mock_inputs = iter(dataloaders["train"]).next()
    _ = model(**mock_inputs["inputs"])

    trainer = hydra.utils.instantiate(cfg.trainer)
    trainer.fit(model, dataloaders["train"], dataloaders["val"])

    checkpoint_dir = trainer.checkpoint_callback.dirpath
    with open(os.path.join(checkpoint_dir, "config.yml"), "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    best = {
        "best_ckpt_path": trainer.checkpoint_callback.best_model_path,
        "best_metric": trainer.early_stopping_callback.best_score.item(),
    }

    best_str = json.dumps(best, indent=4)
    print(best_str)

    with open(os.path.join(checkpoint_dir, "best_ckpt.json"), "w") as f:
        f.write(best_str)

    shutil.copyfile(
        best["best_ckpt_path"],
        os.path.join(checkpoint_dir, "best_ckpt.ckpt")
    )
    return best
