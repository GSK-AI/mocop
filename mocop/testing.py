import json
import os
from typing import Dict, Union

import hydra
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from training import _split_data
from utils import utils


def test(cfg: Union[Dict, DictConfig]) -> nn.Module:
    if isinstance(cfg, dict):
        cfg = OmegaConf.create(cfg)
    print(OmegaConf.to_yaml(cfg))

    model = hydra.utils.instantiate(cfg.test_model)

    dataloaders = hydra.utils.call(cfg.dataloaders)
    mock_inputs = iter(dataloaders["train"]).next()
    _ = model(**mock_inputs["inputs"])
    if "test_model_ckpt" in cfg:
        ckpt = torch.load(cfg.test_model_ckpt)
        model.load_state_dict(ckpt["state_dict"])

    trainer = hydra.utils.instantiate(cfg.trainer)
    test_metrics = trainer.validate(
        model=model, dataloaders=dataloaders["test"], verbose=True
    )
    if "test_model_ckpt" in cfg:
        test_dir = os.path.dirname(cfg.test_model_ckpt)
    else:
        test_dir = os.path.dirname(cfg.test_model._args_[0])
    test_path = os.path.join(test_dir, "test_metrics.json")
    test_str = json.dumps(test_metrics[0], indent=4)
    with open(test_path, "w") as f:
        f.write(test_str)
    return test_metrics
