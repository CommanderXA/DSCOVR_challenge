import logging

import torch

import hydra
from omegaconf import DictConfig

from dscovry.config import Config
from dscovry.model import DSCOVRYModel


@hydra.main(version_base=None, config_path="conf", config_name="config")
def my_app(cfg: DictConfig) -> None:
    log = logging.getLogger(__name__)
    Config.setup(cfg, log)

    # model
    model = DSCOVRYModel().to(Config.device)
    model = torch.compile(model)

    # load the model
    checkpoint = torch.load(f"models/{cfg.model.name}_{cfg.hyper.n_hidden}.pt")
    model.load_state_dict(checkpoint["model"])
    model.eval()

    logging.info(
        f"Model parameters amount: {model.get_parameters_amount():,} (Trained on {checkpoint['epochs']} epochs)"
    )

    # define the logic

    print()


if __name__ == "__main__":
    # initializations
    # torch.manual_seed(42)
    torch.set_float32_matmul_precision("high")

    # main app
    my_app()
