import logging

import hydra
from omegaconf import DictConfig

import torch

from flask_cors import CORS

from app import create_app

from nn.dscovry.config import Config
from nn.dscovry.model import DSCOVRYModel


@hydra.main(version_base=None, config_path="nn/conf", config_name="config")
def run(cfg: DictConfig) -> None:
    log = logging.getLogger(__name__)
    Config.setup(cfg, log)

    model = DSCOVRYModel().to(Config.device)
    # load the model
    checkpoint = torch.load(
        f"models/{cfg.model.name}_{cfg.hyper.n_hidden}.pt", map_location=Config.device
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()

    app = create_app(model)
    CORS(app)
    app.run()


if __name__ == "__main__":
    run()
