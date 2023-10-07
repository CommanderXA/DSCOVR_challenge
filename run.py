import logging
import hydra
from omegaconf import DictConfig
from app import create_app
from nn.dscovry.config import Config

from nn.dscovry.model import DSCOVRYModel


@hydra.main(version_base=None, config_path="nn/conf", config_name="config")
def run(cfg: DictConfig) -> None:
    log = logging.getLogger(__name__)
    Config.setup(cfg, log)

    model = DSCOVRYModel()

    app = create_app(model)
    app.run()


if __name__ == "__main__":
    run()
