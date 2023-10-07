import logging

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import hydra
from omegaconf import DictConfig
from tqdm import tqdm

from dscovry.config import Config
from dscovry.model import DSCOVRYModel
from dscovry.dataset import DSCOVRDataset


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

    # dataset
    dataset = DSCOVRDataset(cfg.data.csv_file)
    train_loader = DataLoader(
        dataset=dataset, batch_size=cfg.hyper.batch_size, shuffle=False
    )

    with tqdm(iter(train_loader)) as tepoch:
        for batch_sample in tepoch:
            # enable mixed precision
            with torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=cfg.hyper.use_amp
            ):
                # data, targets
                x, targets = batch_sample
                targets = targets.unsqueeze(1)

                # forward
                logits = model(x)

                # compute the loss
                loss: torch.Tensor = F.mse_loss(logits, targets)
                print(targets)
                print(logits)
                print(loss.item())

    print()


if __name__ == "__main__":
    # initializations
    # torch.manual_seed(42)
    torch.set_float32_matmul_precision("high")

    # main app
    my_app()
