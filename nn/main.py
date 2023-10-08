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
from utils import evaluate_accuracy


@hydra.main(version_base=None, config_path="conf", config_name="config")
def my_app(cfg: DictConfig) -> None:
    log = logging.getLogger(__name__)
    Config.setup(cfg, log)

    # model
    model = DSCOVRYModel().to(Config.device)
    # model = torch.compile(model)

    # load the model
    checkpoint = torch.load(f"models/{cfg.model.name}_{cfg.hyper.n_hidden}_2m.pt")
    model.load_state_dict(checkpoint["model"])

    model.eval()
    model.train()

    logging.info(
        f"Model parameters amount: {model.get_parameters_amount():,} (Trained on {checkpoint['epochs']} epochs)"
    )

    # dataset
    dataset = DSCOVRDataset(["./data/data_2023.csv"])
    dataloader = DataLoader(
        dataset=dataset, batch_size=cfg.hyper.batch_size, shuffle=False
    )

    # evaluation
    evaluate(model, dataloader, cfg)


@torch.no_grad
def evaluate(model: DSCOVRYModel, dataloader: DataLoader, cfg):
    epoch_now_accuracy = 0.0
    epoch_future_accuracy = 0.0
    epoch_loss = 0
    with tqdm(iter(dataloader)) as tepoch:
        tepoch.set_description("Evaluating")
        for batch_sample in tepoch:
            # enable mixed precision
            with torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=cfg.hyper.use_amp
            ):
                # data, targets
                x, targets1, targets2 = batch_sample
                targets1 = targets1.unsqueeze(1)
                targets2 = targets2.unsqueeze(1)

                # forward
                logits1, logits2 = model(x)

                # compute the loss
                loss: torch.Tensor = F.mse_loss(logits1, targets1) + F.mse_loss(
                    logits2, targets2
                )
                epoch_now_accuracy += evaluate_accuracy(
                    logits1, targets1, Config.cfg.hyper.tolerance
                )
                epoch_future_accuracy += evaluate_accuracy(
                    logits2, targets2, Config.cfg.hyper.tolerance
                )
                epoch_loss += loss.item()

    accuracy_now = epoch_now_accuracy / len(dataloader)
    accuracy_future = epoch_future_accuracy / len(dataloader)
    loss = epoch_loss / len(dataloader)
    print(f"\n|\n| Accuracy (now): {accuracy_now:.2f}%")
    print(f"| Accuracy (future): {accuracy_future:.2f}%")
    print(f"| Accuracy tolerance: {Config.cfg.hyper.tolerance}")
    print(f"| Loss: {loss}\n|\n")


if __name__ == "__main__":
    # initializations
    # torch.manual_seed(42)
    torch.set_float32_matmul_precision("high")

    # main app
    my_app()
