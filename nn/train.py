import os
import math
import time
import logging

import torch
from torch.optim import AdamW
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler

import hydra
from tqdm import tqdm
from omegaconf import DictConfig

from dscovry.dataset import DSCOVRDataset
from dscovry.model import DSCOVRYModel
from dscovry.config import Config
from utils import evaluate_accuracy


@hydra.main(version_base=None, config_path="conf", config_name="config")
def train(cfg: DictConfig) -> None:
    log = logging.getLogger(__name__)
    Config.setup(cfg, log)

    if Config.cfg.hyper.use_amp:
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True

    # loading the model
    model = DSCOVRYModel().to(Config.device)

    if Config.cfg.hyper.use_amp:
        dtype = "bfloat16"
        ptdtype = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }[dtype]
        torch.amp.autocast(device_type="cuda", dtype=ptdtype)
        # model = torch.compile(model)

    # optimizers
    scaler = GradScaler(enabled=cfg.hyper.use_amp)
    optimizer = AdamW(
        model.parameters(), lr=cfg.hyper.lr, betas=(cfg.optim.beta1, cfg.optim.beta2)
    )

    if cfg.hyper.pretrained and os.path.exists(
        f"models/{cfg.model.name}_{cfg.hyper.n_hidden}_2m.pt"
    ):
        checkpoint = torch.load(f"models/{cfg.model.name}_{cfg.hyper.n_hidden}_2m.pt")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        Config.set_trained_epochs(checkpoint["epochs"])

    logging.info(
        f"Model parameters amount: {model.get_parameters_amount():,} (Trained on {Config.get_trained_epochs()} epochs)"
    )

    # make pretrained layers not to train
    # for param in model.parameters():
    #     param.requires_grad = False
    # for param in model.b2.parameters():
    #     param.requires_grad = True
    # for param in model.b1.parameters():
    #     param.requires_grad = True

    # dataset
    dataset = DSCOVRDataset(cfg.data.csv_files)
    train_loader = DataLoader(
        dataset=dataset, batch_size=cfg.hyper.batch_size, shuffle=False
    )
    test_dataset = DSCOVRDataset(["./data/data2_2023.csv"])
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=cfg.hyper.batch_size, shuffle=False
    )

    logging.info(f"Training")
    model.train()
    train_step(model, train_loader, test_loader, optimizer, scaler, cfg)


@torch.no_grad
def evaluate(model: DSCOVRYModel, dataloader: DataLoader, cfg):
    model.eval()
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
    model.train()


def train_step(
    model: DSCOVRYModel,
    train_loader: DataLoader,
    test_loader: DataLoader,
    optimizer: AdamW,
    scaler: GradScaler,
    cfg: DictConfig,
) -> None:
    """Performs actual training"""

    finished_epochs = 0
    for epoch in range(1, cfg.hyper.epochs + 1):
        epoch_loss: float = 0.0
        epoch_now_accuracy: float = 0.0
        epoch_future_accuracy: float = 0.0
        start: float = time.time()

        # tqdm bar
        with tqdm(iter(train_loader)) as tepoch:
            tepoch.set_description(f"Epoch: {epoch}")
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
                    loss: torch.Tensor = F.mse_loss(logits1, targets1)
                    loss += F.mse_loss(logits2, targets2)
                    epoch_loss += loss.item()

                # backprop and optimize
                if Config.cfg.hyper.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                # zero the parameter gradients
                optimizer.zero_grad(set_to_none=True)
                epoch_now_accuracy += evaluate_accuracy(
                    logits1, targets1, Config.cfg.hyper.tolerance
                )
                epoch_future_accuracy += evaluate_accuracy(
                    logits2, targets2, Config.cfg.hyper.tolerance
                )

        finished_epochs += 1

        # save model
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "epochs": Config.get_trained_epochs() + finished_epochs,
        }

        # check if models directory does not exist
        if not os.path.exists("models"):
            # create it if it does not exist
            os.mkdir("models")

        # save checkpoint
        torch.save(checkpoint, f"models/{cfg.model.name}_{cfg.hyper.n_hidden}_2m.pt")
        if epoch % 10 == 0:
            torch.save(
                checkpoint, f"models/{cfg.model.name}_{cfg.hyper.n_hidden}_{epoch}_2m.pt"
            )

        # monitor losses
        losses = epoch_loss / len(train_loader)
        accuracy_now = epoch_now_accuracy / len(train_loader)
        accuracy_future = epoch_future_accuracy / len(train_loader)
        logging.info(
            f"Train Loss: {losses}, Train Accuracy (now): {accuracy_now:.2f}%, Train Accuracy (future): {accuracy_future:.2f}%"
        )
        # evaluation
        if epoch % cfg.hyper.eval_iters == 0:
            evaluate(model, test_loader, Config.cfg)

        end: float = time.time()
        seconds_elapsed: float = end - start
        logging.info(
            f"Time elapsed: {int((seconds_elapsed)//60)} min {math.ceil(seconds_elapsed % 60)} s"
        )


if __name__ == "__main__":
    # torch.manual_seed(42)
    # torch.multiprocessing.set_start_method("spawn")
    train()
