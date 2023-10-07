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


@hydra.main(version_base=None, config_path="conf", config_name="config")
def train(cfg: DictConfig) -> None:
    log = logging.getLogger(__name__)
    Config.setup(cfg, log)
    dtype = "bfloat16"
    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype]
    torch.amp.autocast(device_type="cuda", dtype=ptdtype)

    # loading the model
    model = DSCOVRYModel().to(Config.device)
    model = torch.compile(model)

    # optimizers
    scaler = GradScaler(enabled=cfg.hyper.use_amp)
    optimizer = AdamW(
        model.parameters(), lr=cfg.hyper.lr, betas=(cfg.optim.beta1, cfg.optim.beta2)
    )

    if cfg.hyper.pretrained and os.path.exists(
        f"models/{cfg.model.name}_{cfg.hyper.n_hidden}.pt"
    ):
        checkpoint = torch.load(f"models/{cfg.model.name}_{cfg.hyper.n_hidden}.pt")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        Config.set_trained_epochs(checkpoint["epochs"])
    # model.load_state_dict(torch.load(f"models/{cfg.model.name}_{cfg.hyper.n_hidden}.pt"))
    logging.info(
        f"Model parameters amount: {model.get_parameters_amount():,} (Trained on {Config.get_trained_epochs()} epochs)"
    )

    # dataset splits
    dataset = DSCOVRDataset(cfg.data.csv_file)
    train_loader = DataLoader(
        dataset=dataset, batch_size=cfg.hyper.batch_size, shuffle=True
    )

    logging.info(f"Training")
    model.train()
    # train_step(model, train_loader, optimizer, scaler, cfg)


def train_step(
    model: DSCOVRYModel,
    train_loader: DataLoader,
    optimizer: AdamW,
    scaler: GradScaler,
    cfg: DictConfig,
) -> None:
    """Performs actual training"""

    finished_epochs = 0
    for epoch in range(1, cfg.hyper.epochs + 1):
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
                    x, targets = batch_sample

                    # forward
                    logits = model(x)

                    # compute the loss
                    loss: torch.Tensor = F.mse_loss(logits, targets)

                # backprop and optimize
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                # loss.backward()
                # optimizer.step()

                # zero the parameter gradients
                optimizer.zero_grad(set_to_none=True)

        finished_epochs += 1

        # save model
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "epochs": Config.get_trained_epochs() + finished_epochs,
        }
        # save checkpoint
        torch.save(checkpoint, f"models/{cfg.model.name}_{cfg.hyper.block_size}h.pt")

        # monitor losses
        # if epoch % (cfg.hyper.eval_iters) == 0:
        #     logging.info(f"Train Loss: {losses['train']}, Val Loss: {losses['val']}")

        end: float = time.time()
        seconds_elapsed: float = end - start
        logging.info(
            f"Time elapsed: {int((seconds_elapsed)//60)} min {math.ceil(seconds_elapsed % 60)} s"
        )


if __name__ == "__main__":
    # torch.manual_seed(42)
    # torch.multiprocessing.set_start_method("spawn")
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    train()
