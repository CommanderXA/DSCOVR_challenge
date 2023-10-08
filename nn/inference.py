import torch
from torch.utils.data import DataLoader


from nn.dscovry.dataset import DSCOVRSimulation
from nn.dscovry.model import DSCOVRYModel

dataset = DSCOVRSimulation(["./data/data_2023.csv"])
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)


def stream_simulation():
    for sample in dataloader:
        yield sample


def forward(model: DSCOVRYModel, x: torch.Tensor) -> float:
    y = model(x)
    return y
