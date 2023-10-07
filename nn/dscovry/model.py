import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config


class DSCOVRYModel(nn.Module):
    """DSCOVRY Model for predicting the Planetary K-Index"""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features=53, out_features=128)
        self.linear1 = nn.Linear(
            in_features=128, out_features=Config.cfg.hyper.n_hidden
        )
        self.linear2 = nn.Linear(
            in_features=Config.cfg.hyper.n_hidden,
            out_features=Config.cfg.hyper.n_hidden,
        )
        self.lstm = nn.LSTM(
            input_size=Config.cfg.hyper.n_hidden,
            hidden_size=Config.cfg.hyper.n_hidden,
            num_layers=Config.cfg.hyper.n_layers,
            batch_first=True,
        )
        self.linear3 = nn.Linear(in_features=Config.cfg.hyper.n_hidden, out_features=64)
        self.linear4 = nn.Linear(in_features=64, out_features=1)
        self.dropout = nn.Dropout(0.3)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(Config.cfg.hyper.n_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(F.relu(self.bn1(self.linear(x))))
        x = self.dropout(F.relu(self.bn2(self.linear1(x))))
        x = self.dropout(F.relu(self.bn2(self.linear2(x))))
        x, _ = self.lstm(x)
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        return x

    def get_parameters_amount(self) -> int:
        return sum(p.numel() for p in self.parameters())
