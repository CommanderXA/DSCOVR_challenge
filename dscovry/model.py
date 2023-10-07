import torch
import torch.nn as nn
import torch.nn.functional as F


class DSCOVRYModel(nn.Module):
    """DSCOVRY Model for predicting the Planetary K-Index"""

    def __init__(self) -> None:
        super().__init__()
        self.dropout = nn.Dropout(0.3)
        self.input = nn.Linear(in_features=53, out_features=128)
        self.linear1 = nn.Linear(in_features=128, out_features=256)
        self.linear2 = nn.Linear(in_features=256, out_features=256)
        self.lstm = nn.LSTM(
            in_features=256, hidden_size=256, num_layers=2, batch_first=True
        )
        self.linear3 = nn.Linear(in_features=256, out_features=64)
        self.linear4 = nn.Linear(in_features=64, out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.dropout(self.linear1(x)))
        x = F.relu(self.dropout(self.linear2(x)))
        x, _ = self.lstm(x)
        x = F.relu(self.dropout(self.linear3(x)))
        x = F.relu(self.linear4(x))
        return x
