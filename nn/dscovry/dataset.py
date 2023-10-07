import torch
from torch.utils.data import Dataset

import pandas as pd

from .config import Config


class DSCOVRDataset(Dataset):
    """DSCOVR Dataset class"""

    def __init__(self, annotation_file: str) -> None:
        self.data = pd.read_csv(
            annotation_file,
            delimiter=",",
            parse_dates=[0],
            infer_datetime_format=True,
            na_values="0",
            header=None,
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> (torch.Tensor, torch.Tensor):
        row = self.data.iloc[idx]
        print(row[1:54])
        X = torch.tensor(
            self.data.iloc[idx, 1:54].fillna(0).values,
            dtype=torch.float32,
            device=Config.device,
        )
        print(X)
        Y = torch.tensor(row[54], dtype=torch.float32, device=Config.device)
        return X, Y
