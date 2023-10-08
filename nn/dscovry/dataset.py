from datetime import datetime
import torch
from torch.utils.data import Dataset

import pandas as pd

from .config import Config


class DSCOVRDataset(Dataset):
    """DSCOVR Dataset class"""

    def __init__(self, annotation_files: list[str]) -> None:
        data = []
        for i, file in enumerate(annotation_files):
            data.append(
                pd.read_csv(
                    file,
                    delimiter=",",
                    parse_dates=[0],
                    na_values="0",
                    header=None,
                )
            )
        self.data = pd.concat(data, axis=0, ignore_index=True)
        print(len(self.data))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> (torch.Tensor, torch.Tensor):
        row = self.data.iloc[idx]
        X = torch.tensor(
            self.data.iloc[idx, 1:54].fillna(0).values,
            dtype=torch.float32,
            device=Config.device,
        )
        Y = torch.tensor(row[54], dtype=torch.float32, device=Config.device)
        return X, Y

class DSCOVRSimulation(Dataset):
    """DSCOVR data simulation class"""

    def __init__(self, annotation_files: list[str]) -> None:
        data = []
        for i, file in enumerate(annotation_files):
            data.append(
                pd.read_csv(
                    file,
                    delimiter=",",
                    parse_dates=[0],
                    na_values="0",
                    header=None,
                )
            )
        self.data = pd.concat(data, axis=0, ignore_index=True)
        print(len(self.data))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> (datetime, torch.Tensor, torch.Tensor):
        row = self.data.iloc[idx]
        T = self.data.iloc[idx, 0].to_pydatetime().timestamp()
        X = torch.tensor(
            self.data.iloc[idx, 1:54].fillna(0).values,
            dtype=torch.float32,
            device=Config.device,
        )
        Y = torch.tensor(row[54], dtype=torch.float32, device=Config.device)
        return T, X, Y