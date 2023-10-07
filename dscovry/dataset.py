import torch
import pandas


class DSCOVRDataset(torch.utils.data.Dataset):
    """DSCOVR Dataset class"""

    def __init__(self, annotation_file: str) -> None:
        self.data = pandas.read_csv(
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
        X = torch.tensor(row[1:54], dtype=torch.float32)
        Y = torch.tensor(row[54], dtype=torch.float32)
        return X, Y
