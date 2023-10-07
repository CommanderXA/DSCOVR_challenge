import torch
import pandas


class DSCOVRDataset(torch.utils.data.Dataset):
    """DSCOVR Dataset class"""

    def __init__(self, annotation_file: str) -> None:
        self.file = pandas.read_csv(
            annotation_file,
            delimiter=",",
            parse_dates=[0],
            infer_datetime_format=True,
            na_values="0",
            header=None,
        )
        self.file.dropna(subset=[1, 2, 3], inplace=True)
        self.threshold = 0.5
        self.fc_range = [a for a in range(4, 54, 1)]
        self.file.dropna(subset=self.fc_range, thresh=self.threshold, inplace=True)
        self.file.fillna(0, inplace=True)

    def __len__(self) -> int:
        return len(self.file)

    def __getitem__(self, idx: int) -> torch.Tensor:
        rows = self.file[3000:3050]
        # rows = self.file[idx * 180 : idx * 180 + 180]
        print(rows)
