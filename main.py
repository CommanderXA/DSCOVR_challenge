import torch
from torch.utils.data import DataLoader

from dscovry.dataset import DSCOVRDataset
from dscovry.model import DSCOVRYModel

dataset = DSCOVRDataset(
    "./data/dsc_fc_summed_spectra_2016_v01.csv", "./data/k_index_2016.csv"
)
dataloader = DataLoader(dataset, batch_size=4)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = DSCOVRYModel().to(device)

for _ in dataloader:
    print()
