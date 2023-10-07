import torch
from torch.utils.data import DataLoader

from dataset import DSCOVRDataset

dataset = DSCOVRDataset("./data/dsc_fc_summed_spectra_2016_v01.csv")
dataloader = DataLoader(dataset, batch_size=4)

for _ in dataloader:
    print()