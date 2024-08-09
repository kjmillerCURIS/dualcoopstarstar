import os
import sys
from torch.utils.data import Dataset, DataLoader


class SkibidiDataset(Dataset):

    def __init__(self):
        pass

    def __getitem__(self, idx):
        s = ['skibidi', 'toilet', 'meow', 'mrow', 'kitty', 'cat'][idx]
        return {'impath' : s}

    def __len__(self):
        return 6


if __name__ == '__main__':
    my_dataset = SkibidiDataset()
    my_dataloader = DataLoader(my_dataset, batch_size=2)
    for batch in my_dataloader:
        print(batch)
