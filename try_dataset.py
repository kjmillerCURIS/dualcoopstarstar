import os
import sys
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    
    def __init__(self):
        pass

    def __len__(self):
        return 4200

    def __getitem__(self, idx):
        return {'idx' : idx}


if __name__ == '__main__':
    ds = MyDataset()
    dl = DataLoader(ds, batch_size=8, shuffle=True)
    for batch in dl:
        import pdb
        pdb.set_trace()
