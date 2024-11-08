from typing import List

from torch.utils.data import IterableDataset


class SequentialDataLoader(IterableDataset):
    def __init__(self, dataloaders):
        super().__init__()
        self.dataloaders = dataloaders

    def __iter__(self):
        for dataloader_idx, dl in enumerate(self.dataloaders):
            for batch_idx, batch in enumerate(dl):
                yield batch, batch_idx, dataloader_idx

    def __len__(self):
        return sum(len(dl) for dl in self.dataloaders)
