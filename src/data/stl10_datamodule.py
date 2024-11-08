import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import STL10


class Alb2TorchvisionWrapper:
    def __init__(self, alb_transform):
        self._alb_transform = alb_transform

    def __call__(self, image):
        return self._alb_transform(image=np.array(image))['image']


class STL10DataModule(LightningDataModule):

    def __init__(
        self,
        data_dir: str = "data/stl10",
        train_val_split: Tuple[int, int, int] = (0.8, 0.2),
        train_transform = None,
        val_transform = None,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.train_transform = Alb2TorchvisionWrapper(train_transform) if train_transform else None
        self.val_transform = Alb2TorchvisionWrapper(val_transform) if val_transform else None

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size

    @property
    def num_classes(self) -> int:
        return 10

    def prepare_data(self) -> None:
        if not os.path.isdir(os.path.join(self.hparams.data_dir, 'stl10_binary')):
            STL10(self.hparams.data_dir, split='train', download=True)
            STL10(self.hparams.data_dir, split='test', download=True)
            # STL10(self.hparams.data_dir, split='unlabeled', download=True)

    def setup(self, stage: Optional[str] = None) -> None:
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            trainset = STL10(self.hparams.data_dir, split='train', transform=self.train_transform)
            valset = STL10(self.hparams.data_dir, split='train', transform=self.val_transform)
            self.data_train, _ = random_split(
                dataset=trainset,
                lengths=self.hparams.train_val_split,
                generator=torch.Generator().manual_seed(42),
            )
            _, self.data_val = random_split(
                dataset=valset,
                lengths=self.hparams.train_val_split,
                generator=torch.Generator().manual_seed(42),
            )
            self.data_test = STL10(self.hparams.data_dir, split='test', transform=self.val_transform)

    def train_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        pass

    def state_dict(self) -> Dict[Any, Any]:
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        pass


if __name__ == "__main__":
    _ = STL10DataModule()
