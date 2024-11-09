import os
from typing import Any, Dict, Optional, Tuple, Union, Callable, List
from pathlib import Path

import numpy as np
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import CIFAR10

from src.utils import Alb2TorchvisionWrapper


class CIFAR10WithDroppedClasses(CIFAR10):
    def __init__(
        self,
        root: Union[str, Path],
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        classes_to_drop: List[str] = None
    ):
        super().__init__(
            root, train, transform, target_transform, download
        )
        classes_to_drop = classes_to_drop or []
        self.classes_to_drop = classes_to_drop
        classes = [class_ for class_ in self.classes if class_ not in classes_to_drop]
        class_to_idx = {class_: idx for idx, class_ in enumerate(classes)}

        data, targets = [], []
        for item, target in zip(self.data, self.targets):
            if self.classes[target] not in self.classes_to_drop:
                data.append(item)
                targets.append(class_to_idx[self.classes[target]])
        self.data, self.targets = data, targets
        self.classes, self.class_to_idx = classes, class_to_idx


class CIFAR10DataModule(LightningDataModule):

    def __init__(
        self,
        data_dir: str = "data/cifar10",
        train_val_split: Tuple[int, int, int] = (0.9, 0.1),
        train_transform = None,
        val_transform = None,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        classes_to_drop: List[str] = None,
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
        return 10 - len(self.hparams.classes_to_drop)

    def prepare_data(self) -> None:
        if not os.path.isdir(os.path.join(self.hparams.data_dir, 'cifar-10-batches-py')):
            CIFAR10WithDroppedClasses(self.hparams.data_dir, train=True, download=True, classes_to_drop=self.hparams.classes_to_drop)
            CIFAR10WithDroppedClasses(self.hparams.data_dir, train=False, download=True, classes_to_drop=self.hparams.classes_to_drop)

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
            trainset = CIFAR10WithDroppedClasses(self.hparams.data_dir, train=True, transform=self.train_transform, classes_to_drop=self.hparams.classes_to_drop)
            valset = CIFAR10WithDroppedClasses(self.hparams.data_dir, train=True, transform=self.val_transform, classes_to_drop=self.hparams.classes_to_drop)
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
            self.data_test = CIFAR10WithDroppedClasses(self.hparams.data_dir, train=False, transform=self.val_transform, classes_to_drop=self.hparams.classes_to_drop)

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
