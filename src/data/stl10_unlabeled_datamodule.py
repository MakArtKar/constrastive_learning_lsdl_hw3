import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import STL10

from src.transforms import DuplicateTransform


class STL10UnlabeledDataModule(LightningDataModule):

    def __init__(
        self,
        data_dir: str = "data/stl10",
        train_val_test_split: Tuple[int, int, int] = (0.8, 0.1, 0.1),
        train_transform = None,
        batch_size: int = 64,
        dataloader_kwargs: dict = {},
        linear_eval_cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        if train_transform:
            self.train_transform = DuplicateTransform(train_transform)
        else:
            self.train_transform = None

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size

    def prepare_data(self) -> None:
        if not os.path.isdir(os.path.join(self.hparams.data_dir, 'stl10_binary')):
            if self.hparams.linear_eval_cfg is not None:
                STL10(self.hparams.data_dir, split='train', download=True)
                STL10(self.hparams.data_dir, split='test', download=True)
            STL10(self.hparams.data_dir, split='unlabeled', download=True)

    def setup(self, stage: Optional[str] = None) -> None:
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

            if self.hparams.linear_eval_cfg is not None:
                if self.hparams.linear_eval_cfg.batch_size % self.trainer.world_size != 0:
                    raise RuntimeError(
                        f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                    )
                self.hparams.linear_eval_cfg.batch_size_per_device = self.hparams.linear_eval_cfg.batch_size // self.trainer.world_size

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            dataset = STL10(self.hparams.data_dir, split='unlabeled', transform=self.train_transform)
            self.data_train, self.data_val, self.data_test = random_split(
                dataset=dataset,
                lengths=self.hparams.train_val_test_split,
                generator=torch.Generator().manual_seed(42),
            )
            if self.hparams.linear_eval_cfg is not None:
                trainset = STL10(self.hparams.data_dir, split='train', transform=self.linear_eval_cfg.train_transform)
                valset = STL10(self.hparams.data_dir, split='train', transform=self.linear_eval_cfg.val_transform)
                self.linear_eval_data_train, _ = random_split(
                    dataset=trainset,
                    lengths=self.hparams.linear_eval_cfg.train_val_split,
                    generator=torch.Generator().manual_seed(42),
                )
                _, self.linear_eval_data_val = random_split(
                    dataset=valset,
                    lengths=self.hparams.linear_eval_cfg.train_val_split,
                    generator=torch.Generator().manual_seed(42),
                )
                self.linear_eval_data_test = STL10(self.hparams.data_dir, split='test', transform=self.linear_eval_cfg.val_transform)

    def train_dataloader(self) -> DataLoader[Any]:
        train_dataloader = DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            shuffle=True,
            **self.hparams.dataloader_kwargs
        )
        if self.hparams.linear_eval_cfg is None:
            return train_dataloader

        linear_eval_train_dataloader = DataLoader(
            dataset=self.linear_eval_data_train,
            batch_size=self.hparams.linear_eval_cfg.batch_size_per_device,
            shuffle=True,
            **self.hparams.linear_eval_cfg.dataloader_kwargs
        )
        return SequentialDataLoader([train_dataloader, linear_eval_train_dataloader])

    def val_dataloader(self) -> DataLoader[Any]:
        val_dataloader = DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            shuffle=False,
            **self.hparams.dataloader_kwargs
        )
        if self.hparams.linear_eval_cfg is None:
            return val_dataloader

        linear_eval_val_dataloader = DataLoader(
            dataset=self.linear_eval_data_val,
            batch_size=self.hparams.linear_eval_cfg.batch_size_per_device,
            shuffle=True,
            **self.hparams.linear_eval_cfg.dataloader_kwargs
        )
        return SequentialDataLoader([val_dataloader, linear_eval_val_dataloader])

    def test_dataloader(self) -> DataLoader[Any]:
        test_dataloader = DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            shuffle=False,
            **self.hparams.dataloader_kwargs
        )
        if self.hparams.linear_eval_cfg is None:
            return test_dataloader

        linear_eval_test_dataloader = DataLoader(
            dataset=self.linear_eval_data_test,
            batch_size=self.hparams.linear_eval_cfg.batch_size_per_device,
            shuffle=True,
            **self.hparams.linear_eval_cfg.dataloader_kwargs
        )
        return SequentialDataLoader([test_dataloader, linear_eval_test_dataloader])

    def teardown(self, stage: Optional[str] = None) -> None:
        pass

    def state_dict(self) -> Dict[Any, Any]:
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        pass


if __name__ == "__main__":
    _ = STL10UnlabeledDataModule()
