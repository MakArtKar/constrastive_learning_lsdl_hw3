from typing import Any, Dict, Optional, Tuple

import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms

from STL10.stl10_input import *


def download_stl10(data_dir):
    from tqdm import tqdm

    os.makedirs(data_dir)
    UNLABELED_PATH = os.path.join(data_dir, 'unlabeled_img')
    TEST_PATH = os.path.join(data_dir, 'test_img')

    # download data if needed
    download_and_extract()

    # test to check if the whole dataset is read correctly
    images = read_all_images(DATA_PATH)
    print(images.shape)

    labels = read_labels(LABEL_PATH)
    print(labels.shape)

    # save images to disk
    save_images(images, labels)

    unlabeled_images = read_all_images(os.path.join(DATA_DIR, 'stl10_binary/unlabeled_X.bin'))
    if not os.path.exists(UNLABELED_PATH):
        os.makedirs(UNLABELED_PATH, exist_ok=True)
    for i, image in enumerate(tqdm(unlabeled_images)):
        save_image(image, os.path.join(UNLABELED_PATH, str(i)))

    test_images = read_all_images(os.path.join(DATA_DIR, 'stl10_binary/test_X.bin'))
    labels = read_labels(os.path.join(DATA_DIR, 'stl10_binary/test_y.bin'))
    os.makedirs(TEST_PATH, exist_ok=True)
    for label in range(1, 11):
        os.makedirs(os.path.join(TEST_PATH, str(label)), exist_ok=True)
    for i, (image, label) in tqdm(enumerate(zip(test_images, labels)), total=len(labels)):
        save_image(image, os.path.join(TEST_PATH, str(label), str(i)))

    os.rename('img', os.path.join(data_dir, 'img'))
    os.rename('data/stl10_binary.tar.gz', os.path.join(data_dir, 'stl10_binary.tar.gz'))
    os.rename('data/stl10_binary', os.path.join(data_dir, 'stl10_binary'))


class STL10DataModule(LightningDataModule):

    def __init__(
        self,
        data_dir: str = "data/stl10",
        train_val_split: Tuple[int, int, int] = (0.8, 0.2),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # data transformations
        self.transforms = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size

    @property
    def num_classes(self) -> int:
        return 10

    def prepare_data(self) -> None:
        if not os.path.isdir(self.hparams.data_dir):
            print('Start to prepare_data...')
            download_stl10(self.hparams.data_dir)
            print('Finished to prepare_data...')

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
            trainset = ImageFolder(os.path.join(self.hparams.data_dir, 'img'), transform=self.transforms)
            testset = ImageFolder(os.path.join(self.hparams.data_dir, 'test_img'), transform=self.transforms)
            self.data_train, self.data_val = random_split(
                dataset=trainset,
                lengths=self.hparams.train_val_split,
                generator=torch.Generator().manual_seed(42),
            )
            self.data_test = testset

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
