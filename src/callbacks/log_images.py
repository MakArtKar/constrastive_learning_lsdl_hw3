import lightning as pl
import torchvision
import torch

class LogImagesCallback(pl.Callback):
    def __init__(self, num_images=8, mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261)):
        """
        Callback to log images from train, validation, and test sets.

        Args:
            num_images (int): Number of images to log from each set.
            mean (tuple): Mean used for normalization, to undo it before logging.
            std (tuple): Standard deviation used for normalization, to undo it before logging.
        """
        super().__init__()
        self.num_images = num_images
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)

    def denormalize(self, images):
        """Undo normalization for visualization."""
        return images * self.std + self.mean

    def log_images(self, trainer, pl_module, dataloader, stage):
        if trainer.logger is None:
            return
        # Fetch a batch of images and labels
        images_list, labels = next(iter(dataloader))
        if isinstance(images_list, torch.Tensor):
            images_list = [images_list]
        for i, images in enumerate(images_list):
            images, labels = images[:self.num_images], labels[:self.num_images]

            # Move images to the appropriate device
            images = images.to(pl_module.device)

            # Forward pass to get predictions if model is in eval mode
            pl_module.eval()
            with torch.no_grad():
                preds = pl_module(images)
            pl_module.train()

            # Undo normalization
            images = self.denormalize(images.cpu())

            # Log the images to the logger
            trainer.logger.log_image(key=f"{stage}_images{i + 1}", images=list(images))

    def on_train_epoch_end(self, trainer, pl_module):
        """Log images from the training set at the end of each training epoch."""
        if hasattr(trainer.datamodule, "train_dataloader"):
            self.log_images(trainer, pl_module, trainer.datamodule.train_dataloader(), "train")

    def on_validation_epoch_end(self, trainer, pl_module):
        """Log images from the validation set at the end of each validation epoch."""
        if hasattr(trainer.datamodule, "val_dataloader"):
            self.log_images(trainer, pl_module, trainer.datamodule.val_dataloader(), "val")

    def on_test_epoch_end(self, trainer, pl_module):
        """Log images from the test set at the end of the test epoch (usually only once)."""
        if hasattr(trainer.datamodule, "test_dataloader"):
            self.log_images(trainer, pl_module, trainer.datamodule.test_dataloader(), "test")
