import albumentations as A
import numpy as np


class DuplicateTransform:
    def __init__(self, transform: A.BasicTransform):
        self.transform = transform

    def __call__(self, image, **kwargs):
        image = np.array(image, dtype=np.uint8)
        return [self.transform(image=image)['image'], self.transform(image=image)['image']]
