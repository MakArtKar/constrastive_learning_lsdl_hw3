import albumentations as A


class DuplicateTransform:
    def __init__(self, transform: A.BasicTransform):
        self.transform = transform

    def __call__(self, image, **kwargs):
        return {'image': [self.transform(image=image)['image'], self.transform(image=image)['image']]}
