from augmentations.helper import *


class SimCLRTransform:
    """
    Transform defined in SimCLR
    https://arxiv.org/pdf/2002.05709.pdf
    """

    def __init__(self, size):
        normalize = norm_mean_std(size)
        if size == 224:  # ImageNet
            self.transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(size=size),
                    transforms.RandomHorizontalFlip(),
                    get_color_distortion(s=1.0),
                    transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                    transforms.ToTensor(),
                    normalize
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(size=(size, size)),
                    transforms.RandomHorizontalFlip(),
                    get_color_distortion(s=1.0),
                    transforms.ToTensor(),
                    normalize
                ]
            )

    def __call__(self, x):
        return self.transform(x), self.transform(x)


