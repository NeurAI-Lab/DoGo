from augmentations.helper import *


class TestTransform:
    """
    Transform defined in SimCLR
    https://arxiv.org/pdf/2002.05709.pdf
    """

    def __init__(self, size):
        normalize = norm_mean_std(size)
        self.transform = transforms.Compose(
            [
                transforms.Resize(size=(size, size)),
                transforms.ToTensor(),
                normalize
            ]
        )

    def __call__(self, x):
        return self.transform(x)


