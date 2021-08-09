import torch.nn as nn
import torchvision
from models.wide_resnet import WideResNet


class LinearEvaluation(nn.Module):
    """
    Linear Evaluation model
    """

    def __init__(self, n_features, n_classes):
        super(LinearEvaluation, self).__init__()
        self.model = nn.Linear(n_features, n_classes)

    def forward(self, x):
        return self.model(x)


def get_encoder(encoder, img_size):
    """
    Get Resnet backbone
    """
    def cifar_resnet(resnet):
        f = []
        for name, module in resnet.named_children():
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                f.append(module)
        return nn.Sequential(*f)

    mobilenetv2 = torchvision.models.mobilenet_v2(pretrained=False)
    shufflenetv2 = torchvision.models.shufflenet_v2_x1_0(pretrained=False)
    resnet18 = torchvision.models.resnet18(pretrained=False)
    resnet34 = torchvision.models.resnet34(pretrained=False)
    resnet50 = torchvision.models.resnet50(pretrained=False)
    return {
        'wrn10': [WideResNet(depth=10, num_classes=1000, widen_factor=2), 1000],
        'wrn16': [WideResNet(depth=16, num_classes=1000, widen_factor=2), 1000],
        'wrn16-4': [WideResNet(depth=16, num_classes=1000, widen_factor=4), 1000],
        'wrn16-8': [WideResNet(depth=16, num_classes=1000, widen_factor=8), 1000],
        'wrn28': [WideResNet(depth=28, num_classes=1000, widen_factor=2), 1000],
        'mobilenetv2': [mobilenetv2, mobilenetv2.classifier[1].out_features],
        'shufflenetv2': [shufflenetv2, shufflenetv2.fc.out_features],
        'resnet18': [resnet18 if img_size >= 100 else cifar_resnet(resnet18), resnet18.fc.in_features],
        'resnet34': [resnet18 if img_size >= 100 else cifar_resnet(resnet34), resnet34.fc.in_features],
        'resnet50': [resnet50 if img_size >= 100 else cifar_resnet(resnet50), resnet50.fc.in_features]
    }[encoder]