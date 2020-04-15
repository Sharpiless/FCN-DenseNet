import torch.nn as nn
import torch.nn.functional as F
import pdb


def nothing(x):
    return x


dim_dict = {
    'resnet101': [512, 1024, 2048],
    'resnet152': [512, 1024, 2048],
    'resnet50': [512, 1024, 2048],
    'resnet34': [128, 256, 512],
    'resnet18': [128, 256, 512],
    'densenet121': [256, 512, 1024],
    'densenet161': [384, 1056, 2208],
    'densenet169': [256, 640, 1664],
    'densenet201': [256, 896, 1920]
}


class Deconv(nn.Module):
    def __init__(self, base='vgg'):
        super(Deconv, self).__init__()
        if base == 'vgg':
            self.pred5 = nn.Sequential(
                nn.Conv2d(512, 1, kernel_size=1),
                nn.ReLU()
            )
            self.reduce_channels = [nothing, nothing, nothing]
        else:
            self.pred5 = nn.Sequential(
                nn.Conv2d(512, 1, kernel_size=1),
                nn.ReLU(),
                nn.UpsamplingBilinear2d(scale_factor=2)
            )
            self.reduce_channels = nn.ModuleList([
                nn.Conv2d(in_dim, out_dim, kernel_size=1) for in_dim, out_dim in zip(dim_dict[base], [256, 512, 512])
            ])
        self.pred4 = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=1),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2)
        )
        self.pred3 = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=1),
            nn.UpsamplingBilinear2d(scale_factor=8)
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.fill_(0)

    def forward(self, x):
        x = [r(_x) for r, _x in zip(self.reduce_channels, x)]
        pred5 = self.pred5(x[2])
        pred4 = self.pred4(pred5 + x[1])
        pred3 = self.pred3(pred4 + x[0])
        return pred3
