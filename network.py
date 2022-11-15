import torch
import torch.nn as nn
import torch.nn.functional as F


class CoCALC(nn.Module):
    def __init__(self):
        super(CoCALC, self).__init__()

        def conv_bn(in_channel, out_channel, stride):
            return nn.Sequential(
                nn.Conv2d(in_channel, out_channel, 3, stride, 1, bias=False),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(inplace=True))

        def conv_dw(in_channel, out_channel, stride):
            return nn.Sequential(
                nn.Conv2d(in_channel, in_channel, 3, stride,
                          1, groups=in_channel, bias=False),
                nn.BatchNorm2d(in_channel),
                nn.ReLU(inplace=True),

                nn.Conv2d(in_channel, out_channel, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(inplace=True))

        self.model = nn.Sequential(
            conv_bn(1,  32, 2),
            conv_dw(32,  64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            nn.AvgPool2d(7)
        )
        self.fc = nn.Linear(128, 324)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 128)
        x = self.fc(x)
        return x
