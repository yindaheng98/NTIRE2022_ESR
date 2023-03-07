import torch
import torch.nn as nn

import models.rfdn_baseline.block as B
from .RFDN import RFDN


def up_conv(in_channels, out_channels, upscale_factor=2, **kwargs):
    return nn.Sequential(
        B.conv_layer(in_channels, out_channels * upscale_factor ** 2, **kwargs),
        nn.PixelShuffle(upscale_factor)
    )


class RFDNM(RFDN):
    def __init__(self, in_nc=3, nf=50, **kwargs):
        super().__init__(in_nc=in_nc, nf=nf, **kwargs)
        self.down_conv1 = B.conv_layer(nf, nf, kernel_size=3, stride=2)
        self.down_conv2 = B.conv_layer(nf, nf, kernel_size=3, stride=2)
        self.down_conv3 = B.conv_layer(nf, nf, kernel_size=3, stride=2)
        self.up_conv1 = up_conv(nf, nf, upscale_factor=2, kernel_size=3)
        self.up_conv2 = up_conv(nf, nf, upscale_factor=2, kernel_size=3)
        self.up_conv3 = up_conv(nf, nf, upscale_factor=2, kernel_size=3)

    def forward(self, input):
        out_fea = self.fea_conv(input)
        out_B1 = self.B1(out_fea)
        out_B2 = self.up_conv1(self.B2(self.down_conv1(out_B1)))
        out_B3 = self.up_conv2(self.B3(self.down_conv2(out_B2)))
        out_B4 = self.up_conv3(self.B4(self.down_conv3(out_B3)))

        out_B = self.c(torch.cat([out_B1, out_B2, out_B3, out_B4], dim=1))
        out_lr = self.LR_conv(out_B) + out_fea

        output = self.upsampler(out_lr)

        return output
