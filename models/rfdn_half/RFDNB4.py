import torch
import torch.nn as nn

import models.rfdn_baseline.block as B
from .RFDNB2 import RFDNB2


class RFDNB4(RFDNB2):
    def __init__(self, in_nc=3, nf=50, **kwargs):
        super().__init__(in_nc=in_nc, nf=nf, **kwargs)
        self.fea_conv1 = B.conv_layer(in_nc, nf, kernel_size=3)
        self.fea_conv3 = B.conv_layer(in_nc, nf, kernel_size=3)

    def forward(self, input):
        out_fea = self.fea_conv(input)
        out_B1 = self.B1(out_fea)
        out_B2 = self.B2(self.fea_conv1(input))
        out_B3 = self.B3(self.fea_conv2(input))
        out_B4 = self.B4(self.fea_conv3(input))

        out_B = self.c(torch.cat([out_B1, out_B2, out_B3, out_B4], dim=1))
        out_lr = self.LR_conv(out_B) + out_fea

        output = self.upsampler(out_lr)

        return output


class RFDNB4_P(nn.Module):
    def __init__(self, model: RFDNB4, in_nc=3, nf=50, num_modules=4, out_nc=3, upscale=4):
        super().__init__()

        self.fea_conv1234 = B.conv_layer(in_nc, nf * 4, kernel_size=3)

        self.B1234 = B.RFDB(in_channels=nf * 4)
        self.c = B.conv_block(nf * num_modules, nf, kernel_size=1, act_type='lrelu')

        self.LR_conv = B.conv_layer(nf, nf, kernel_size=3)

        upsample_block = B.pixelshuffle_block
        self.upsampler = upsample_block(nf, out_nc, upscale_factor=4)
        self.scale_idx = 0

    def forward(self, input):
        out_fea1234 = self.fea_conv1234(input)
        out_B1234 = self.B1234(out_fea1234)

        out_B = self.c(out_B1234)
        out_lr = self.LR_conv(out_B) + out_fea1234[:, 0:out_fea1234.shape[1] // 4]

        output = self.upsampler(out_lr)

        return output
