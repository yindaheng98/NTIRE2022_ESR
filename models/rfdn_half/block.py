import torch
import torch.nn as nn

from models.rfdn_baseline.block import conv_layer, activation, ESA


class RFDBS(nn.Module):
    def __init__(self, in_channels, distillation_rate=0.25):
        super(RFDBS, self).__init__()
        self.dc = self.distilled_channels = in_channels // 2
        self.rc = self.remaining_channels = in_channels
        self.c1_d = conv_layer(in_channels, self.dc, 1)
        self.c1_r = conv_layer(in_channels, self.rc, 3)
        self.c4 = conv_layer(self.remaining_channels, self.dc, 3)
        self.act = activation('lrelu', neg_slope=0.05)
        self.c5 = conv_layer(self.dc * 2, in_channels, 1)
        self.esa = ESA(in_channels, nn.Conv2d)

    def forward(self, input):
        distilled_c1 = self.act(self.c1_d(input))
        r_c1 = (self.c1_r(input))
        r_c1 = self.act(r_c1 + input)

        r_c4 = self.act(self.c4(r_c1))

        out = torch.cat([distilled_c1, r_c4], dim=1)
        out_fused = self.esa(self.c5(out))

        return out_fused

