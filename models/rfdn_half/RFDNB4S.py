import torch

from .RFDNB4 import RFDNB4
from .block import RFDBS


class RFDNB4S(RFDNB4):
    def __init__(self, in_nc=3, nf=50, **kwargs):
        super().__init__(in_nc=in_nc, nf=nf, **kwargs)
        self.B2S1 = RFDBS(in_channels=nf)
        self.B2S2 = RFDBS(in_channels=nf)
        self.B2S3 = RFDBS(in_channels=nf)

    def forward(self, input):
        out_fea = self.fea_conv(input)
        out_B1 = self.B1(out_fea)
        out_B2 = self.B2(self.B2S1(self.fea_conv1(input)))
        out_B3 = self.B3(self.B2S2(self.fea_conv2(input)))
        out_B4 = self.B4(self.B2S3(self.fea_conv3(input)))

        out_B = self.c(torch.cat([out_B1, out_B2, out_B3, out_B4], dim=1))
        out_lr = self.LR_conv(out_B) + out_fea

        output = self.upsampler(out_lr)

        return output
