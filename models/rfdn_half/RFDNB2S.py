import torch

from .RFDNB2 import RFDNB2, RFDNB2_P
from .block import RFDBS, conv_layer_p


class RFDNB2S(RFDNB2):
    def __init__(self, in_nc=3, nf=50, **kwargs):
        super().__init__(in_nc=in_nc, nf=nf, **kwargs)
        self.B2S2 = RFDBS(in_channels=nf)

    def forward(self, input):
        out_fea = self.fea_conv(input)
        out_B1 = self.B1(out_fea)
        out_B2 = self.B2(out_B1)
        out_B3 = self.B3(self.B2S2(self.fea_conv2(input)))
        out_B4 = self.B4(out_B3)

        out_B = self.c(torch.cat([out_B1, out_B2, out_B3, out_B4], dim=1))
        out_lr = self.LR_conv(out_B) + out_fea

        output = self.upsampler(out_lr)

        return output


class RFDNB2S_P(RFDNB2_P):
    def __init__(self, model: RFDNB2S, in_nc=3, nf=50):
        super().__init__(model, in_nc=in_nc, nf=nf)
        self.B2S2 = model.B2S2

    def forward(self, input):
        out_fea12 = self.fea_conv12(torch.cat([input, input], dim=1))
        out_fea12[:, out_fea12.shape[1] // 2:, ...] = self.B2S2(out_fea12[:, out_fea12.shape[1] // 2:, ...])
        out_B13 = self.B13(out_fea12)
        out_B24 = self.B24(out_B13)

        out_B = self.c(torch.cat([
            out_B13[:, 0:out_B13.shape[1] // 2],
            out_B24[:, 0:out_B24.shape[1] // 2],
            out_B13[:, out_B13.shape[1] // 2:],
            out_B24[:, out_B24.shape[1] // 2:],
        ], dim=1))
        out_lr = self.LR_conv(out_B) + out_fea12[:, 0:out_fea12.shape[1] // 2]

        output = self.upsampler(out_lr)

        return output
