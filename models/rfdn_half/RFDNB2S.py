import torch

from .RFDNB2 import RFDNB2, RFDNB2_P
from .block import RFDBS


class RFDNB2S(RFDNB2):
    def __init__(self, in_nc=3, nf=50, **kwargs):
        super().__init__(in_nc=in_nc, nf=nf, **kwargs)
        self.B2S2 = RFDBS(in_channels=nf)

    def forward(self, input):
        return self.tail(self.fea_conv(input), self.B2S2(self.fea_conv2(input)))


class RFDNB2S_P(RFDNB2_P):
    def __init__(self, model: RFDNB2S, in_nc=3, nf=50):
        super().__init__(model, in_nc=in_nc, nf=nf)
        self.B2S2 = model.B2S2

    def forward(self, input):
        out_fea12 = self.fea_conv12(torch.cat([input, input], dim=1))
        out_fea12[:, out_fea12.shape[1] // 2:, ...] = self.B2S2(out_fea12[:, out_fea12.shape[1] // 2:, ...])
        return self.tail(out_fea12)
