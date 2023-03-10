import torch

from .RFDNB4 import RFDNB4, RFDNB4_P
from .block import RFDBS, RFDBS_P


class RFDNB4S(RFDNB4):
    def __init__(self, in_nc=3, nf=50, **kwargs):
        super().__init__(in_nc=in_nc, nf=nf, **kwargs)
        self.B2S1 = RFDBS(in_channels=nf)
        self.B2S2 = RFDBS(in_channels=nf)
        self.B2S3 = RFDBS(in_channels=nf)

    def forward(self, input):
        return self.tail(self.fea_conv(input), self.B2S1(self.fea_conv1(input)),
                         self.B2S2(self.fea_conv2(input)), self.B2S3(self.fea_conv3(input)))


class RFDNB4S_P(RFDNB4_P):
    def __init__(self, model: RFDNB4S, in_nc=3, nf=50):
        super().__init__(model, in_nc=in_nc, nf=nf)
        self.B2S123 = RFDBS_P([model.B2S1, model.B2S2, model.B2S3], in_channels=nf)

    def forward(self, input):
        out_fea1234 = self.fea_conv1234(torch.cat([input, input, input, input], dim=1))
        out_fea1234[:, out_fea1234.shape[1] // 4:, ...] = self.B2S123(out_fea1234[:, out_fea1234.shape[1] // 4:, ...])
        return self.tail(out_fea1234)
