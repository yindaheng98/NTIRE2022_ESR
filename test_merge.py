import random

import torch
import torch.nn as nn

from models.rfdn_baseline.block import ESA
from models.rfdn_half.block import ESA_P, conv_p


def main():
    in_channels = random.randint(4, 10)
    convs = [
        ESA(n_feats=in_channels, conv=nn.Conv2d)
        for _ in range(random.randint(4, 10))
    ]
    conv = ESA_P(convs, n_feats=in_channels, conv=conv_p)
    batch = random.randint(4, 10)
    inputs = [torch.randn(batch, in_channels, 224, 224) for i in range(len(convs))]
    conv_output = torch.cat([c(i) for c, i in zip(convs, inputs)], dim=1)
    print(conv_output.shape)
    convs_output = conv(torch.cat(inputs, dim=1))
    print(convs_output.shape)
    print(torch.sum(torch.abs(conv_output - convs_output)))


if __name__ == "__main__":
    main()
