import random

import torch

from models.rfdn_baseline.block import RFDB
from models.rfdn_half.block import RFDB_P


def main():
    in_channels = random.randint(4, 10)
    convs = [
        RFDB(in_channels=in_channels)
        for _ in range(random.randint(4, 10))
    ]
    conv = RFDB_P(convs, in_channels=in_channels)
    batch = random.randint(4, 10)
    inputs = [torch.randn(batch, in_channels, 224, 224) for i in range(len(convs))]
    conv_output = torch.cat([c(i) for c, i in zip(convs, inputs)], dim=1)
    print(conv_output.shape)
    convs_output = conv(torch.cat(inputs, dim=1))
    print(convs_output.shape)
    print(torch.sum(torch.abs(conv_output - convs_output)))


if __name__ == "__main__":
    main()
