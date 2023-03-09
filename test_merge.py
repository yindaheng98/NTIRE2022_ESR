import random

import torch
import torch.nn as nn

from models.rfdn_half.block import conv_layer
from models.rfdn_half.block import conv_p
from models.rfdn_half.block import conv_layer_p


def main():
    in_channels = random.randint(4, 10)
    out_channels = random.randint(4, 10)
    kernel_size = random.randint(4, 10)
    convs = [
        conv_layer(in_channels=in_channels,
                   out_channels=out_channels,
                   kernel_size=kernel_size)
        for _ in range(random.randint(4, 10))
    ]
    conv = conv_layer_p(convs, in_channels, out_channels, kernel_size)
    batch = random.randint(4, 10)
    inputs = [torch.randn(batch, in_channels, 224, 224) for i in range(len(convs))]
    conv_output = torch.cat([c(i) for c, i in zip(convs, inputs)], dim=1)
    print(conv_output.shape)
    convs_output = conv(torch.cat(inputs, dim=1))
    print(convs_output.shape)
    print(torch.sum(conv_output - convs_output))


if __name__ == "__main__":
    main()
