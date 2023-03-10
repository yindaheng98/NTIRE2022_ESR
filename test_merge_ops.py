import random

import torch
import torch.nn as nn

from models.rfdn_half.block import RFDBS as model, RFDBS_P as model_p


def main():
    in_channels = random.randint(4, 10)
    convs = [
        model(in_channels=in_channels)
        for _ in range(random.randint(4, 10))
    ]
    conv = model_p(convs, in_channels=in_channels)
    batch = random.randint(4, 10)
    inputs = [torch.randn(batch, in_channels, 224, 224) for i in range(len(convs))]
    conv_output = torch.cat([c(i) for c, i in zip(convs, inputs)], dim=1)
    print(conv_output.shape)
    convs_output = conv(torch.cat(inputs, dim=1))
    print(convs_output.shape)
    print(torch.max(torch.abs(conv_output - convs_output)))


if __name__ == "__main__":
    main()
