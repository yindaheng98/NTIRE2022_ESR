import random

import torch
import torch.nn as nn


def main():
    in_channels = random.randint(4, 10)
    out_channels = random.randint(4, 10)
    kernel_size = random.randint(4, 10)
    convs = [
        nn.Conv2d(in_channels=in_channels,
                  out_channels=out_channels,
                  kernel_size=kernel_size)
        for _ in range(random.randint(4, 10))
    ]
    conv = nn.Conv2d(in_channels=in_channels * len(convs),
                     out_channels=out_channels * len(convs),
                     kernel_size=kernel_size)
    print(convs[0].bias.shape, conv.bias.shape)
    print(convs[0].weight.shape, conv.weight.shape)
    conv.weight.requires_grad = False
    for i in range(len(convs)):
        convs[i].weight.requires_grad = False
        for j in range(len(convs)):
            conv.weight[j * out_channels:(j + 1) * out_channels, i * in_channels:(i + 1) * in_channels, ...] = \
                convs[i].weight if j == i else torch.zeros(convs[i].weight.shape)
    conv.bias.requires_grad = False
    for i in range(len(convs)):
        convs[i].bias.requires_grad = False
        conv.bias[i * out_channels:(i + 1) * out_channels] = convs[i].bias
    batch = random.randint(4, 10)
    inputs = [torch.randn(batch, in_channels, 224, 224) for i in range(len(convs))]
    conv_output = torch.cat([c(i) for c, i in zip(convs, inputs)], dim=1)
    print(conv_output.shape)
    convs_output = conv(torch.cat(inputs, dim=1))
    print(convs_output.shape)
    print(torch.sum(conv_output - convs_output))


if __name__ == "__main__":
    main()
