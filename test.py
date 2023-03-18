import torch
import torch.nn as nn
import math

in_channels = 50
out_channels = 60
conv = nn.Conv2d(in_channels, out_channels, kernel_size=3)

in_channels_scale = 1.5
out_channels_scale = 1.6
in_channels_big = math.ceil(in_channels * in_channels_scale)
out_channels_big = math.ceil(out_channels * out_channels_scale)
conv_big = nn.Conv2d(in_channels_big, out_channels_big, kernel_size=3)

with torch.no_grad():
    conv_big.weight[:, :, ...] = torch.zeros(conv_big.weight.shape)
    conv_big.weight[0:out_channels, 0:in_channels, ...] = conv.weight
    conv_big.bias[:] = torch.zeros(conv_big.bias.shape)
    conv_big.bias[0:out_channels] = conv.bias

batch = 4
sample_input = torch.randn(batch, in_channels, 123, 456)
sample_input_big = torch.randn(batch, in_channels_big, 123, 456)
sample_input_big[0:batch, 0:in_channels, ...] = sample_input
output = conv(sample_input)
output_big = conv_big(sample_input_big)
print(output)
print(output.shape)
print(output_big.shape)
print(torch.mean(torch.abs(output_big[:, 0:out_channels, ...] - output) / torch.abs(output)))
