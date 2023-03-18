import torch
import torch.nn as nn


def cat_p(n, *args: torch.Tensor):
    return torch.cat([
        torch.cat([
            arg[:, i * arg.shape[1] // n:(i + 1) * arg.shape[1] // n, ...]
            for arg in args
        ], dim=1)
        for i in range(n)
    ], dim=1)


class Conv2dCat(nn.Conv2d):
    def __init__(self, n: int, in_channels: list[int]):
        for in_channel in in_channels:
            assert in_channel % n == 0
        super().__init__(in_channels=sum(in_channels), out_channels=sum(in_channels), kernel_size=1)
        with torch.no_grad():
            self.weight[...] = torch.zeros(self.weight.shape)
            self.bias[...] = torch.zeros(self.bias.shape)
            for tensor_i in range(len(in_channels)):  # 哪个tensor
                channels_of_tensor = in_channels[tensor_i]  # 这个tensor有多少channel
                channels_to_move = channels_of_tensor // n  # 以多少channel为单位进行移动
                channel_start_i_of_tensor = sum(in_channels[0:tensor_i])  # 这个tensor的channel开始于何处
                for split_i in range(n):  # 第几批移动
                    # 这一批要移动哪些channel
                    src_idx = channel_start_i_of_tensor + split_i * channels_to_move
                    # 这一批channel要移动到哪
                    dst_idx = split_i * sum(in_channels) // n + sum(in_channels[0:tensor_i]) // n
                    self.weight[dst_idx:dst_idx + channels_to_move, src_idx:src_idx + +channels_to_move, 0, 0] = \
                        torch.eye(channels_to_move)
            print(self.weight[:, :, 0, 0])
            # self.weight[:, :, 0, 0] = torch.eye(sum(in_channels))


n = 4
in_channels = [4, 8, 12, 16]
batch_size = 3
in_tensors = [(torch.randn((batch_size, c, 123, 456)) + 10) for c in in_channels]
cat_tensor = cat_p(n, *in_tensors)
cat_conv_tensor = Conv2dCat(n=n, in_channels=in_channels)(torch.cat(in_tensors, dim=1))
print(cat_tensor.shape)
print(cat_conv_tensor.shape)
print(torch.mean(torch.abs(torch.cat(in_tensors, dim=1) - cat_conv_tensor) / torch.abs(cat_conv_tensor)))
print(torch.mean(torch.abs(cat_tensor - cat_conv_tensor) / torch.abs(cat_tensor)))
