import torch
import torch.nn as nn

import models.rfdn_baseline.block as B
from models.rfdn_baseline.block import conv_layer, activation, ESA


class RFDBS(nn.Module):
    def __init__(self, in_channels, distillation_rate=0.25):
        super(RFDBS, self).__init__()
        self.dc = self.distilled_channels = in_channels // 2
        self.rc = self.remaining_channels = in_channels
        self.c1_d = conv_layer(in_channels, self.dc, 1)
        self.c1_r = conv_layer(in_channels, self.rc, 3)
        self.c4 = conv_layer(self.remaining_channels, self.dc, 3)
        self.act = activation('lrelu', neg_slope=0.05)
        self.c5 = conv_layer(self.dc * 2, in_channels, 1)
        self.esa = ESA(in_channels, nn.Conv2d)

    def forward(self, input):
        distilled_c1 = self.act(self.c1_d(input))
        r_c1 = (self.c1_r(input))
        r_c1 = self.act(r_c1 + input)

        r_c4 = self.act(self.c4(r_c1))

        out = torch.cat([distilled_c1, r_c4], dim=1)
        out_fused = self.esa(self.c5(out))

        return out_fused


set0 = False


def conv_p(convs: list[nn.Conv2d], in_channels, out_channels, kernel_size, stride=1, padding=0, **kwargs):
    conv = nn.Conv2d(in_channels * len(convs), out_channels * len(convs),
                     kernel_size, stride=stride, padding=padding, **kwargs)
    with torch.no_grad():
        for i in range(len(convs)):
            for j in range(len(convs)):
                if j == i:
                    conv.weight[j * out_channels:(j + 1) * out_channels, i * in_channels:(i + 1) * in_channels, ...] = \
                        convs[i].weight
                elif set0:
                    conv.weight[j * out_channels:(j + 1) * out_channels, i * in_channels:(i + 1) * in_channels, ...] = \
                        torch.zeros(convs[i].weight.shape)
        for i in range(len(convs)):
            conv.bias[i * out_channels:(i + 1) * out_channels] = convs[i].bias
    return conv


def conv_layer_p(convs: list[nn.Conv2d], in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):
    padding = int((kernel_size - 1) / 2) * dilation
    return conv_p(convs, in_channels, out_channels,
                  kernel_size, stride, padding=padding,
                  bias=True, dilation=dilation, groups=groups)


class ESA_P(ESA):
    def __init__(self, models: list[ESA], n_feats, conv):
        super().__init__(n_feats * len(models), nn.Conv2d)
        f = n_feats // 4
        self.conv1 = conv([m.conv1 for m in models], n_feats, f, kernel_size=1)
        self.conv_f = conv([m.conv_f for m in models], f, f, kernel_size=1)
        self.conv_max = conv([m.conv_max for m in models], f, f, kernel_size=3, padding=1)
        self.conv2 = conv([m.conv2 for m in models], f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv([m.conv3 for m in models], f, f, kernel_size=3, padding=1)
        self.conv3_ = conv([m.conv3_ for m in models], f, f, kernel_size=3, padding=1)
        self.conv4 = conv([m.conv4 for m in models], f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)


def cat_p(n, *args: torch.Tensor):
    return torch.cat([
        torch.cat([
            arg[:, i * arg.shape[1] // n:(i + 1) * arg.shape[1] // n, ...]
            for arg in args
        ], dim=1)
        for i in range(n)
    ], dim=1)


class RFDB_P(B.RFDB):
    def __init__(self, models: list[B.RFDB], in_channels, distillation_rate=0.25):
        super(RFDB_P, self).__init__(in_channels * len(models), distillation_rate=distillation_rate)
        self.dc = self.distilled_channels = in_channels // 2
        self.rc = self.remaining_channels = in_channels
        self.c1_d = conv_layer_p([m.c1_d for m in models], in_channels, self.dc, 1)
        self.c1_r = conv_layer_p([m.c1_r for m in models], in_channels, self.rc, 3)
        self.c2_d = conv_layer_p([m.c2_d for m in models], self.remaining_channels, self.dc, 1)
        self.c2_r = conv_layer_p([m.c2_r for m in models], self.remaining_channels, self.rc, 3)
        self.c3_d = conv_layer_p([m.c3_d for m in models], self.remaining_channels, self.dc, 1)
        self.c3_r = conv_layer_p([m.c3_r for m in models], self.remaining_channels, self.rc, 3)
        self.c4 = conv_layer_p([m.c4 for m in models], self.remaining_channels, self.dc, 3)
        self.act = activation('lrelu', neg_slope=0.05)
        self.c5 = conv_layer_p([m.c5 for m in models], self.dc * 4, in_channels, 1)
        self.esa = ESA_P([m.esa for m in models], in_channels, conv_p)
        self.n = len(models)

    def forward(self, input):
        distilled_c1 = self.act(self.c1_d(input))
        r_c1 = (self.c1_r(input))
        r_c1 = self.act(r_c1 + input)

        distilled_c2 = self.act(self.c2_d(r_c1))
        r_c2 = (self.c2_r(r_c1))
        r_c2 = self.act(r_c2 + r_c1)

        distilled_c3 = self.act(self.c3_d(r_c2))
        r_c3 = (self.c3_r(r_c2))
        r_c3 = self.act(r_c3 + r_c2)

        r_c4 = self.act(self.c4(r_c3))

        out = cat_p(self.n, distilled_c1, distilled_c2, distilled_c3, r_c4)
        out_fused = self.esa(self.c5(out))

        return out_fused


class RFDBS_P(nn.Module):
    def __init__(self, models: list[RFDBS], in_channels, distillation_rate=0.25):
        super(RFDBS_P, self).__init__()
        self.dc = self.distilled_channels = in_channels // 2
        self.rc = self.remaining_channels = in_channels
        self.c1_d = conv_layer_p([m.c1_d for m in models], in_channels, self.dc, 1)
        self.c1_r = conv_layer_p([m.c1_r for m in models], in_channels, self.rc, 3)
        self.c4 = conv_layer_p([m.c4 for m in models], self.remaining_channels, self.dc, 3)
        self.act = activation('lrelu', neg_slope=0.05)
        self.c5 = conv_layer_p([m.c5 for m in models], self.dc * 2, in_channels, 1)
        self.esa = ESA_P([m.esa for m in models], in_channels, conv_p)
        self.n = len(models)

    def forward(self, input):
        distilled_c1 = self.act(self.c1_d(input))
        r_c1 = (self.c1_r(input))
        r_c1 = self.act(r_c1 + input)

        r_c4 = self.act(self.c4(r_c1))

        out = cat_p(self.n, distilled_c1, r_c4)
        out_fused = self.esa(self.c5(out))

        return out_fused
