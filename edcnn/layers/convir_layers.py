# Layers rimaneggiati del paper di ConvIR
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import NamedTuple


class BasicConv(nn.Module):
    """`BasicConv`: Classe implementante un layer convolutivo (sia in downsampling che in transpose) sandwitched da operazioni
    comuni, ovvero\n
    - Batch Normalization\n
    - Nonlinearity di tipo ReLU
    """
    class Input(NamedTuple):
        """`BasicConv.Input`: Parametri del costruttore di `BasicConv`"""
        in_channel: int
        """`BasicConv.Input.in_channel`: """
        out_channel: int
        """BasicConv.Input.out_channel: """
        kernel_size: int
        """BasicConv.Input.kernel_size: """
        stride: int
        """BasicConv.Input.stride:"""
        bias: bool = True
        """BasicConv.Input.bias:"""
        norm: bool = False
        """BasicConv.Input.norm:"""
        relu: bool = True
        """BasicConv.Input.relu:"""
        transpose: bool = False
        """BasicConv.Input.transpose:"""

    def __init__(self, conf: Input) -> None:
        super(BasicConv, self).__init__()
        bias = True
        if conf.bias and conf.norm:
            bias = False

        padding = conf.kernel_size // 2
        layers = list()
        if conf.transpose:
            padding = conf.kernel_size // 2 - 1
            layers.append(
                nn.ConvTranspose2d(conf.in_channel, conf.out_channel, conf.kernel_size, padding=padding,
                                   stride=conf.stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(conf.in_channel, conf.out_channel, conf.kernel_size, padding=padding, stride=conf.stride,
                          bias=bias))
        if conf.norm:
            layers.append(nn.BatchNorm2d(conf.out_channel))
        if conf.relu:
            layers.append(nn.GELU())
        self.main = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main.forward(x)


class dynamic_filter(nn.Module):
    def __init__(self, inchannels, kernel_size=3, dilation=1, stride=1, group=8):
        super(dynamic_filter, self).__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.group = group
        self.dilation = dilation

        self.conv = nn.Conv2d(inchannels, group * kernel_size ** 2, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(group * kernel_size ** 2)
        self.act = nn.Tanh()

        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        self.lamb_l = nn.Parameter(torch.zeros(inchannels), requires_grad=True)
        self.lamb_h = nn.Parameter(torch.zeros(inchannels), requires_grad=True)
        self.pad = nn.ReflectionPad2d(self.dilation * (kernel_size - 1) // 2)

        self.ap = nn.AdaptiveAvgPool2d((1, 1))
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.inside_all = nn.Parameter(torch.zeros(inchannels, 1, 1), requires_grad=True)

    def forward(self, x):
        identity_input = x
        low_filter = self.ap(x)
        low_filter = self.conv(low_filter)
        low_filter = self.bn(low_filter)

        n, c, h, w = x.shape
        x = F.unfold(self.pad(x), kernel_size=self.kernel_size, dilation=self.dilation).reshape(n, self.group,
                                                                                                c // self.group,
                                                                                                self.kernel_size ** 2,
                                                                                                h * w)

        n, c1, p, q = low_filter.shape
        low_filter = low_filter.reshape(n, c1 // self.kernel_size ** 2, self.kernel_size ** 2, p * q).unsqueeze(2)

        low_filter = self.act(low_filter)

        low_part = torch.sum(x * low_filter, dim=3).reshape(n, c, h, w)

        out_low = low_part * (self.inside_all + 1.) - self.inside_all * self.gap(identity_input)

        out_low = out_low * self.lamb_l[None, :, None, None]

        out_high = (identity_input) * (self.lamb_h[None, :, None, None] + 1.)

        return out_low + out_high


class cubic_attention(nn.Module):
    def __init__(self, dim, group, dilation, kernel) -> None:
        super().__init__()

        self.H_spatial_att = spatial_strip_att(dim, dilation=dilation, group=group, kernel=kernel)
        self.W_spatial_att = spatial_strip_att(dim, dilation=dilation, group=group, kernel=kernel, H=False)
        self.gamma = nn.Parameter(torch.zeros(dim, 1, 1))
        self.beta = nn.Parameter(torch.ones(dim, 1, 1))

    def forward(self, x):
        out = self.H_spatial_att(x)
        out = self.W_spatial_att(out)
        return self.gamma * out + x * self.beta


class spatial_strip_att(nn.Module):
    def __init__(self, dim, kernel=3, dilation=1, group=2, H=True) -> None:
        super().__init__()

        self.k = kernel
        pad = dilation * (kernel - 1) // 2
        self.kernel = (1, kernel) if H else (kernel, 1)
        self.padding = (kernel // 2, 1) if H else (1, kernel // 2)
        self.dilation = dilation
        self.group = group
        self.pad = nn.ReflectionPad2d((pad, pad, 0, 0)) if H else nn.ReflectionPad2d((0, 0, pad, pad))
        self.conv = nn.Conv2d(dim, group * kernel, kernel_size=1, stride=1, bias=False)
        self.ap = nn.AdaptiveAvgPool2d((1, 1))
        self.filter_act = nn.Tanh()
        self.inside_all = nn.Parameter(torch.zeros(dim, 1, 1), requires_grad=True)
        self.lamb_l = nn.Parameter(torch.zeros(dim), requires_grad=True)
        self.lamb_h = nn.Parameter(torch.zeros(dim), requires_grad=True)
        gap_kernel = (None, 1) if H else (1, None)
        self.gap = nn.AdaptiveAvgPool2d(gap_kernel)

    def forward(self, x):
        identity_input = x.clone()
        filter = self.ap(x)
        filter = self.conv(filter)
        n, c, h, w = x.shape
        x = F.unfold(self.pad(x), kernel_size=self.kernel, dilation=self.dilation).reshape(n, self.group,
                                                                                           c // self.group, self.k,
                                                                                           h * w)
        n, c1, p, q = filter.shape
        filter = filter.reshape(n, c1 // self.k, self.k, p * q).unsqueeze(2)
        filter = self.filter_act(filter)
        out = torch.sum(x * filter, dim=3).reshape(n, c, h, w)

        out_low = out * (self.inside_all + 1.) - self.inside_all * self.gap(identity_input)
        out_low = out_low * self.lamb_l[None, :, None, None]
        out_high = identity_input * (self.lamb_h[None, :, None, None] + 1.)

        return out_low + out_high


class MultiShapeKernel(nn.Module):
    class Input(NamedTuple):
        dim: int
        kernel_size: int = 3
        dilation: int = 1
        group: int = 8
    def __init__(self, dim, kernel_size=3, dilation=1, group=8):
        super().__init__()

        self.square_att = dynamic_filter(inchannels=dim, dilation=dilation, group=group, kernel_size=kernel_size)
        self.strip_att = cubic_attention(dim, group=group, dilation=dilation, kernel=kernel_size)

    def forward(self, x):
        x1 = self.strip_att(x)
        x2 = self.square_att(x)

        return x1 + x2


class DeepPoolLayer(nn.Module):
    class Input(NamedTuple):
        k: int
        k_out: int

    def __init__(self, k, k_out):
        super(DeepPoolLayer, self).__init__()
        self.pools_sizes = [8, 4, 2]
        dilation = [7, 9, 11]
        pools, convs, dynas = [], [], []
        for j, i in enumerate(self.pools_sizes):
            pools.append(nn.AvgPool2d(kernel_size=i, stride=i))
            convs.append(nn.Conv2d(k, k, 3, 1, 1, bias=False))
            dynas.append(MultiShapeKernel(dim=k, kernel_size=3, dilation=dilation[j]))
        self.pools = nn.ModuleList(pools)
        self.convs = nn.ModuleList(convs)
        self.dynas = nn.ModuleList(dynas)
        self.relu = nn.GELU()
        self.conv_sum = nn.Conv2d(k, k_out, 3, 1, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_size = x.size()
        resl = x
        y_up = None
        for i in range(len(self.pools_sizes)):
            if i == 0:
                y = self.dynas[i].forward(self.convs[i](self.pools[i](x)))
            else:
                y = self.dynas[i].forward(self.convs[i](self.pools[i](x) + y_up))
            resl = torch.add(resl, F.interpolate(y, x_size[2:], mode='bilinear', align_corners=True))
            if i != len(self.pools_sizes) - 1:
                y_up = F.interpolate(y, scale_factor=2, mode='bilinear', align_corners=True)
        resl = self.relu.forward(resl)
        resl = self.conv_sum.forward(resl)

        return resl


class ResBlock(nn.Module):
    class Input(NamedTuple):
        in_channel: int
        out_channel: int
        filter_: bool = False

    def __init__(self, conf: Input) -> None:
        super(ResBlock, self).__init__()
        self.main = nn.Sequential(
            BasicConv(BasicConv.Input(
                in_channel=conf.in_channel,
                out_channel=conf.out_channel,
                kernel_size=3,
                stride=1
            )),
            DeepPoolLayer(in_channel, out_channel) if filter_ else nn.Identity(),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x):
        return self.main(x) + x

