

from __future__ import annotations

import torch
import torch.nn as nn


class ConvNormAct(nn.Module):

    def __init__(self,  nchan: int, bias: bool = False):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv3d(nchan, nchan, kernel_size=5, padding=2, bias=bias),
            nn.InstanceNorm3d(nchan),
            nn.PReLU(),
        )

    def forward(self, x):
        out = self.conv_block(x)
        return out


def _make_nconv(nchan: int, depth: int, bias: bool = False):
    layers = []
    for _ in range(depth):
        layers.append(ConvNormAct(nchan, bias))
    return nn.Sequential(*layers)


class InputTransition(nn.Module):

    def __init__(
        self, in_channels: int, out_channels: int, bias: bool = False
    ):
        super().__init__()

        if out_channels % in_channels != 0:
            raise ValueError(
                f"out channels should be divisible by in_channels. Got in_channels={in_channels}, out_channels={out_channels}."
            )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=5, padding=2, bias=bias),
            nn.InstanceNorm3d(out_channels),
        )
        self.act_function = nn.PReLU()
        

    def forward(self, x):
        out = self.conv_block(x)
        repeat_num = self.out_channels // self.in_channels
        x16 = x.repeat([1, repeat_num, 1, 1, 1])
        out = self.act_function(torch.add(out, x16))
        return out


class DownTransition(nn.Module):

    def __init__(
        self,
        in_channels: int,
        nconvs: int,
        bias: bool = False,
    ):
        super().__init__()

        out_channels = 2 * in_channels
        self.down_op = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=2, stride=2, bias=bias),
            nn.InstanceNorm3d(out_channels),
            nn.PReLU()
        )
        self.ops = _make_nconv(out_channels, nconvs, bias)
        self.act_function = nn.PReLU()

    def forward(self, x):
        down = self.down_op(x)
        out = self.ops(down)
        out = self.act_function(torch.add(out, down))
        return out


class UpTransition(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        nconvs: int,
        bias: bool = False,
    ):
        super().__init__()
        self.up_conv = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels // 2, kernel_size=2, stride=2, bias=bias),
            nn.InstanceNorm3d(out_channels // 2),
            nn.PReLU()
        )
        self.ops = _make_nconv(out_channels, nconvs, bias)
        self.act_function = nn.PReLU()

    def forward(self, x, skipx):
        out = self.up_conv(x)
        xcat = torch.cat((out, skipx), 1)
        out = self.ops(xcat)
        out = self.act_function(torch.add(out, xcat))
        return out


class OutputTransition(nn.Module):

    def __init__(
        self, in_channels: int, out_channels: int,  bias: bool = False
    ):
        super().__init__()
        self.conv_block = _make_nconv(in_channels, 1, bias)
        self.conv2 = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=True)

    def forward(self, x):
        out = self.conv_block(x)
        out = self.conv2(out)
        return out


class VNet(nn.Module):
    """
    V-Net based on `Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation
    <https://arxiv.org/pdf/1606.04797.pdf>`_.

    Args:
        in_channels: number of input channels for the network. Defaults to 1.
            The value should meet the condition that ``16 % in_channels == 0``.
        out_channels: number of output channels for the network. Defaults to 1.
        bias: whether to have a bias term in convolution blocks. Defaults to False.
            According to `Performance Tuning Guide <https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html>`_,
            if a conv layer is directly followed by a batch norm layer, bias should be False.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        bias: bool = False,
    ):
        super().__init__()

        self.in_tr = InputTransition(in_channels, 16, bias=bias)
        self.down_tr32 = DownTransition(16, 1, bias=bias)
        self.down_tr64 = DownTransition(32, 2, bias=bias)
        self.down_tr128 = DownTransition(64, 3, bias=bias)
        self.down_tr256 = DownTransition(128, 2, bias=bias)
        self.up_tr256 = UpTransition(256, 256, 2, bias=bias)
        self.up_tr128 = UpTransition(256, 128, 2, bias=bias)
        self.up_tr64 = UpTransition(128, 64, 1, bias=bias)
        self.up_tr32 = UpTransition(64, 32, 1, bias=bias)
        self.out_tr = OutputTransition(32, out_channels, bias=bias)

    def forward(self, x):
        out16 = self.in_tr(x)
        out32 = self.down_tr32(out16)
        out64 = self.down_tr64(out32)
        out128 = self.down_tr128(out64)
        out256 = self.down_tr256(out128)
        x = self.up_tr256(out256, out128)
        x = self.up_tr128(x, out64)
        x = self.up_tr64(x, out32)
        x = self.up_tr32(x, out16)
        x = self.out_tr(x)
        return x
