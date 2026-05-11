"""
This code is adapted from: https://github.com/YaoleiQi/DSCNet
Modified on 2026/1/27:
- Removed repeat-based grid construction (use broadcasting/expand instead)
- Replaced manual gather + trilinear interpolation with torch.nn.functional.grid_sample
- Cached base grid via register_buffer to reduce per-forward overhead & memory spikes
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import cat


def _safe_gn_groups(channels: int) -> int:
    """Pick a reasonable GroupNorm group count that divides channels."""
    # Original code uses out_ch//4, but that can become 0 or not divide channels.
    g = max(1, channels // 4)
    while channels % g != 0 and g > 1:
        g -= 1
    return g


class Conv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.gn = nn.GroupNorm(_safe_gn_groups(out_ch), out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.gn(x)
        x = self.relu(x)
        return x


def _make_base_grid(D: int, H: int, W: int, device, dtype):
    """
    Create base voxel coordinate grids (z,y,x) in index space.
    Returned tensors have shape (1,1,D,H,W) and broadcast to (N,K,D,H,W).
    """
    z = torch.arange(D, device=device, dtype=dtype)
    y = torch.arange(H, device=device, dtype=dtype)
    x = torch.arange(W, device=device, dtype=dtype)
    zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")  # (D,H,W)
    return zz[None, None].contiguous(), yy[None, None].contiguous(), xx[None, None].contiguous()


class DCNGridSample3D(nn.Module):
    """
    Memory-optimized 3D deformable sampling using grid_sample.

    Input:
      x:      (N,C,D,H,W)
      offset: (N,6K,D,H,W)  -> split into offset1/offset2, each (3K)
              then split each into z/y/x of shape (N,K,D,H,W)

    Output:
      morph==0: (N,C,D,H,K*W)   (expand W)
      morph==1: (N,C,D,K*H,W)   (expand H)
      morph==2: (N,C,K*D,H,W)   (expand D)
    """

    def __init__(
        self,
        kernel_size: int,
        extend_scope: float,
        morph: int,
        if_offset: bool,
        align_corners: bool = True,
        padding_mode: str = "border",
    ):
        super().__init__()
        self.K = int(kernel_size)
        self.extend_scope = float(extend_scope)
        self.morph = int(morph)  # 0/1/2
        self.if_offset = bool(if_offset)

        self.align_corners = bool(align_corners)
        self.padding_mode = str(padding_mode)

        # Cache buffers (created lazily on first forward; refreshed if shape/device/dtype changes)
        # self.register_buffer("_base_z", torch.empty(0), persistent=False)
        # self.register_buffer("_base_y", torch.empty(0), persistent=False)
        # self.register_buffer("_base_x", torch.empty(0), persistent=False)
        # self._cached_shape = None  # (D,H,W)

        self._base_z = None
        self._base_y = None
        self._base_x = None
        self._cached_shape = None
        self._cached_device = None
        self._cached_dtype = None

    @staticmethod
    def _centered_cumsum(offset_k: torch.Tensor) -> torch.Tensor:
        """
        Build centered cumulative offsets along K dimension.
        offset_k: (N,K,D,H,W)

        Behavior matches the original loop:
          center = 0
          right side accumulates forward
          left side accumulates backward
        """
        N, K, D, H, W = offset_k.shape
        center = K // 2

        out = offset_k.clone()
        out[:, center] = 0

        # Right side: cumulative sum from center+1 to end
        if center + 1 < K:
            out[:, center + 1 :] = torch.cumsum(offset_k[:, center + 1 :], dim=1)

        # Left side: cumulative sum from center-1 down to 0
        if center > 0:
            left = torch.flip(offset_k[:, :center], dims=[1])  # reverse along K
            left = torch.cumsum(left, dim=1)
            out[:, :center] = torch.flip(left, dims=[1])

        return out

    # def _get_base(self, D: int, H: int, W: int, device, dtype):
    #     need_refresh = (
    #         self._cached_shape != (D, H, W)
    #         or self._base_z.numel() == 0
    #         or self._base_z.device != device
    #         or self._base_z.dtype != dtype
    #     )
    #     if need_refresh:
    #         bz, by, bx = _make_base_grid(D, H, W, device=device, dtype=dtype)
    #         self._base_z = bz
    #         self._base_y = by
    #         self._base_x = bx
    #         self._cached_shape = (D, H, W)
    #     return self._base_z, self._base_y, self._base_x

    def _get_base(self, D: int, H: int, W: int, device, dtype):
        need_refresh = (
            self._cached_shape != (D, H, W)
            or self._base_z is None
            or self._cached_device != device
            or self._cached_dtype != dtype
        )
        if need_refresh:
            bz, by, bx = _make_base_grid(D, H, W, device=device, dtype=dtype)
            self._base_z = bz.contiguous()
            self._base_y = by.contiguous()
            self._base_x = bx.contiguous()
            self._cached_shape = (D, H, W)
            self._cached_device = device
            self._cached_dtype = dtype
        return self._base_z, self._base_y, self._base_x

    def _norm(self, coord: torch.Tensor, size: int) -> torch.Tensor:
        """
        Normalize voxel index coordinate to [-1,1] for grid_sample.
        coord is in [0, size-1] (or may be outside; grid_sample will handle via padding_mode).
        """
        if self.align_corners:
            # When align_corners=True, -1 and 1 map exactly to 0 and size-1.
            return 2.0 * coord / (size - 1) - 1.0
        # When align_corners=False, -1 and 1 map to "outside" by half a pixel.
        return 2.0 * (coord + 0.5) / size - 1.0


    def forward(self, x: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 5, "x must be (N,C,D,H,W)"
        assert offset.dim() == 5, "offset must be (N,6K,D,H,W)"

        N, C, D, H, W = x.shape
        K = self.K
        device, dtype = x.device, x.dtype

        zcoord, ycoord, xcoord = self._get_base(D, H, W, device, dtype)
        z1, y1, x1 = torch.split(offset, K, dim=1)  # each (N,K,D,H,W)

        # Spread along the expanded axis: [-K//2, ..., K//2]
        spread = torch.linspace(-(K // 2), K // 2, steps=K, device=device, dtype=dtype).view(1, K, 1, 1, 1)

        if self.morph == 0:
            # Expand W axis (x) by K: W_out = K*W
            xcoord = xcoord + spread

            if self.if_offset:
                zcoord = zcoord + self._centered_cumsum(z1) * self.extend_scope
                ycoord = ycoord + self._centered_cumsum(y1) * self.extend_scope

            # (N,K,D,H,W) -> (N,D,H,K*W)
            zcoord = zcoord.expand(N, K, D, H, W).permute(0, 2, 3, 1, 4).reshape(N, D, H, K * W)
            ycoord = ycoord.expand(N, K, D, H, W).permute(0, 2, 3, 1, 4).reshape(N, D, H, K * W)
            xcoord = xcoord.expand(N, K, D, H, W).permute(0, 2, 3, 1, 4).reshape(N, D, H, K * W)

        elif self.morph == 1:
            # Expand H axis (y) by K: H_out = K*H
            ycoord = ycoord + spread

            if self.if_offset:
                xcoord = xcoord + self._centered_cumsum(x1) * self.extend_scope
                zcoord = zcoord + self._centered_cumsum(z1) * self.extend_scope

            # (N,K,D,H,W) -> (N,D,K*H,W)
            zcoord = zcoord.expand(N, K, D, H, W).permute(0, 2, 1, 3, 4).reshape(N, D, K * H, W)
            ycoord = ycoord.expand(N, K, D, H, W).permute(0, 2, 1, 3, 4).reshape(N, D, K * H, W)
            xcoord = xcoord.expand(N, K, D, H, W).permute(0, 2, 1, 3, 4).reshape(N, D, K * H, W)

        else:
            # Expand D axis (z) by K: D_out = K*D
            zcoord = zcoord + spread

            if self.if_offset:
                xcoord = xcoord + self._centered_cumsum(x1) * self.extend_scope
                ycoord = ycoord + self._centered_cumsum(y1) * self.extend_scope

            # (N,K,D,H,W) -> (N,K*D,H,W)
            zcoord = zcoord.expand(N, K, D, H, W).reshape(N, K * D, H, W)
            ycoord = ycoord.expand(N, K, D, H, W).reshape(N, K * D, H, W)
            xcoord = xcoord.expand(N, K, D, H, W).reshape(N, K * D, H, W)

        # Normalize sampling coordinates to [-1,1] in the input sampling space
        xcoord = self._norm(xcoord, W)
        ycoord = self._norm(ycoord, H)
        zcoord = self._norm(zcoord, D)

        # grid_sample expects grid[...,0]=x, grid[...,1]=y, grid[...,2]=z
        grid = torch.stack([xcoord, ycoord, zcoord], dim=-1)  # (N,D_out,H_out,W_out,3)
        out = F.grid_sample(
            x,
            grid,
            mode="bilinear",
            padding_mode=self.padding_mode,
            align_corners=self.align_corners,
        )
        return out  # (N,C,D_out,H_out,W_out)


class DCN_Conv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, extend_scope: float, morph: int, if_offset: bool):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.morph = int(morph)
        self.if_offset = bool(if_offset)

        # Predict offsets: 6*K channels (two sets of 3*K, each for z/y/x)
        self.offset_conv = nn.Conv3d(in_ch, 3 * self.kernel_size, 3, padding=1)
        self.bn = nn.BatchNorm3d(3 * self.kernel_size)

        # Deformable sampler with caching + grid_sample
        self.sampler = DCNGridSample3D(
            kernel_size=self.kernel_size,
            extend_scope=extend_scope,
            morph=self.morph,
            if_offset=self.if_offset,
            align_corners=True,
            padding_mode="border",
        )

        # After deformation, convolve along the expanded axis with stride=K to fold back
        if self.morph == 0:
            self.dcn_conv = nn.Conv3d(in_ch, out_ch, kernel_size=(1, 1, self.kernel_size), stride=(1, 1, self.kernel_size), padding=0)
        elif self.morph == 1:
            self.dcn_conv = nn.Conv3d(in_ch, out_ch, kernel_size=(1, self.kernel_size, 1), stride=(1, self.kernel_size, 1), padding=0)
        elif self.morph == 2:
            self.dcn_conv = nn.Conv3d(in_ch, out_ch, kernel_size=(self.kernel_size, 1, 1), stride=(self.kernel_size, 1, 1), padding=0)
        else:
            raise ValueError("morph must be 0, 1, or 2")

        self.gn = nn.GroupNorm(_safe_gn_groups(out_ch), out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        # Offsets are bounded by tanh, then scaled inside sampler by extend_scope
        offset = torch.tanh(self.bn(self.offset_conv(f)))  # (N,6K,D,H,W)
        deformed = self.sampler(f, offset)

        x = self.dcn_conv(deformed)
        x = self.gn(x)
        x = self.relu(x)
        return x


class EncoderConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.gn = nn.GroupNorm(_safe_gn_groups(out_ch), out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.gn(x)
        x = self.relu(x)
        return x


class DecoderConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.gn = nn.GroupNorm(_safe_gn_groups(out_ch), out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.gn(x)
        x = self.relu(x)
        return x


class DSCNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channel_num: int = 16,
        extend_scope: float = 1.0,
        if_offset: bool = True,
    ):
        super().__init__()
        self.extend_scope = float(extend_scope)
        self.if_offset = bool(if_offset)
        self.number = int(base_channel_num)

        # --- Encoder stage 0 ---
        self.conv00 = EncoderConv(in_channels, self.number)
        self.conv0x = DCN_Conv(in_channels, self.number, 3, self.extend_scope, 0, self.if_offset)
        self.conv0y = DCN_Conv(in_channels, self.number, 3, self.extend_scope, 1, self.if_offset)
        self.conv0z = DCN_Conv(in_channels, self.number, 3, self.extend_scope, 2, self.if_offset)
        self.conv1 = EncoderConv(4 * self.number, self.number)

        # --- Encoder stage 1 ---
        self.conv20 = EncoderConv(self.number, 2 * self.number)
        self.conv2x = DCN_Conv(self.number, 2 * self.number, 3, self.extend_scope, 0, self.if_offset)
        self.conv2y = DCN_Conv(self.number, 2 * self.number, 3, self.extend_scope, 1, self.if_offset)
        self.conv2z = DCN_Conv(self.number, 2 * self.number, 3, self.extend_scope, 2, self.if_offset)
        self.conv3 = EncoderConv(8 * self.number, 2 * self.number)

        # --- Encoder stage 2 ---
        self.conv40 = EncoderConv(2 * self.number, 4 * self.number)
        self.conv4x = DCN_Conv(2 * self.number, 4 * self.number, 3, self.extend_scope, 0, self.if_offset)
        self.conv4y = DCN_Conv(2 * self.number, 4 * self.number, 3, self.extend_scope, 1, self.if_offset)
        self.conv4z = DCN_Conv(2 * self.number, 4 * self.number, 3, self.extend_scope, 2, self.if_offset)
        self.conv5 = EncoderConv(16 * self.number, 4 * self.number)

        # --- Encoder stage 3 ---
        self.conv60 = EncoderConv(4 * self.number, 8 * self.number)
        self.conv6x = DCN_Conv(4 * self.number, 8 * self.number, 3, self.extend_scope, 0, self.if_offset)
        self.conv6y = DCN_Conv(4 * self.number, 8 * self.number, 3, self.extend_scope, 1, self.if_offset)
        self.conv6z = DCN_Conv(4 * self.number, 8 * self.number, 3, self.extend_scope, 2, self.if_offset)
        self.conv7 = EncoderConv(32 * self.number, 8 * self.number)

        # --- Decoder stage 4 ---
        self.conv120 = EncoderConv(12 * self.number, 4 * self.number)
        self.conv12x = DCN_Conv(12 * self.number, 4 * self.number, 3, self.extend_scope, 0, self.if_offset)
        self.conv12y = DCN_Conv(12 * self.number, 4 * self.number, 3, self.extend_scope, 1, self.if_offset)
        self.conv12z = DCN_Conv(12 * self.number, 4 * self.number, 3, self.extend_scope, 2, self.if_offset)
        self.conv13 = EncoderConv(16 * self.number, 4 * self.number)

        # --- Decoder stage 5 ---
        self.conv140 = DecoderConv(6 * self.number, 2 * self.number)
        self.conv14x = DCN_Conv(6 * self.number, 2 * self.number, 3, self.extend_scope, 0, self.if_offset)
        self.conv14y = DCN_Conv(6 * self.number, 2 * self.number, 3, self.extend_scope, 1, self.if_offset)
        self.conv14z = DCN_Conv(6 * self.number, 2 * self.number, 3, self.extend_scope, 2, self.if_offset)
        self.conv15 = DecoderConv(8 * self.number, 2 * self.number)

        # --- Decoder stage 6 ---
        self.conv160 = DecoderConv(3 * self.number, self.number)
        self.conv16x = DCN_Conv(3 * self.number, self.number, 3, self.extend_scope, 0, self.if_offset)
        self.conv16y = DCN_Conv(3 * self.number, self.number, 3, self.extend_scope, 1, self.if_offset)
        self.conv16z = DCN_Conv(3 * self.number, self.number, 3, self.extend_scope, 2, self.if_offset)
        self.conv17 = DecoderConv(4 * self.number, self.number)

        self.out_conv = nn.Conv3d(self.number, out_channels, 1)

        self.maxpooling = nn.MaxPool3d(2)
        self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N,C,D,H,W)

        # --- block0 ---
        x_00_0 = self.conv00(x)
        x_0x_0 = self.conv0x(x)
        x_0y_0 = self.conv0y(x)
        x_0z_0 = self.conv0z(x)
        x_0_1 = self.conv1(cat([x_00_0, x_0x_0, x_0y_0, x_0z_0], dim=1))

        # --- block1 ---
        x1 = self.maxpooling(x_0_1)
        x_20_0 = self.conv20(x1)
        x_2x_0 = self.conv2x(x1)
        x_2y_0 = self.conv2y(x1)
        x_2z_0 = self.conv2z(x1)
        x_1_1 = self.conv3(cat([x_20_0, x_2x_0, x_2y_0, x_2z_0], dim=1))

        # --- block2 ---
        x2 = self.maxpooling(x_1_1)
        x_40_0 = self.conv40(x2)
        x_4x_0 = self.conv4x(x2)
        x_4y_0 = self.conv4y(x2)
        x_4z_0 = self.conv4z(x2)
        x_2_1 = self.conv5(cat([x_40_0, x_4x_0, x_4y_0, x_4z_0], dim=1))

        # --- block3 ---
        x3 = self.maxpooling(x_2_1)
        x_60_0 = self.conv60(x3)
        x_6x_0 = self.conv6x(x3)
        x_6y_0 = self.conv6y(x3)
        x_6z_0 = self.conv6z(x3)
        x_3_1 = self.conv7(cat([x_60_0, x_6x_0, x_6y_0, x_6z_0], dim=1))

        # --- block4 ---
        u4 = self.up(x_3_1)
        m4 = cat([u4, x_2_1], dim=1)  # 8n + 4n = 12n
        x_120_2 = self.conv120(m4)
        x_12x_2 = self.conv12x(m4)
        x_12y_2 = self.conv12y(m4)
        x_12z_2 = self.conv12z(m4)
        x_2_3 = self.conv13(cat([x_120_2, x_12x_2, x_12y_2, x_12z_2], dim=1))

        # --- block5 ---
        u5 = self.up(x_2_3)
        m5 = cat([u5, x_1_1], dim=1)  # 4n + 2n = 6n
        x_140_2 = self.conv140(m5)
        x_14x_2 = self.conv14x(m5)
        x_14y_2 = self.conv14y(m5)
        x_14z_2 = self.conv14z(m5)
        x_1_3 = self.conv15(cat([x_140_2, x_14x_2, x_14y_2, x_14z_2], dim=1))

        # --- block6 ---
        u6 = self.up(x_1_3)
        m6 = cat([u6, x_0_1], dim=1)  # 2n + 1n = 3n
        x_160_2 = self.conv160(m6)
        x_16x_2 = self.conv16x(m6)
        x_16y_2 = self.conv16y(m6)
        x_16z_2 = self.conv16z(m6)
        x_0_3 = self.conv17(cat([x_160_2, x_16x_2, x_16y_2, x_16z_2], dim=1))

        out = self.out_conv(x_0_3)
        return out