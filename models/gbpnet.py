"""GBP-Net: Glancing Beyond Patch Network for 3D neuron segmentation.

Two-pathway U-Net (CUN for global context, HRUN for local high-res) fused via
cross-attention transformer and Mamba-based fusion. Adapted from the original
GBP-Net repo for integration with the neuron-trace framework.

Key changes vs. original:
- Self-contained (no dependency on connectomics package)
- Configurable isotropy per-layer (default isotropic for C2-cubes1937)
- Transformer reshape generalised to arbitrary spatial dims (no hardcoded [2,w,w])
- FusionLayer input channels controlled by filters[0]
- Transformer n_voxels passed as constructor arg (computed from input_size)
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def get_activation(mode: str = "relu") -> nn.Module:
    return {
        "relu": nn.ReLU(inplace=True),
        "leaky_relu": nn.LeakyReLU(negative_slope=0.2, inplace=True),
        "elu": nn.ELU(alpha=1.0, inplace=True),
        "gelu": nn.GELU(),
        "silu": nn.SiLU(inplace=True),
        "none": nn.Identity(),
    }[mode]


def get_norm_3d(norm: str, channels: int, momentum: float = 0.1) -> nn.Module:
    if norm == "bn":
        return nn.BatchNorm3d(channels, momentum=momentum)
    if norm == "sync_bn":
        return nn.SyncBatchNorm(channels, momentum=momentum)
    if norm == "in":
        return nn.InstanceNorm3d(channels, momentum=momentum)
    if norm == "gn":
        num_groups = min(8, channels)
        while channels % num_groups != 0:
            num_groups //= 2
        return nn.GroupNorm(num_groups, channels)
    if norm == "none":
        return nn.Identity()
    raise ValueError(f"Unknown norm: {norm}")


def conv3d_norm_act(
    in_ch: int,
    out_ch: int,
    kernel_size=(3, 3, 3),
    stride=1,
    padding=(1, 1, 1),
    dilation=(1, 1, 1),
    groups: int = 1,
    bias: bool = False,
    pad_mode: str = "replicate",
    norm_mode: str = "bn",
    act_mode: str = "relu",
) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv3d(
            in_ch,
            out_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            padding_mode=pad_mode,
            bias=bias,
        ),
        get_norm_3d(norm_mode, out_ch),
        get_activation(act_mode),
    )


# ---------------------------------------------------------------------------
# SE attention
# ---------------------------------------------------------------------------


class SELayer3d(nn.Module):
    def __init__(self, channels: int, reduction: int = 4, act_mode: str = "relu"):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, max(1, channels // reduction), bias=False),
            get_activation(act_mode),
            nn.Linear(max(1, channels // reduction), channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c = x.shape[:2]
        w = self.pool(x).view(b, c)
        return x * self.fc(w).view(b, c, 1, 1, 1)


# ---------------------------------------------------------------------------
# Residual blocks
# ---------------------------------------------------------------------------


class BasicBlock3d(nn.Module):
    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride=1,
        pad_mode: str = "replicate",
        act_mode: str = "relu",
        norm_mode: str = "bn",
        **kwargs,
    ):
        super().__init__()
        self.conv = nn.Sequential(
            conv3d_norm_act(
                in_planes,
                planes,
                (3, 3, 3),
                stride=stride,
                padding=(1, 1, 1),
                pad_mode=pad_mode,
                norm_mode=norm_mode,
                act_mode=act_mode,
            ),
            conv3d_norm_act(
                planes,
                planes,
                (3, 3, 3),
                stride=1,
                padding=(1, 1, 1),
                pad_mode=pad_mode,
                norm_mode=norm_mode,
                act_mode="none",
            ),
        )
        self.proj = nn.Identity()
        if in_planes != planes or stride != 1:
            self.proj = conv3d_norm_act(
                in_planes,
                planes,
                (1, 1, 1),
                stride=stride,
                padding=0,
                norm_mode=norm_mode,
                act_mode="none",
            )
        self.act = get_activation(act_mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x) + self.proj(x))


class BasicBlock3dSE(BasicBlock3d):
    def __init__(
        self, in_planes: int, planes: int, act_mode: str = "leaky_relu", **kwargs
    ):
        super().__init__(
            in_planes=in_planes, planes=planes, act_mode=act_mode, **kwargs
        )
        self.conv = nn.Sequential(self.conv, SELayer3d(planes, act_mode=act_mode))


_BLOCK_DICT = {
    "residual": BasicBlock3d,
    "residual_se": BasicBlock3dSE,
}


# ---------------------------------------------------------------------------
# Cross-attention Transformer components
# ---------------------------------------------------------------------------


class _Attention(nn.Module):
    def __init__(self, D: int = 96, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        assert D % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = D // num_heads
        self.scale = math.sqrt(self.head_dim)
        self.q = nn.Linear(D, D)
        self.k = nn.Linear(D, D)
        self.v = nn.Linear(D, D)
        self.out = nn.Linear(D, D)
        self.drop_a = nn.Dropout(dropout)
        self.drop_p = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        return x.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

    def forward(self, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        q, k, v = (
            self._split(self.q(query)),
            self._split(self.k(key)),
            self._split(self.v(key)),
        )
        attn = self.drop_a(
            self.softmax(torch.matmul(q, k.transpose(-1, -2)) / self.scale)
        )
        ctx = torch.matmul(attn, v).permute(0, 2, 1, 3).contiguous()
        ctx = ctx.view(ctx.shape[0], ctx.shape[1], -1)
        return self.drop_p(self.out(ctx))


class _Mlp(nn.Module):
    def __init__(self, D: int = 96, mlp_dim: int = 512, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(D, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, D)
        self.drop = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.drop(F.gelu(self.fc1(x)))))


class _TransformerBlock(nn.Module):
    def __init__(
        self, D: int = 96, num_heads: int = 4, mlp_dim: int = 512, dropout: float = 0.0
    ):
        super().__init__()
        self.norm_q = nn.LayerNorm(D, eps=1e-6)
        self.norm_kv = nn.LayerNorm(D, eps=1e-6)
        self.attn = _Attention(D, num_heads, dropout)
        self.norm_ff = nn.LayerNorm(D, eps=1e-6)
        self.ffn = _Mlp(D, mlp_dim, dropout)

    def forward(self, x_h: torch.Tensor, x_c: torch.Tensor) -> torch.Tensor:
        x_h = x_h + self.attn(self.norm_q(x_h), self.norm_kv(x_c))
        x_h = x_h + self.ffn(self.norm_ff(x_h))
        return x_h


class _PatchEmbed(nn.Module):
    def __init__(self, in_ch: int, D: int, n_voxels: int):
        super().__init__()
        self.proj = nn.Conv3d(in_ch, D, kernel_size=1)
        self.pos = nn.Parameter(torch.zeros(1, n_voxels, D))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.proj(x).flatten(2).transpose(-1, -2)  # (B, N, D)
        pos = self.pos
        if tokens.shape[1] != pos.shape[1]:
            # interpolate pos embedding to actual token count (handles variable input sizes)
            pos = F.interpolate(
                pos.transpose(1, 2), size=tokens.shape[1], mode="linear", align_corners=False
            ).transpose(1, 2)
        return tokens + pos


class CrossAttentionTransformer(nn.Module):
    """Cross-attention transformer fusing high-res (x_h) and coarse (x_c) bottleneck features."""

    def __init__(
        self,
        in_channels: int,
        n_voxels: int,
        D: int = 96,
        num_heads: int = 4,
        mlp_dim: int = 512,
        depth: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_h = _PatchEmbed(in_channels, D, n_voxels)
        self.embed_c = _PatchEmbed(in_channels, D, n_voxels)
        self.layers = nn.ModuleList(
            [_TransformerBlock(D, num_heads, mlp_dim, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(D, eps=1e-6)

    def forward(self, x_h: torch.Tensor, x_c: torch.Tensor) -> torch.Tensor:
        spatial = x_h.shape[2:]
        e_h = self.embed_h(x_h)
        e_c = self.embed_c(x_c)
        for layer in self.layers:
            e_h = layer(e_h, e_c)
        e_h = self.norm(e_h)
        B, _, D = e_h.shape
        return e_h.permute(0, 2, 1).contiguous().view(B, D, *spatial)


# ---------------------------------------------------------------------------
# Mamba-based Fusion Layer
# ---------------------------------------------------------------------------


class _MambaLayer(nn.Module):
    def __init__(self, dim: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            bimamba_type="v3",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype in (torch.float16, torch.bfloat16):
            x = x.float()
        B, C, *spatial = x.shape
        n = 1
        for s in spatial:
            n *= s
        flat = x.reshape(B, C, n).transpose(-1, -2)  # (B, n, C)
        normed = self.norm(flat)
        # bimamba_type="v3" chunks the sequence into nslices equal parts;
        # pad to the next multiple so chunk() produces equal-length pieces.
        nslices = getattr(self.mamba, "nslices", 1)
        pad = (nslices - n % nslices) % nslices
        if pad:
            normed = F.pad(normed, (0, 0, 0, pad))
        out = self.mamba(normed)
        if pad:
            out = out[:, :n, :]
        return out.transpose(-1, -2).reshape(B, C, *spatial)


class FusionLayer(nn.Module):
    """Cat CUN and HRUN finest-scale features, refine with Mamba + conv."""

    def __init__(self, in_ch: int = 16):
        super().__init__()
        fuse_ch = in_ch * 2
        self.mam1 = _MambaLayer(fuse_ch)
        self.c1 = nn.Conv3d(fuse_ch, in_ch, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(in_ch)
        self.mam2 = _MambaLayer(in_ch)
        self.c2 = nn.Conv3d(in_ch, in_ch, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(in_ch)

    def forward(self, l: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        x = torch.cat((l, s), dim=1)
        x = self.mam1(x) + x
        x = self.bn1(self.c1(x))
        x = self.mam2(x) + x
        x = self.bn2(self.c2(x))
        return x


# ---------------------------------------------------------------------------
# GBPNet
# ---------------------------------------------------------------------------


class GBPNet(nn.Module):
    """Dual-pathway U-Net with cross-attention fusion for 3D neuron segmentation.

    Args:
        block_type: residual block variant ("residual" or "residual_se").
        in_channel: number of input channels (typically 1 for grayscale).
        out_channel: number of output channels (typically 1 for binary seg).
        filters: channel counts at each encoder depth.
        isotropy: per-layer isotropy flag. True → 3×3×3 kernels / stride-2;
                  False → 1×3×3 kernels / stride-(1,2,2). Default all-True.
        is_isotropic: controls the I/O layer kernel shape.
        pad_mode: Conv padding mode.
        act_mode: activation function.
        norm_mode: normalisation type.
        pooling: use MaxPool instead of strided convolutions.
        n_voxels: spatial token count at bottleneck, must equal the product of
                  bottleneck spatial dims (e.g. 8*8*8=512 for 128^3 input with
                  depth-5 filters).
        transformer_D: transformer hidden dim (should equal filters[-1]).
        transformer_depth: number of transformer layers.
        loss_c_weight: weight for the auxiliary coarse-path loss (0 = disabled).
    """

    def __init__(
        self,
        block_type: str = "residual_se",
        in_channel: int = 1,
        out_channel: int = 1,
        filters: List[int] = None,
        isotropy: Optional[List[bool]] = None,
        is_isotropic: bool = True,
        pad_mode: str = "replicate",
        act_mode: str = "leaky_relu",
        norm_mode: str = "bn",
        pooling: bool = False,
        n_voxels: int = 512,
        transformer_D: int = 96,
        transformer_depth: int = 4,
        **kwargs,
    ):
        super().__init__()
        if filters is None:
            filters = [16, 24, 48, 72, 96]
        if isotropy is None:
            isotropy = [True] * len(filters)
        assert len(isotropy) == len(filters)

        self.depth = len(filters)
        self.isotropy = isotropy
        self.pooling = pooling

        block = _BLOCK_DICT[block_type]
        kw = dict(pad_mode=pad_mode, act_mode=act_mode, norm_mode=norm_mode)

        io_ks, io_pad = self._io_kernel(is_isotropic)

        # I/O layers for both pathways
        self.conv_in_c = conv3d_norm_act(
            in_channel, filters[0], io_ks, padding=io_pad, **kw
        )
        self.conv_out_c = nn.Sequential(
            nn.Conv3d(
                in_channel + filters[0], out_channel, io_ks, padding=io_pad, bias=True
            )
        )
        self.conv_in_h = conv3d_norm_act(
            in_channel, filters[0], io_ks, padding=io_pad, **kw
        )
        self.conv_out_h = nn.Sequential(
            nn.Conv3d(
                in_channel + filters[0], out_channel, io_ks, padding=io_pad, bias=True
            )
        )

        # Encoder: CUN (coarse pathway)
        self.down_c = self._build_encoder(filters, isotropy, block, kw)
        # Encoder: HRUN (high-res pathway)
        self.down_h = self._build_encoder(filters, isotropy, block, kw)

        # Cross-attention transformer at bottleneck
        assert (
            filters[-1] == transformer_D
        ), f"transformer_D ({transformer_D}) must equal filters[-1] ({filters[-1]})"
        self.transformer = CrossAttentionTransformer(
            in_channels=filters[-1],
            n_voxels=n_voxels,
            D=transformer_D,
            depth=transformer_depth,
        )

        # Decoder: CUN and HRUN share the same structure
        self.up_c = self._build_decoder(filters, isotropy, block, kw)
        self.up_h = self._build_decoder(filters, isotropy, block, kw)

        # Mamba-based fusion at finest scale
        self.fuse = FusionLayer(in_ch=filters[0])

        self._init_weights()

    # ------------------------------------------------------------------
    # Builder helpers
    # ------------------------------------------------------------------

    def _build_encoder(self, filters, isotropy, block, kw):
        layers = nn.ModuleList()
        for i in range(self.depth):
            prev = max(0, i - 1)
            ks, pad = self._body_kernel(isotropy[i])
            stride = self._stride(isotropy[i], prev, i)
            layers.append(
                nn.Sequential(
                    self._pool(isotropy[i], prev, i),
                    conv3d_norm_act(
                        filters[prev], filters[i], ks, stride=stride, padding=pad, **kw
                    ),
                    block(filters[i], filters[i], **kw),
                )
            )
        return layers

    def _build_decoder(self, filters, isotropy, block, kw):
        layers = nn.ModuleList()
        for j in range(1, self.depth):
            ks, pad = self._body_kernel(isotropy[j])
            layers.append(
                nn.ModuleList(
                    [
                        conv3d_norm_act(
                            filters[j], filters[j - 1], ks, padding=pad, **kw
                        ),
                        block(filters[j - 1], filters[j - 1], **kw),
                    ]
                )
            )
        return layers

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x_h: torch.Tensor, x_c: torch.Tensor):
        x_h_res = x_h.clone()
        x_c_res = x_c.clone()

        # ---- CUN encoding ----
        feat_c = self.conv_in_c(x_c)
        skip_c = []
        for i in range(self.depth - 1):
            feat_c = self.down_c[i](feat_c)
            skip_c.append(feat_c)
        feat_c = self.down_c[-1](feat_c)

        # ---- HRUN encoding ----
        feat_h = self.conv_in_h(x_h)
        skip_h = []
        for i in range(self.depth - 1):
            feat_h = self.down_h[i](feat_h)
            skip_h.append(feat_h)
        feat_h = self.down_h[-1](feat_h)

        # ---- Cross-attention fusion at bottleneck ----
        feat_h = self.transformer(feat_h, feat_c)

        # ---- Dual decoding ----
        for j in range(self.depth - 1):
            i = self.depth - 2 - j  # finest → coarsest order: 3,2,1,0
            feat_c = self._upsample_add(self.up_c[i][0](feat_c), skip_c[i])
            feat_c = self.up_c[i][1](feat_c)

            feat_h = self._upsample_add(self.up_h[i][0](feat_h), skip_h[i])
            feat_h = self.up_h[i][1](feat_h)

            if j == self.depth - 2:  # finest scale: fuse CUN ROI into HRUN
                B, C, D, W, H = feat_c.shape
                roi = feat_c[
                    :, :, D // 3 : 2 * D // 3, W // 3 : 2 * W // 3, H // 3 : 2 * H // 3
                ]
                roi = F.interpolate(
                    roi, size=feat_h.shape[2:], mode="trilinear", align_corners=False
                )
                feat_h = self.fuse(roi, feat_h)

        pred_h = self.conv_out_h(torch.cat((x_h_res, feat_h), dim=1))
        pred_c = self.conv_out_c(torch.cat((x_c_res, feat_c), dim=1))
        return pred_h, pred_c

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _upsample_add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(
            x, size=y.shape[2:], mode="trilinear", align_corners=not self.pooling
        )
        return x + y

    @staticmethod
    def _io_kernel(is_isotropic: bool):
        return ((5, 5, 5), (2, 2, 2)) if is_isotropic else ((1, 5, 5), (0, 2, 2))

    @staticmethod
    def _body_kernel(is_isotropic: bool):
        return ((3, 3, 3), (1, 1, 1)) if is_isotropic else ((1, 3, 3), (0, 1, 1))

    def _stride(self, is_isotropic: bool, prev: int, i: int):
        if self.pooling or prev == i:
            return 1
        return 2 if is_isotropic else (1, 2, 2)

    def _pool(self, is_isotropic: bool, prev: int, i: int):
        if self.pooling and prev != i:
            k = 2 if is_isotropic else (1, 2, 2)
            return nn.MaxPool3d(k, k)
        return nn.Identity()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode="fan_in")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def freeze_cun(self):
        """Freeze all CUN-pathway parameters (call before stage-2 training)."""
        for p in self.conv_in_c.parameters():
            p.requires_grad_(False)
        for p in self.conv_out_c.parameters():
            p.requires_grad_(False)
        for p in self.down_c.parameters():
            p.requires_grad_(False)
        for p in self.up_c.parameters():
            p.requires_grad_(False)


# ---------------------------------------------------------------------------
# GBPNetCUN  —  Stage-1 standalone coarse U-Net
# ---------------------------------------------------------------------------


class GBPNetCUN(nn.Module):
    """Coarse U-Net (CUN) trained independently in Stage 1.

    Attribute names (conv_in_c / down_c / up_c / conv_out_c) are identical to
    GBPNet's CUN pathway, so Stage-1 weights load into GBPNet via::

        gbpnet.load_state_dict(cun_state_dict, strict=False)

    where cun_state_dict strips the Lightning "model." prefix.

    Args: same architectural kwargs as GBPNet (block_type, filters, isotropy, …).
    """

    def __init__(
        self,
        block_type: str = "residual_se",
        in_channel: int = 1,
        out_channel: int = 1,
        filters: List[int] = None,
        isotropy: Optional[List[bool]] = None,
        is_isotropic: bool = False,
        pad_mode: str = "replicate",
        act_mode: str = "leaky_relu",
        norm_mode: str = "bn",
        pooling: bool = False,
        **kwargs,
    ):
        super().__init__()
        if filters is None:
            filters = [16, 24, 48, 72, 96]
        if isotropy is None:
            isotropy = [False, True, True, True, True]
        assert len(isotropy) == len(filters)

        self.depth = len(filters)
        self.isotropy = isotropy
        self.pooling = pooling

        block = _BLOCK_DICT[block_type]
        kw = dict(pad_mode=pad_mode, act_mode=act_mode, norm_mode=norm_mode)
        io_ks, io_pad = GBPNet._io_kernel(is_isotropic)

        # Same names as GBPNet's CUN pathway → weights transfer directly
        self.conv_in_c = conv3d_norm_act(
            in_channel, filters[0], io_ks, padding=io_pad, **kw
        )
        self.conv_out_c = nn.Sequential(
            nn.Conv3d(
                in_channel + filters[0], out_channel, io_ks, padding=io_pad, bias=True
            )
        )
        self.down_c = self._build_encoder(filters, isotropy, block, kw)
        self.up_c = self._build_decoder(filters, isotropy, block, kw)

        self._init_weights()

    def _build_encoder(self, filters, isotropy, block, kw):
        layers = nn.ModuleList()
        for i in range(self.depth):
            prev = max(0, i - 1)
            ks, pad = GBPNet._body_kernel(isotropy[i])
            stride = GBPNet._stride(self, isotropy[i], prev, i)
            layers.append(
                nn.Sequential(
                    GBPNet._pool(self, isotropy[i], prev, i),
                    conv3d_norm_act(
                        filters[prev], filters[i], ks, stride=stride, padding=pad, **kw
                    ),
                    block(filters[i], filters[i], **kw),
                )
            )
        return layers

    def _build_decoder(self, filters, isotropy, block, kw):
        layers = nn.ModuleList()
        for j in range(1, self.depth):
            ks, pad = GBPNet._body_kernel(isotropy[j])
            layers.append(
                nn.ModuleList(
                    [
                        conv3d_norm_act(
                            filters[j], filters[j - 1], ks, padding=pad, **kw
                        ),
                        block(filters[j - 1], filters[j - 1], **kw),
                    ]
                )
            )
        return layers

    def forward(self, x_c: torch.Tensor) -> torch.Tensor:
        x_res = x_c.clone()
        feat = self.conv_in_c(x_c)
        skips = []
        for i in range(self.depth - 1):
            feat = self.down_c[i](feat)
            skips.append(feat)
        feat = self.down_c[-1](feat)
        for j in range(self.depth - 1):
            i = self.depth - 2 - j
            feat = GBPNet._upsample_add(self, self.up_c[i][0](feat), skips[i])
            feat = self.up_c[i][1](feat)
        return self.conv_out_c(torch.cat((x_res, feat), dim=1))

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode="fan_in")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
