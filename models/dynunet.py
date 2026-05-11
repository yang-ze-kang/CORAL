import torch
import torch.nn as nn
from monai.networks.nets import DynUNet

class DynUNetWithDeconv(DynUNet):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_input_block(self):
        return nn.Sequential(
                self.conv_block(
                self.spatial_dims,
                self.in_channels,
                self.filters[0]//2,
                self.kernel_size[0],
                self.strides[0],
                self.norm_name,
                self.act_name,
                dropout=self.dropout,
            ),
            self.conv_block(
                self.spatial_dims,
                self.filters[0]//2,
                self.filters[0],
                self.kernel_size[0],
                self.strides[0],
                self.norm_name,
                self.act_name,
                dropout=self.dropout,
            )
        )

    def get_output_block(self, idx: int):
        final_channels = self.filters[0]
        return nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=final_channels,
                    out_channels=final_channels // 2,
                    kernel_size=2,
                    stride=2,
                    padding=0
                ) if self.spatial_dims == 2 else nn.ConvTranspose3d(
                    in_channels=final_channels,
                    out_channels=final_channels // 2,
                    kernel_size=2,
                    stride=2,
                    padding=0
                ),
                nn.InstanceNorm2d(final_channels // 2) if self.spatial_dims == 2 else nn.InstanceNorm3d(final_channels // 2),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(
                    in_channels=final_channels // 2,
                    out_channels=self.out_channels,
                    kernel_size=2,
                    stride=2,
                    padding=0
                ) if self.spatial_dims == 2 else nn.ConvTranspose3d(
                    in_channels=final_channels // 2,
                    out_channels=self.out_channels,
                    kernel_size=2,
                    stride=2,
                    padding=0
                )
            )
