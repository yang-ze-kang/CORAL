import torch.nn as nn
from typing import Dict, List
from monai.networks.nets import ResNet
from torchvision.models._utils import IntermediateLayerGetter
import torch
import torch.nn.functional as F

from .position_encoding import PositionEmbeddingSine3D
from utils.misc import NestedTensor



class DinoBackbone(nn.Module):
    def __init__(self, name="resnet50", in_channels=1, hidden_channels=[64, 128, 256, 512], return_interm_indices=[0, 1, 2, 3]):
        super().__init__()
        if name == "resnet50":
            model = ResNet(
                spatial_dims=3,
                n_input_channels=in_channels,
                block="bottleneck",
                layers=[3, 4, 6, 3],
                block_inplanes=hidden_channels,
                feed_forward=False,
            )
            model.hidden_channels = hidden_channels
        else:
            raise NotImplementedError
        model.out_channels = hidden_channels[4-len(return_interm_indices):]
        return_layers = {}
        for idx, layer_index in enumerate(return_interm_indices):
            return_layers.update({"layer{}".format(5 - len(return_interm_indices) + idx): "{}".format(layer_index)})
        self.body = IntermediateLayerGetter(model, return_layers=return_layers)

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out = []
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out.append(NestedTensor(x, mask))
        return out
