# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

def weights_init_normal(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find("Conv3d") != -1 or classname.find("ConvTranspose3d") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("Linear") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)

class Conv33(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(Conv33, self).__init__()

        # Replace the first Conv3d layer with a MobileNetV3 depthwise separable convolution
        self.conv1 = nn.Sequential(nn.Conv3d(in_feat, out_feat, kernel_size=(3, 3, 3), stride=1, padding=1, groups=in_feat),
                                   nn.BatchNorm3d(out_feat),
                                   nn.ReLU())
        # Replace the second Conv3d layer with a MobileNetV3 pointwise convolution
        self.conv2 = nn.Sequential(nn.Conv3d(out_feat, out_feat, kernel_size=(1, 1, 1)),
                                   nn.BatchNorm3d(out_feat),
                                   nn.ReLU())
    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


class Conv3x3(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(Conv3x3, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv3d(in_feat, out_feat,
                                             kernel_size=(3, 3, 3),
                                             stride=1,
                                             padding=1),
                                   nn.BatchNorm3d(out_feat),
                                   nn.ReLU())
    def forward(self, inputs):
        outputs = self.conv1(inputs)
        # outputs = self.conv2(outputs)
        return outputs


class UpConcat(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(UpConcat, self).__init__()

        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        # self.deconv = nn.ConvTranspose2d(in_feat, out_feat,
        #                                  kernel_size=3,
        #                                  stride=1,
        #                                  dilation=1)

        self.deconv = nn.ConvTranspose3d(in_feat,
                                         out_feat,
                                         kernel_size=2,
                                         stride=2)

    def forward(self, inputs, down_outputs):
        # TODO: Upsampling required after deconv?
        # outputs = self.up(inputs)
        outputs = self.deconv(inputs)
        out = torch.cat([down_outputs, outputs], 1)
        return out


class UpSample(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(UpSample, self).__init__()

        self.up = nn.Upsample(scale_factor=2, mode='nearest')

        self.deconv = nn.ConvTranspose2d(in_feat,
                                         out_feat,
                                         kernel_size=2,
                                         stride=2)

    def forward(self, inputs, down_outputs):
        # TODO: Upsampling required after deconv?
        outputs = self.up(inputs)
        # outputs = self.deconv(inputs)
        out = torch.cat([outputs, down_outputs], 1)
        return out


class BAM(nn.Module):
    def __init__(self, dim):
        super(BAM, self).__init__()
        self.layer_channel = nn.Sequential(nn.Linear(dim, int(dim / 8)),
                                           nn.Linear(int(dim / 8), dim))
        self.layer_spatial_1 = nn.Sequential(nn.Conv3d(dim, int(dim / 8), kernel_size=1, padding=0, bias=True),
                                             nn.Conv3d(int(dim / 8), int(dim / 8), kernel_size=(7, 7, 3), stride=1,
                                                       padding=(3, 3, 1),
                                                       bias=True))
        self.layer_spatial_2 = nn.Sequential(nn.Conv3d(dim, int(dim / 8), kernel_size=1, padding=0, bias=True),
                                             nn.Conv3d(int(dim / 8), int(dim / 8), kernel_size=(5, 5, 3),stride=1,
                                                       padding=(2, 2, 1),
                                                       bias=True))
        self.layer_spatial_3 = nn.Sequential(nn.Conv3d(dim, int(dim / 8), kernel_size=1, padding=0, bias=True),
                                             nn.Conv3d(int(dim / 8), int(dim / 8), kernel_size=(3, 3, 1), stride=1,
                                                       padding=(1, 1, 0),
                                                       bias=True))
        self.layer_spatial = nn.Sequential(
            nn.Conv3d(int(dim / 8) * 3, int(dim / 8), kernel_size=1, padding=0, bias=True),
            nn.Conv3d(int(dim / 8), 1, kernel_size=1, padding=0, bias=True))

    def forward(self, x):
        wh = [x.shape[2], x.shape[3], x.shape[4]]
        x1 = F.avg_pool3d(x, wh)
        x1 = x1.view(x1.shape[0], -1)
        x1 = self.layer_channel(x1)
        x1 = x1.view(x1.shape[0], -1, 1, 1, 1).expand(x.shape[0], x.shape[1], x.shape[2], x.shape[3], x.shape[4])
        x2_1 = self.layer_spatial_1(x1)
        x2_2 = self.layer_spatial_2(x1)
        x2_3 = self.layer_spatial_3(x1)
        x2 = torch.cat([x2_1, x2_2, x2_3], dim=1)
        x2 = self.layer_spatial(x2)
        x2 = x2.expand(x.shape[0], x.shape[1], x.shape[2], x.shape[3], x.shape[4])
        x1 = x1 + x2
        x1 = torch.sigmoid(x1)
        x = x * x1
        return x

class MSFE_net(nn.Module):
    def __init__(self,num_channels=1,num_feat=[16, 32, 64, 128]):
        super(MSFE_net, self).__init__()
        self.down1 = nn.Sequential(Conv33(num_channels, num_feat[0]))

        self.down2 = nn.Sequential(nn.MaxPool3d(kernel_size=2),
                                   Conv33(num_feat[0], num_feat[1]))

        self.down3 = nn.Sequential(nn.MaxPool3d(kernel_size=2),
                                   Conv33(num_feat[1], num_feat[2]))

        self.down4 = nn.Sequential(nn.MaxPool3d(kernel_size=2),
                                   Conv33(num_feat[2], num_feat[3]))

        self.attention_layer_channel1 = BAM(dim=num_feat[0])
        self.attention_layer_channel2 = BAM(dim=num_feat[1])
        self.attention_layer_channel3 = BAM(dim=num_feat[2])
        self.attention_layer_channel4 = BAM(dim=num_feat[3])
    def forward(self,inputs):
        # print(inputs.data.size())
        down1_feat = self.down1(inputs)
        down1_feat = self.attention_layer_channel1(down1_feat)
        # print(down1_feat.size())
        down2_feat = self.down2(down1_feat)
        down2_feat = self.attention_layer_channel2(down2_feat)
        # print(down2_feat.size())
        down3_feat = self.down3(down2_feat)
        down3_feat = self.attention_layer_channel3(down3_feat)
        # print(down3_feat.size())
        down4_feat = self.down4(down3_feat)
        down4_feat = self.attention_layer_channel4(down4_feat)
        # print(down4_feat.size())

        feature_map = [down1_feat, down2_feat, down3_feat, down4_feat]
        return feature_map,down4_feat

class EFC_net(nn.Module):

    
    def __init__(self, hidden_size=256, img_size=4096):
        super(EFC_net, self).__init__()
        self.conv_block = Conv3x3(hidden_size,1)
        self.classify = nn.Sequential(
            nn.Linear(img_size,img_size//4),
            nn.BatchNorm1d(img_size//4),
            nn.ReLU(inplace=True),
            nn.Dropout1d(0.2),
            nn.Linear(img_size//4, 1),
        )

    def forward(self, x):
        feature_map = self.conv_block(x)
        feature_map = torch.flatten(feature_map, 1)
        out = self.classify(feature_map)
        return out


class PSAD_net(nn.Module):

    def __init__(self, num_feat):
        super(PSAD_net, self).__init__()
        self.up1 = UpConcat(num_feat[3], num_feat[2])
        self.upconv1 = Conv3x3(num_feat[3], num_feat[2])

        self.up2 = UpConcat(num_feat[2], num_feat[1])
        self.upconv2 = Conv3x3(num_feat[2], num_feat[1])

        self.up3 = UpConcat(num_feat[1], num_feat[0])
        self.upconv3 = Conv3x3(num_feat[1], num_feat[0])

        self.final = nn.Sequential(nn.Conv3d(num_feat[0], 1,kernel_size=1))

    def forward(self,feature_map,num=0):
        down1_feat,down2_feat,down3_feat,down4_feat = feature_map

        # Sigle spad
        down1_feat = down1_feat
        down2_feat = down2_feat
        down3_feat = down3_feat
        down4_feat = down4_feat

        up1_feat = self.up1(down4_feat, down3_feat)
        up1_feat = self.upconv1(up1_feat)

        up2_feat = self.up2(up1_feat, down2_feat)
        up2_feat = self.upconv2(up2_feat)

        up3_feat = self.up3(up2_feat, down1_feat)
        up3_feat = self.upconv3(up3_feat)

        outputs = self.final(up3_feat)
        #print(outputs.size())
        return outputs


class DTAnet(nn.Module):
    def __init__(self, in_channels=1, num_classes=2, use_snr_classify=True):
        super().__init__()
        # num_feat = [16, 32, 64, 128]
        num_feat = [32, 64, 128, 256]
        self.use_snr_classify = use_snr_classify
        self.msfe = MSFE_net(in_channels, num_feat)
        self.classify = EFC_net(hidden_size=num_feat[-1], img_size=4096)

        self.SN_pasd = PSAD_net(num_feat)
        self.WS_pasd = PSAD_net(num_feat)

    def forward(self, inputs):
        feature_map, down4_feat = self.msfe(inputs)   # feature_map: List[Tensor], each [B, C, D, H, W]
        if not self.use_snr_classify:
            out_sn = self.SN_pasd(feature_map)
            out_ws = self.WS_pasd(feature_map)
            out_avg = 0.5 * (out_sn + out_ws)
            if self.training:
                return {"mask": out_avg, "mask_sn": out_sn, "mask_ws": out_ws}
            return out_avg

        classify_out = self.classify(down4_feat)      # [B, 1]

        B = inputs.shape[0]

        # One scalar probability per sample: [B]
        prob = torch.sigmoid(classify_out).squeeze(1)   # [B]
        route = prob >= 0.5                              # [B] bool

        idx_sn = torch.where(route)[0]                   # 1D long indices
        idx_ws = torch.where(~route)[0]                  # 1D long indices

        def _select_feat_list(feats, idx):
            # feats: list of tensors, each [B, ...]
            # idx:  1D long tensor [N]
            return [f.index_select(0, idx) for f in feats]

        outputs = None

        # SN branch
        if idx_sn.numel() > 0:
            feat_sn = _select_feat_list(feature_map, idx_sn)
            out_sn = self.SN_pasd(feat_sn)               # [Nsn, 1, D, H, W]

            outputs = out_sn.new_empty((B, *out_sn.shape[1:]))  # [B, 1, D, H, W]
            outputs[idx_sn] = out_sn

        # WS branch
        if idx_ws.numel() > 0:
            feat_ws = _select_feat_list(feature_map, idx_ws)
            out_ws = self.WS_pasd(feat_ws)               # [Nws, 1, D, H, W]

            if outputs is None:
                outputs = out_ws.new_empty((B, *out_ws.shape[1:]))
            outputs[idx_ws] = out_ws

        if self.training:
            return {"mask": outputs, "pred_label": classify_out}
        else:
            return outputs
