import torch
import torch.nn as nn
import torch.nn.functional as F

from models.polar_net import polar_net as PolarUnet
from models.SalsaNext import SalsaNext, EnhanceSemanticContextBlock
from models.kpconv.blocks import KPConv
from models.swin_transformer import SwinTransformer

import torch_scatter

ALIGN=False
BatchNorm = nn.BatchNorm2d

def get_range_model(**kwargs):
    model = SalsaNext(**kwargs)
    return model

def get_polar_model(**kwargs):
    model = PolarUnet(**kwargs)
    return model

def resample_grid(predictions, pxpy):
    resampled = F.grid_sample(predictions, pxpy)

    return resampled


class KPClassifier(nn.Module):
    def __init__(self, in_channels=128, out_channels=128):
        super(KPClassifier, self).__init__()
        self.kpconv = KPConv(
            kernel_size=15,
            p_dim=3,
            in_channels=in_channels,
            out_channels=out_channels,
            KP_extent=1.2,
            radius=0.60,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x, pxyz, pknn):
        # x should be tne concated feature Nx2C
        res = []
        for i in range(x.shape[0]):
            points = pxyz[i, ...]
            feats = x[i, ...].transpose(0, 2).squeeze()
            feats = self.kpconv(points, points, pknn[i, ...], feats)
            res.append(feats.unsqueeze(2).transpose(0, 2).unsqueeze(2))
        res = torch.cat(res, axis=0)
        res = self.relu(self.bn(res))
        return res

class M2CNet(nn.Module):

    def __init__(self, ARCH, layers=34, kernal_size=1, n_class=19, flow=False, data_type=torch.float32):
        super(M2CNet, self).__init__()
        range_model = get_range_model(nclasses=n_class)
        polar_model = get_polar_model(grid_size=ARCH['polar']['grid_size'],
                                      n_height=ARCH['polar']['grid_size'][-1],
                                      n_class=n_class,
                                      layers=layers,
                                      kernal_size=kernal_size,
                                      fea_compre=ARCH['polar']['grid_size'][-1])
        # channels distribution
        self.channels = {18: [64, 128, 256, 512],
                         34: [64, 128, 256, 512],
                         50: [256, 512, 1024, 2048]}[layers]
        self.n_class = n_class
        self.data_type = data_type
        self.n_height = ARCH['polar']['grid_size'][-1]

        # for range net
        self.range_layer0 = range_model.layer0
        self.range_layer1 = range_model.layer1
        self.range_layer2 = range_model.layer2
        self.range_layer3 = range_model.layer3
        self.range_layer4 = range_model.layer4
        
        self.range_up4 = range_model.range_up4
        self.range_up3 = range_model.range_up3
        self.range_up2 = range_model.range_up2
        self.range_up1 = range_model.range_up1
        self.range_cls = range_model.cls

        # for polar net
        self.grid_size = ARCH['polar']['grid_size']
        # for preprocess
        self.polar_preprocess = polar_model.preprocess
        self.polar_reformat_data = polar_model.reformat_data
        # for module
        self.polar_PPmodel = polar_model.PPmodel
        self.polar_compress = polar_model.fea_compression
        if kernal_size != 1:
            self.local_pool_op = torch.nn.MaxPool2d(kernal_size, stride=1, padding=(kernal_size - 1) // 2, dilation=1)
        else:
            self.local_pool_op = None

        self.polar_layer0 = polar_model.unet_model.network.layer0
        self.polar_layer1 = polar_model.unet_model.network.layer1
        self.polar_layer2 = polar_model.unet_model.network.layer2
        self.polar_layer3 = polar_model.unet_model.network.layer3
        self.polar_layer4 = polar_model.unet_model.network.layer4
        self.polar_up4 = polar_model.unet_model.network.up4
        self.polar_delayer4 = polar_model.unet_model.network.delayer4
        self.polar_up3 = polar_model.unet_model.network.up3
        self.polar_delayer3 = polar_model.unet_model.network.delayer3
        self.polar_up2 = polar_model.unet_model.network.up2
        self.polar_delayer2 = polar_model.unet_model.network.delayer2

        self.polar_up1 = polar_model.unet_model.network.up1
        self.polar_delayer1 = polar_model.unet_model.network.delayer1

        self.polar_cls = polar_model.unet_model.network.cls

        # flow [64, 128, 256, 512]
        self.flow = flow
        self.flow_l2_r2p = R2B_flow(64, self.data_type, level=2)
        self.flow_l3_r2p = R2B_flow(128, self.data_type, level=3)
        self.flow_l4_r2p = R2B_flow(256, self.data_type, level=4)
        self.flow_l5_r2p = R2B_flow(512, self.data_type, level=5)

        self.flow_l2_p2r = B2R_flow(64, self.data_type, level=2)
        self.flow_l3_p2r = B2R_flow(128, self.data_type, level=3)
        self.flow_l4_p2r = B2R_flow(256, self.data_type, level=4)
        self.flow_l5_p2r = B2R_flow(512, self.data_type, level=5)

        # aspp
        self.range_aspp = MESCASPP(in_channels=self.channels[3],
                               out_channels=256,
                               atrous_rates=(3, 6, 12, 18))     

        self.polar_aspp = MESCASPP(in_channels=self.channels[3],
                               out_channels=256,
                               atrous_rates=(3, 6, 12, 18))       

        self.kpconv_range = KPClassifier(n_class)
        self.kpconv_polar = KPClassifier(n_class)

        self.kpconv = KPClassifier(256)
        self.kpconv_cls_range = nn.Conv2d(128, n_class, kernel_size=1)
        self.kpconv_cls_polar = nn.Conv2d(128, n_class, kernel_size=1)
        self.kpconv_cls_fusion = nn.Conv2d(128, n_class, kernel_size=1)

    def forward(self, x, pt_fea, xy_ind, num_pt_each, r2p_matrix, p2r_matrix, pxpy_range, pxpypz_polar, pxyz, knns):
        # polar view preprocess
        cat_pt_fea, unq, unq_inv, batch_size, cur_dev = self.polar_preprocess(pt_fea, xy_ind, num_pt_each)
        processed_cat_pt_fea = self.polar_PPmodel(cat_pt_fea)
        # torch scatter does not support float16
        pooled_data, pooled_idx = torch_scatter.scatter_max(processed_cat_pt_fea, unq_inv, dim=0)
        processed_pooled_data = self.polar_compress(pooled_data)

        out_data = self.polar_reformat_data(processed_pooled_data, unq,
                                            batch_size, cur_dev, self.data_type)
        _, _, polar_h, polar_w = out_data.shape # batchsize×32×480×360

        _, _, range_h, range_w = x.shape

        self.p2r_matrix = p2r_matrix
        self.r2p_matrix = r2p_matrix
        self.polar_h = polar_h
        self.range_w = range_w

        # feature extract
        polar_x1 = self.polar_layer0(out_data)  # 1/2 64x240x180
        polar_x2 = self.polar_layer1(polar_x1)  # 1/4 64x120x90
        polar_x3 = self.polar_layer2(polar_x2)  # 1/8 128x60x45

        range_x1 = self.range_layer0(x)  # 1/2 b x 64 x 192 x 768
        range_x2, range_x2_ = self.range_layer1(range_x1)  # 1/4 b x 64 x 96 x 384
        range_x3, range_x3_ = self.range_layer2(range_x2)  # 1/8 b x 128 x 48 x 192
        
        #  Encoder flow1
        range_x3 = self.flow_l3_p2r(self.p2r_matrix.clone(),
                                        range_x3, polar_x3)
        polar_x3 = self.flow_l3_r2p(self.r2p_matrix.clone(),
                                        polar_x3, range_x3)
        # feature extract
        range_x4, range_x4_ = self.range_layer3(range_x3)  # 1/16  b x 256 x 24 x 96
        polar_x4 = self.polar_layer3(polar_x3)  # 1/16  bx256x30x23
        range_x5, range_x5_ = self.range_layer4(range_x4)  # 1/32  b x 512 x 12 x 48
        polar_x5 = self.polar_layer4(polar_x4)  # 1/32 bx512x15x12
        # Encoder flow2
        range_x5 = self.flow_l5_p2r(self.p2r_matrix.clone(),
                                        range_x5, polar_x5)
        polar_x5 = self.flow_l5_r2p(self.r2p_matrix.clone(),
                                        polar_x5, range_x5)

        # ASPP    b x 512 x h0 x w0
        range_x5 =self.range_aspp(range_x5)
        polar_x5 =self.polar_aspp(polar_x5)

        # Decoder feature flow + Upsampling
        range_p4, polar_p4 = self.feature_flow(5, range_x5, polar_x5, range_x5_, polar_x4,
                                               self.flow_l5_p2r, self.flow_l5_r2p,
                                               self.range_up4, self.polar_up4, self.polar_delayer4, flow=False)
        range_p3, polar_p3 = self.feature_flow(4, range_p4, polar_p4, range_x4_, polar_x3,
                                               self.flow_l4_p2r, self.flow_l4_r2p,
                                               self.range_up3, self.polar_up3, self.polar_delayer3, flow=True)

        range_p2, polar_p2 = self.feature_flow(3, range_p3, polar_p3, range_x3_, polar_x2,
                                               self.flow_l3_p2r, self.flow_l3_r2p,
                                               self.range_up2, self.polar_up2, self.polar_delayer2, flow=False)
        
        range_p1, polar_p1 = self.feature_flow(2, range_p2, polar_p2, range_x2_, polar_x1,
                                               self.flow_l2_p2r, self.flow_l2_r2p,
                                               self.range_up1, self.polar_up1, self.polar_delayer1, flow=True)

        range_x = self.range_cls(range_p1) 
        range_x = F.interpolate(range_x, size=(range_h, range_w), mode='bilinear', align_corners=ALIGN)

        polar_x = self.polar_cls(F.pad(polar_p1, (1, 1, 0, 0), mode='circular'))
        polar_x = F.interpolate(polar_x, size=(polar_h, polar_w), mode='bilinear', align_corners=ALIGN)

        # reformat polar feature
        polar_x = polar_x.permute(0, 2, 3, 1).contiguous()
        new_shape = list(polar_x.size())[:3] + [self.n_height, self.n_class]
        polar_x = polar_x.view(new_shape)
        polar_x = polar_x.permute(0, 4, 1, 2, 3).contiguous()

        range_pred = F.grid_sample(range_x, pxpy_range, align_corners=ALIGN)
        polar_pred = F.grid_sample(polar_x, pxpypz_polar, align_corners=ALIGN).squeeze(2)

        range_fea = self.kpconv_range(range_pred, pxyz, knns)
        polar_fea = self.kpconv_polar(polar_pred, pxyz, knns)

        fusion_fea = torch.cat([range_fea, polar_fea], dim=1)
        fusion_fea = self.kpconv(fusion_fea, pxyz, knns)

        # bx19xN
        range_pred_kpconv = self.kpconv_cls_range(range_fea)
        polar_pred_kpconv = self.kpconv_cls_polar(polar_fea)
        fusion_pred_kpconv = self.kpconv_cls_fusion(fusion_fea)

        return fusion_pred_kpconv, range_pred_kpconv, polar_pred_kpconv, range_x, polar_x

    def feature_flow(self, level,
                     range_p_pre, polar_p_pre,
                     range_x, polar_x,
                     flow_p2r, flow_r2p,
                     range_up, polar_up, polar_delayer, flow=False):
        # flow on level
        if flow:
            fused_range_p_pre = flow_p2r(self.p2r_matrix.clone(),
                                         range_p_pre, polar_p_pre)
            # factor changed to be same or [1, factor] for horizontal conv
            fused_polar_p_pre = flow_r2p(self.r2p_matrix.clone(),
                                         polar_p_pre, range_p_pre)
        else:
            fused_range_p_pre, fused_polar_p_pre = range_p_pre, polar_p_pre

        # for range
        range_p = range_up(fused_range_p_pre, range_x)

        # for polar
        polar_p = F.interpolate(fused_polar_p_pre, polar_x.shape[-2:], mode='bilinear', align_corners=ALIGN)
        polar_p = F.pad(polar_p, (1, 1, 0, 0), mode='circular')
        polar_p = polar_up(polar_p)
        polar_p = torch.cat([polar_p, polar_x], dim=1)
        polar_p = polar_delayer(polar_p)

        return range_p, polar_p

class B2R_flow(nn.Module):
    def __init__(self, fea_dim, data_type, level=5):
        super(B2R_flow, self).__init__()
        self.fea_dim = fea_dim
        self.data_type = data_type
        self.level = level

        self.fusion = nn.Sequential(
            nn.Conv2d(fea_dim * 2, fea_dim, kernel_size=3, padding=1, bias=False),
            BatchNorm(fea_dim),
            nn.ReLU(inplace=True)
        )

        SwinTransformer_fusion = SwinTransformer(window_size=[4, 16])
        self.SwinTransformer_fusion_layer1 = SwinTransformer_fusion.layers[0]
        self.SwinTransformer_fusion_layer2 = SwinTransformer_fusion.layers[1]
        self.SwinTransformer_fusion_layer3 = SwinTransformer_fusion.layers[2]
        self.SwinTransformer_fusion_layer4 = SwinTransformer_fusion.layers[3]

        embed_dim = 64
        self.embed_dim = embed_dim
        self.norm512 = nn.LayerNorm(8*embed_dim) # 512
        self.norm256 = nn.LayerNorm(4*embed_dim) # 256
        self.norm128 = nn.LayerNorm(2*embed_dim) # 128
        self.norm64 = nn.LayerNorm(embed_dim)    # 64

    def forward(self, flow_matrix, range_fea, polar_fea):
        """
        range_fea: [N, C1, 64, 2048]
        polar_fea: [N, C2, 480, 360]
        flow_matrix: [N, 64, 2048, 2], need to be [-1, 1] for grid sample
        """
        # rescale the flow matrix
        _, _, H, W = range_fea.shape
        N, C, _, _ = polar_fea.shape
        # because for F.grid_sample, i,j,k index w,h,d (i.e., reverse order)
        flow_matrix = torch.flip(flow_matrix, dims=[-1])
        flow_matrix_scaled = F.interpolate(flow_matrix.permute(0, 3, 1, 2).contiguous().float(),
                                           (H, W), mode='nearest')  # N*2*H*W
        flow_matrix_scaled = flow_matrix_scaled.permute(0, 2, 3, 1).contiguous() # N*H*W*2
        flow_fea = F.grid_sample(polar_fea, flow_matrix_scaled, padding_mode='zeros', align_corners=ALIGN) # N*C*H*W

        fea = torch.cat((range_fea, flow_fea), dim=1)
        res = self.fusion(fea) # B x C x H x W
        res = res.flatten(2).transpose(1, 2)
        if self.level == 5:
            res = self.norm512(res)
            res, H, W = self.SwinTransformer_fusion_layer1(range_fea, H, W, res)
            res = res.view(-1, H, W, 8*self.embed_dim).permute(0, 3, 1, 2).contiguous()
        elif self.level == 4:
            res = self.norm256(res)
            res, H, W = self.SwinTransformer_fusion_layer2(range_fea, H, W, res)
            res = res.view(-1, H, W, 4*self.embed_dim).permute(0, 3, 1, 2).contiguous()
        elif self.level == 3:
            res = self.norm128(res)
            res, H, W = self.SwinTransformer_fusion_layer3(range_fea, H, W, res)
            res = res.view(-1, H, W, 2*self.embed_dim).permute(0, 3, 1, 2).contiguous()
        elif self.level == 2:
            res = self.norm64(res)
            res, H, W = self.SwinTransformer_fusion_layer4(range_fea, H, W, res)
            res = res.view(-1, H, W, self.embed_dim).permute(0, 3, 1, 2).contiguous()
        fea  = range_fea + res

        return fea

class R2B_flow(nn.Module):

    def __init__(self, fea_dim, data_type, level=5):
        super(R2B_flow, self).__init__()
        self.fea_dim = fea_dim
        self.data_type=data_type
        self.level = level

        self.fusion = nn.Sequential(
            nn.Conv2d(fea_dim * 2, fea_dim, kernel_size=3, padding=(1, 0), bias=False),
            BatchNorm(fea_dim),
            nn.ReLU(inplace=True)
        )

        SwinTransformer_fusion = SwinTransformer(window_size=[8, 8])
        self.SwinTransformer_fusion_layer1 = SwinTransformer_fusion.layers[0]
        self.SwinTransformer_fusion_layer2 = SwinTransformer_fusion.layers[1]
        self.SwinTransformer_fusion_layer3 = SwinTransformer_fusion.layers[2]
        self.SwinTransformer_fusion_layer4 = SwinTransformer_fusion.layers[3]

        embed_dim = 64
        self.embed_dim = embed_dim
        self.norm512 = nn.LayerNorm(8*embed_dim) # 512
        self.norm256 = nn.LayerNorm(4*embed_dim) # 256
        self.norm128 = nn.LayerNorm(2*embed_dim) # 128
        self.norm64 = nn.LayerNorm(embed_dim)    # 64

    def forward(self, flow_matrix, polar_fea, range_fea):
        """
        range_fea: [N, C1, 64, 2048]
        polar_fea: [N, C2, 480, 360]
        flow_matrix: [N, 480, 360, 32, 2]
        """
        range_fea_5d = range_fea.unsqueeze(2)

        _, _, H, W = polar_fea.shape
        N, C, _, _ = range_fea.shape

        N, h, w, K, c = flow_matrix.shape
        flow_matrix = flow_matrix.view(N, h, w, K * c).permute(0, 3, 1, 2).contiguous()
        flow_matrix_scaled = F.interpolate(flow_matrix.float(), (H, W), mode='nearest')
        flow_matrix_scaled = flow_matrix_scaled.permute(0, 2, 3, 1).view(N, H, W, K, c)
        flow_matrix_scaled = F.pad(flow_matrix_scaled, pad=(0, 1), mode='constant', value=0.0)
        flow_fea = F.grid_sample(range_fea_5d, flow_matrix_scaled, padding_mode='zeros', align_corners=ALIGN) # N*C*H*W*K

        flow_fea = torch.max(flow_fea, dim=-1)[0] # N*C*H*W

        fea = torch.cat((polar_fea, flow_fea), dim=1)
        fea = F.pad(fea, (1, 1, 0, 0), mode='circular')
        res = self.fusion(fea)

        res = self.fusion(fea) # B x C x H x W
        res = res.flatten(2).transpose(1, 2)
        if self.level == 5:
            res = self.norm512(res)
            res, H, W = self.SwinTransformer_fusion_layer1(polar_fea, H, W, res)
            res = res.view(-1, H, W, 8*self.embed_dim).permute(0, 3, 1, 2).contiguous()
        elif self.level == 4:
            res = self.norm256(res)
            res, H, W = self.SwinTransformer_fusion_layer2(polar_fea, H, W, res)
            res = res.view(-1, H, W, 4*self.embed_dim).permute(0, 3, 1, 2).contiguous()
        elif self.level == 3:
            res = self.norm128(res)
            res, H, W = self.SwinTransformer_fusion_layer3(polar_fea, H, W, res)
            res = res.view(-1, H, W, 2*self.embed_dim).permute(0, 3, 1, 2).contiguous()
        elif self.level == 2:
            res = self.norm64(res)
            res, H, W = self.SwinTransformer_fusion_layer4(polar_fea, H, W, res)
            res = res.view(-1, H, W, self.embed_dim).permute(0, 3, 1, 2).contiguous()
        fea  = polar_fea + res
        return fea

class ASPPConv(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super(ASPPConv, self).__init__()
        self.denseConv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            BatchNorm(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            BatchNorm(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            BatchNorm(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            BatchNorm(out_channels),
            nn.ReLU(),
            EnhanceSemanticContextBlock(256, 256))
    def forward(self, x):
        res = self.denseConv(x)
        fea = res
        return fea


class ASPPPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__()
        self.avgpool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        size = x.shape[-2:]
        fea = self.avgpool(x)
        fea = F.interpolate(fea, size=size, mode="bilinear", align_corners=ALIGN)
        return fea


class MESCASPP(nn.Module):                        
    def __init__(self, in_channels, out_channels=256, atrous_rates=(3, 6, 12, 18)):
        super(MESCASPP, self).__init__()
        modules = []
        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            )
        )

        rate1, rate2, rate3, rate4 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPConv(in_channels, out_channels, rate4))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(6 * out_channels, in_channels, 1, padding=0, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        output = self.project(res)
        return output
