# !/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.
import imp

import __init__ as booger
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter, Softmax

class ResContextBlock(nn.Module):
    def __init__(self, in_filters, out_filters):
        super(ResContextBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_filters, out_filters, kernel_size=(1, 1), stride=1)
        self.act1 = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(out_filters, out_filters, (3,3), padding=1)
        self.act2 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(out_filters)

        self.conv3 = nn.Conv2d(out_filters, out_filters, (3,3),dilation=2, padding=2)
        self.act3 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm2d(out_filters)


    def forward(self, x):

        shortcut = self.conv1(x)
        shortcut = self.act1(shortcut)

        resA = self.conv2(shortcut)
        resA = self.act2(resA)
        resA1 = self.bn1(resA)

        resA = self.conv3(resA1)
        resA = self.act3(resA)
        resA2 = self.bn2(resA)

        output = shortcut + resA2
        return output


class ResBlock(nn.Module):
    def __init__(self, in_filters, out_filters, dropout_rate, kernel_size=(3, 3), stride=1,
                 pooling=True, drop_out=True):
        super(ResBlock, self).__init__()
        self.pooling = pooling
        self.drop_out = drop_out
        self.conv1 = nn.Conv2d(in_filters, out_filters, kernel_size=(1, 1), stride=stride)
        self.act1 = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(in_filters, out_filters, kernel_size=(3,3), padding=1)
        self.act2 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(out_filters)

        self.conv3 = nn.Conv2d(out_filters, out_filters, kernel_size=(3,3),dilation=2, padding=2)
        self.act3 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm2d(out_filters)

        self.conv4 = nn.Conv2d(out_filters, out_filters, kernel_size=(2, 2), dilation=2, padding=1)
        self.act4 = nn.LeakyReLU()
        self.bn3 = nn.BatchNorm2d(out_filters)

        self.conv5 = nn.Conv2d(out_filters*3, out_filters, kernel_size=(1, 1))
        self.act5 = nn.LeakyReLU()
        self.bn4 = nn.BatchNorm2d(out_filters)

        if pooling:
            self.dropout = nn.Dropout2d(p=dropout_rate)
            self.pool = nn.AvgPool2d(kernel_size=kernel_size, stride=2, padding=1)
        else:
            self.dropout = nn.Dropout2d(p=dropout_rate)

    def forward(self, x):
        shortcut = self.conv1(x)
        shortcut = self.act1(shortcut)

        resA = self.conv2(x)
        resA = self.bn1(resA)
        resA1 = self.act2(resA)

        resA = self.conv3(resA1)
        resA = self.bn2(resA)
        resA2 = self.act3(resA)

        resA = self.conv4(resA2)
        resA = self.bn3(resA)
        resA3 = self.act4(resA)

        concat = torch.cat((resA1,resA2,resA3),dim=1)
        resA = self.conv5(concat)
        resA = self.bn4(resA)
        resA = self.act5(resA)
        resA = shortcut + resA


        if self.pooling:
            if self.drop_out:
                resB = self.dropout(resA)
            else:
                resB = resA
            resB = self.pool(resB)
            return resB, resA
        else:
            if self.drop_out:
                resB = self.dropout(resA)
            else:
                resB = resA
            return resB


class UpBlock(nn.Module):
    def __init__(self, in_filters, out_filters, dropout_rate, drop_out=True):
        super(UpBlock, self).__init__()
        self.drop_out = drop_out
        self.in_filters = in_filters
        self.out_filters = out_filters

        self.dropout1 = nn.Dropout2d(p=dropout_rate)

        self.dropout2 = nn.Dropout2d(p=dropout_rate)

        self.conv1 = nn.Conv2d(in_filters//4 + in_filters, out_filters, (3,3), padding=1)
        self.act1 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(out_filters)

        self.conv2 = nn.Conv2d(out_filters, out_filters, (3,3),dilation=2, padding=2)
        self.act2 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm2d(out_filters)

        self.conv3 = nn.Conv2d(out_filters, out_filters, (2,2), dilation=2,padding=1)
        self.act3 = nn.LeakyReLU()
        self.bn3 = nn.BatchNorm2d(out_filters)


        self.conv4 = nn.Conv2d(out_filters*3,out_filters,kernel_size=(1,1))
        self.act4 = nn.LeakyReLU()
        self.bn4 = nn.BatchNorm2d(out_filters)

        self.dropout3 = nn.Dropout2d(p=dropout_rate)

    def forward(self, x, skip):
        '''
        nn.PixelShuffle(x)
        input:batchsizexCinxHinxWin
        output:batchsizexCoutxHoutxWout

        Cout = Cin // x^2
        Hout = Hin * x
        Wout = Win * x
        '''
        upA = nn.PixelShuffle(2)(x)
        if self.drop_out:
            upA = self.dropout1(upA)

        upB = torch.cat((upA,skip),dim=1)
        if self.drop_out:
            upB = self.dropout2(upB)

        upE = self.conv1(upB)
        upE = self.bn1(upE)
        upE1 = self.act1(upE)

        upE = self.conv2(upE1)
        upE = self.bn2(upE)
        upE2 = self.act2(upE)

        upE = self.conv3(upE2)
        upE = self.bn3(upE)
        upE3 = self.act3(upE)

        concat = torch.cat((upE1,upE2,upE3),dim=1)
        upE = self.conv4(concat)
        upE = self.bn4(upE)
        upE = self.act4(upE)
        if self.drop_out:
            upE = self.dropout3(upE)

        return upE


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        

    def forward(self, x):
        identity = x
        
        n,c,h,w = x.size()
        x_h = self.pool_h(x) # n,c,h,1
        x_w = self.pool_w(x).permute(0, 1, 3, 2) # n,c,w,1

        y = torch.cat([x_h, x_w], dim=2) # n,c,h+w,1
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out
    
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)  # 7,3     3,1
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        res = x
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        output = res * self.sigmoid(x)
        return output
class eca_layer(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)

class RCCSA(nn.Module):
    def __init__(self, in_planes=32, kernel_size=7):
        super(RCCSA, self).__init__()
        self.ca = CoordAtt(in_planes, in_planes)
        self.sa = SpatialAttention(kernel_size=kernel_size)
        self.eca = eca_layer()
    def forward(self, x):
        res = x
        output1 = self.ca(x)
        output2 = self.eca(x)
        output = output1 * output2
        output = self.sa(output)
        output = res + output
        return output

class SalsaNext(nn.Module):
    def __init__(self, nclasses):# 19
        super(SalsaNext, self).__init__()
        self.nclasses = nclasses
        self.layer0 = nn.Sequential(
        ResContextBlock(5, 32),
        RCCSA(32, 7),
        ResContextBlock(32, 64),
        nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3,stride=2, padding=1)
        )

        self.layer1 = ResBlock(64, 64, 0.2, pooling=True, drop_out=False)
        self.layer2 = ResBlock(64, 128, 0.2, pooling=True)
        self.layer3 = ResBlock(128, 256, 0.2, pooling=True)
        self.layer4 = ResBlock(256, 512, 0.2, pooling=True)

        self.range_up4 = UpBlock(512, 256, 0.2)
        self.range_up3 = UpBlock(256, 128, 0.2)
        self.range_up2 = UpBlock(128, 64, 0.2)
        self.range_up1 = UpBlock(64, 64, 0.2)

        self.cls = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, nclasses, kernel_size=1)
        )
        self.logits = nn.Conv2d(32, nclasses, kernel_size=(1, 1))

    def forward(self, x):
        downCntx = self.layer0(x)

        down0c = self.layer1(downCntx)
        down1c = self.layer2(down0c)
        down2c = self.layer3(down1c)
        down3c= self.layer4(down2c)

        up4e = self.range_up4(down3c,down2c)
        up3e = self.range_up3(up4e, down1c)
        up2e = self.range_up2(up3e, down0c)

        logits = self.cls(up2e)

        # logits = logits
        logits = F.softmax(logits, dim=1)
        return logits

class EnhanceSemanticContextBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EnhanceSemanticContextBlock, self).__init__()
        self.GlobalAvgPooling = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.Sigmoid(),
        )
        self.conv2 = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.Sigmoid(),
        )
    def forward(self, x):
        res = x # b x 256 x h x w
        # print(res.shape)

        x1 = self.GlobalAvgPooling(x) # b x 256 x 1 x 1
        x1 = self.conv1(x1) # b x 256 x 1 x 1
        # print(x1.shape)

        x2 = self.conv2(x) # b x 256 x h x w
        # print(x2.shape)

        output = res * x2 * x1
        return output
