
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .USC import USC
from .SCCA import SCCA


class conv_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))
        self.dropout = nn.Dropout(p=0.3)
    def forward(self, x):
        x = self.conv(x)
        x = self.dropout(x)
        return x


class Up_Block(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, img_size, config):
        super().__init__()
        self.scale_factor = (img_size // 14, img_size // 14)
        
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))
        
        self.conv = nn.Sequential(
            nn.Conv2d(2*out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

        self.scca = SCCA(skip_ch, in_ch//2, img_size, config) # spatial channel cross-attention (SCCA) module
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, decoder, o_i):
        d_i = self.up(decoder)
        o_hat_i = self.scca(d_i, o_i)
        x = torch.cat((o_hat_i, d_i), dim=1)
        x = self.conv(x)
        x = self.dropout(x)
        return x


class USCT_UNet(nn.Module):
    def __init__(self, config, n_channels=3, n_classes=1, img_size=448, vis=False):
        super(USCT_UNet, self).__init__()
        self.vis = vis
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.img_size = img_size
        in_channels = config.base_channel
        
        n1 = 32
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16] #  32, 64, 128, 256, 512

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(n_channels, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        #  U-shaped skip connections (USC)
        self.USC = USC(config, vis, img_size,
                                     channel_num=[in_channels, in_channels*2, in_channels*4, in_channels*8],
                                     patchSize=config.patch_sizes)

        self.Up5 = Up_Block(filters[4], filters[3], filters[3], self.img_size//8, config)
        self.Up4 = Up_Block(filters[3], filters[2], filters[2], self.img_size//4, config)
        self.Up3 = Up_Block(filters[2], filters[1], filters[1], self.img_size//2, config)
        self.Up2 = Up_Block(filters[1], filters[0], filters[0], self.img_size, config)

        self.pred = nn.Sequential(
            nn.Conv2d(filters[0], filters[0]//2, kernel_size=1),
            nn.BatchNorm2d(filters[0]//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters[0]//2, n_classes, kernel_size=1),)

    def forward(self, x):
        E1 = self.Conv1(x)

        E2 = self.Maxpool1(E1)
        E2 = self.Conv2(E2)

        E3 = self.Maxpool2(E2)
        E3 = self.Conv3(E3)

        E4 = self.Maxpool3(E3)
        E4 = self.Conv4(E4)

        E5 = self.Maxpool4(E4)
        E5 = self.Conv5(E5)

        # U-shaped skip connections (USC)
        O1,O2,O3,O4,att_weights = self.USC(E1, E2, E3, E4)

        d4 = self.Up5(E5, O4)
        d3 = self.Up4(d4, O3)
        d2 = self.Up3(d3, O2)
        d1 = self.Up2(d2, O1)

        if self.n_classes == 1:
            out = nn.Sigmoid()(self.pred(d1))
        else:
            out = self.pred(d1)
       
        if self.vis: # visualize the attention maps
            return out, att_weights
        else:
            return out
        
        