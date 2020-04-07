import torch
import torch.nn as nn
import torch.nn.functional as F

from res_block import res_block


class decoder(nn.Module):
    
    def __init__(
        self,
        base_filters=16
    ):
        
        super(decoder,self).__init__()
        
        self.bf = base_filters
        
        self.upsample4 = nn.ConvTranspose2d(
            in_channels = self.bf * 16,
            out_channels = self.bf * 8,
            kernel_size = 2,
            stride = 2
        )
        
        self.conv4 = nn.Conv2d(
            in_channels = self.bf * 16,
            out_channels = self.bf * 8,
            kernel_size = 1
        )
        
        self.up_block4 = res_block(
            channel_in = self.bf * 8,
            downsample = False
        )
        
        self.upsample3 = nn.ConvTranspose2d(
            in_channels = self.bf * 8,
            out_channels = self.bf * 4,
            kernel_size = 2,
            stride = 2
        )
        
        self.conv3 = nn.Conv2d(
            in_channels = self.bf * 8,
            out_channels = self.bf * 4,
            kernel_size = 1
        )
        
        self.up_block3 = res_block(
            channel_in = self.bf * 4,
            downsample = False
        )
        
        self.upsample2 = nn.ConvTranspose2d(
            in_channels = self.bf * 4,
            out_channels = self.bf * 2,
            kernel_size = 2,
            stride = 2
        )
        
        self.conv2 = nn.Conv2d(
            in_channels = self.bf * 4,
            out_channels = self.bf * 2,
            kernel_size = 1
        )
            
        self.up_block2 = res_block(
            channel_in = self.bf * 2,
            downsample = False
        )
        
        self.upsample1 = nn.ConvTranspose2d(
            in_channels = self.bf * 2,
            out_channels = self.bf,
            kernel_size = 2,
            stride = 2
        )
        
        self.conv1 = nn.Conv2d(
            in_channels = self.bf * 2,
            out_channels = self.bf,
            kernel_size = 1
        )
        
        self.up_block1 = res_block(
            channel_in = self.bf,
            downsample = False
        )
        
    def forward(self, x):
        
        up4 = self.upsample4(x)
        self.up_level4 = self.up_block4(up4)
        
        up3 = self.upsample3(self.up_level4)
        self.up_level3 = self.up_block3(up3)
        
        up2 = self.upsample2(self.up_level3)
        self.up_level2 = self.up_block2(up2)
        
        up1 = self.upsample1(self.up_level2)
        self.up_level1 = self.up_block1(up1)
        
        return self.up_level1        