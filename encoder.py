import torch
import torch.nn as nn
import torch.nn.functional as F

from res_block import res_block

class encoder(nn.Module):

    '''
    base_filters: int
        Number of channels at the beginning
    '''
    def __init__(
        self,
        base_filters=16
    ):
        
        super(encoder, self).__init__()
        
        self.bf = base_filters
        
        self.down_block2 = res_block(
            channel_in = self.bf,
            downsample = True
        )
        
        self.down_block3 = res_block(
            channel_in = self.bf * 2,
            downsample = True
        )
        
        self.down_block4 = res_block(
            channel_in = self.bf * 4,
            downsample = True
        )
        
        self.down_bridge = res_block(
            channel_in = self.bf * 8,
            downsample = True
        )
        
    def forward(self,x):
        
        self.down_level2 = self.down_block2(x)
        self.down_level3 = self.down_block3(self.down_level2)
        self.down_level4 = self.down_block4(self.down_level3)
        self.codes = self.down_bridge(self.down_level4)
        
        return self.codes