import torch
import torch.nn as nn
import torch.nn.functional as F

class in_block(nn.Module):
    
    '''
    channel_in: int
        Number of input channels
    downsample: boolean
        True if the block is used for encoding (downsample)
        number of channels will be doubled
        False for decoding (upsample)
        number of channels will keep the same
    This function initializes the components of the block
    '''
    def __init__(
        self,
        channel_in,
        channel_out,
    ):
        super(in_block, self).__init__()
        
        self.channel_in = channel_in
        self.channel_out = channel_out
        
        self.conv1 = nn.Conv2d(
            kernel_size = 3,
            in_channels = self.channel_in,
            out_channels = self.channel_out,
            padding = 1,
        )
        
        self.bn1 = nn.BatchNorm2d(num_features=self.channel_out)
        
        self.conv2 = nn.Conv2d(
            kernel_size = 3,
            padding = 1,
            in_channels = self.channel_in,
            out_channels = self.channel_out,
        )
        
        self.conv3 = nn.Conv2d(
            kernel_size = 3,
            padding = 1,
            in_channels = self.channel_in,
            out_channels = self.channel_out,
        )
    
        self.bn3 = nn.BatchNorm2d(num_features = self.channel_out)
        
    '''
    x: tensor
    Input data
    This function forward the data through the block
    '''    
    def forward(self,x):
        
        path = self.conv1(x)
        path = self.bn1(path)
        path = F.leaky_relu(path)
        path = F.dropout(path, p = 0.2)
        
        path = self.conv2(path)
        
        residual = self.conv3(x)
        residual = self.bn3(residual)
        
        self.down_level1 = path + residual
        
        return self.down_level1
    