import torch
import torch.nn as nn
import torch.nn.functional as F

class res_block(nn.Module):

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
        downsample = False,
    ):
        
        super(res_block, self).__init__()
    
        self.channel_in = channel_in
    
        if downsample:
            self.channel_out = 2 * channel_in
            self.conv1_stride = 2
            self.conv3_stride = 2
        else:
            self.channel_out = self.channel_in
            self.conv1_stride = 1
            self.conv3_stride = 1
        
        self.bn1 = nn.BatchNorm2d(num_features = self.channel_in)
    
        self.conv1 = nn.Conv2d(
            in_channels = self.channel_in,
            kernel_size = 3,
            out_channels = self.channel_out,
            stride = self.conv1_stride,
            padding = 1
        )
    
        self.bn2 = nn.BatchNorm2d(num_features = self.channel_out)
    
        self.conv2 = nn.Conv2d(
            in_channels = self.channel_out,
            out_channels = self.channel_out,
            kernel_size = 3,
            padding = 1
        )
    
        self.conv3 = nn.Conv2d(
            in_channels = self.channel_in,
            out_channels = self.channel_out,
            stride = self.conv3_stride,
            padding = 1,
            kernel_size = 3
        )
    
        self.bn3 = nn.BatchNorm2d(num_features = self.channel_out)
    
    
    
    '''
    x: tensor
    Input data
    This function forward the data through the block
    '''
    def forward(self,x):
        
        path = self.bn1(x)
        path = F.leaky_relu(path)
        path = F.dropout(path, p = 0.2)
        
        path = self.conv1(path)
        path = self.bn2(path)
        path = F.leaky_relu(path)
        path = F.dropout(path, p = 0.2)
        
        path = self.conv2(path)
        
        residual = self.conv3(x)
        residual = self.bn3(residual)
        
        output = path + residual
        
        return output