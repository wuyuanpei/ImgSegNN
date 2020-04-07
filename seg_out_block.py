import torch
import torch.nn as nn
import torch.nn.functional as F

class seg_out_block(nn.Module):
    
    def __init__(
        self,
        base_filters = 16,
        n_classes = 21
    ):
        
        super(seg_out_block, self).__init__()
        
        self.bf = base_filters
        self.n_classes = n_classes
        self.conv = nn.Conv2d(
            in_channels = self.bf,
            out_channels = self.n_classes,
            kernel_size = 1
        )
        
    def forward(self,x):
        self.output = self.conv(x)
        return self.output