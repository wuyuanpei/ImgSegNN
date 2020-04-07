import torch
import torch.nn as nn
import torch.nn.functional as F

from res_block import res_block
from seg_out_block import seg_out_block
from in_block import in_block
from encoder import encoder
from decoder import decoder

class seg_path(nn.Module):
    
    def __init__(
        self, 
        in_block,
        encoder,
        decoder,
        seg_out_block
    ):
        super(seg_path, self).__init__()
        
        self.in_block = in_block
        self.encoder = encoder
        self.decoder = decoder
        self.seg_out_block = seg_out_block
        
    def forward(self, x):
        
        self.down_level1 = self.in_block(x)
        
        self.down_level2 = self.encoder.down_block2(self.down_level1)
        self.down_level3 = self.encoder.down_block3(self.down_level2)
        self.down_level4 = self.encoder.down_block4(self.down_level3)
        self.codes = self.encoder.down_bridge(self.down_level4)
        
        self.up4 = self.decoder.upsample4(self.codes)
        up4_dummy = torch.cat([self.up4, self.down_level4],1)
        up4_dummy = self.decoder.conv4(up4_dummy)
        self.up_level4 = self.decoder.up_block4(up4_dummy)
        
        self.up3 = self.decoder.upsample3(self.up_level4)
        up3_dummy = torch.cat([self.up3, self.down_level3],1)
        up3_dummy = self.decoder.conv3(up3_dummy)
        self.up_level3 = self.decoder.up_block3(up3_dummy)

        self.up2 = self.decoder.upsample2(self.up_level3)
        up2_dummy = torch.cat([self.up2, self.down_level2],1)
        up2_dummy = self.decoder.conv2(up2_dummy)
        self.up_level2 = self.decoder.up_block2(up2_dummy)    
        
        self.up1 = self.decoder.upsample1(self.up_level2)
        up1_dummy = torch.cat([self.up1, self.down_level1], 1)
        up1_dummy = self.decoder.conv1(up1_dummy)
        self.up_level1 = self.decoder.up_block1(up1_dummy)
        
        self.output = self.seg_out_block(self.up_level1)
        
        return self.output