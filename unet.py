import torch.nn as nn
import torch
import torch.nn.functional as F

'''
    This file contains the original Unet and its modification
    Specifically:
    Number of layers can be adjusted, 4 or 3
    Activation function can be adjusted: ReLU, LeakyReLU
    Scale of channels can be adjusted
    Deconvolution and Batch-Normalization can be turned on or off
'''
class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, func):
        super(unetConv2, self).__init__()

        if is_batchnorm:
            if func == "R":
                self.conv1 = nn.Sequential(
                    nn.Conv2d(in_size, out_size, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(out_size), nn.ReLU()
                )
                self.conv2 = nn.Sequential(
                    nn.Conv2d(out_size, out_size, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(out_size), nn.ReLU()
                )
            elif func == "L":
                self.conv1 = nn.Sequential(
                    nn.Conv2d(in_size, out_size, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(out_size), nn.LeakyReLU()
                )
                self.conv2 = nn.Sequential(
                    nn.Conv2d(out_size, out_size, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(out_size), nn.LeakyReLU()
                )
        else:
            if func == "R":
                self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, kernel_size=3, stride=1, padding=1), nn.ReLU())
                self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size, kernel_size=3, stride=1, padding=1), nn.ReLU())
            elif func == "L":
                self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, kernel_size=3, stride=1, padding=1), nn.LeakyReLU())
                self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size, kernel_size=3, stride=1, padding=1), nn.LeakyReLU())

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


class unetUp(nn.Module):
    def __init__(self, in_size, out_size, is_deconv, func):
        super(unetUp, self).__init__()
        self.conv = unetConv2(in_size, out_size, False, func=func)
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        return self.conv(torch.cat([inputs1, outputs2], 1))

class unet(nn.Module):
    '''
        channel_scale: [16, 32, 64, 128, 256] * channel_scale is the number of channels in each level of unet
        n_classes: number of output classes
        is_deconv: use deconvolution in desampling part
        in_channels: number of input channels. 3 for RGB by default
        is_batchnorm: use batchnormalization
        layers: number of levels in the unet, 4 or 3
        func: activation function used in unet, "R" for ReLU, "L" for LeakyReLU
    '''
    def __init__(
        self, channel_scale=4, n_classes=21, is_deconv=True, in_channels=3, is_batchnorm=True, layers = 4, func = "R"
    ):
        super(unet, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.layers = layers

        channels = [16, 32, 64, 128, 256]
        channels = [int(x * channel_scale) for x in channels]
        # downsampling
        self.conv1 = unetConv2(self.in_channels, channels[0], self.is_batchnorm, func=func)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = unetConv2(channels[0], channels[1], self.is_batchnorm, func=func)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = unetConv2(channels[1], channels[2], self.is_batchnorm, func=func)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = unetConv2(channels[2], channels[3], self.is_batchnorm, func=func)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.center = unetConv2(channels[3], channels[4], self.is_batchnorm, func=func)

        # upsampling
        self.up_concat4 = unetUp(channels[4], channels[3], self.is_deconv, func=func)
        self.up_concat3 = unetUp(channels[3], channels[2], self.is_deconv, func=func)
        self.up_concat2 = unetUp(channels[2], channels[1], self.is_deconv, func=func)
        self.up_concat1 = unetUp(channels[1], channels[0], self.is_deconv, func=func)

        # final conv (without any concat)
        self.final = nn.Conv2d(channels[0], n_classes, 1)

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        

        if self.layers == 4:   
            maxpool4 = self.maxpool4(conv4) 
            center = self.center(maxpool4)
            up4 = self.up_concat4(conv4, center)
            up3 = self.up_concat3(conv3, up4)
        else:
            up3 = self.up_concat3(conv3, conv4)

        up2 = self.up_concat2(conv2, up3)
        up1 = self.up_concat1(conv1, up2)

        final = self.final(up1)

        return final