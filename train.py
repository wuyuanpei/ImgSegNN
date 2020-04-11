from torch.utils import data
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from loader import pascalVOCLoader 
from in_block import in_block
from encoder import encoder
from decoder import decoder
from seg_out_block import seg_out_block
from seg_path import seg_path

from trainer import trainer

from unet import unet

# # Different Blocks of the NN
# inNet = in_block()
# encoderNet = encoder()
# decoderNet = decoder()
# outNet = seg_out_block()

# # Whole NN
# net = seg_path(
#         in_block=inNet,
#         encoder=encoderNet,
#         decoder=decoderNet,
#         seg_out_block=outNet)
net = unet()

# Objective function and optimization method
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001)

# The path to save the trained NN
save_path = "./Models/UNet.net"

# Build the trainer
t = trainer(
        net = net,
        criterion = criterion,
        optimizer = optimizer,
        save_path = save_path,
        rounds = 2
    )

# Train
t.train()

