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

# Different Blocks of the NN
inNet = in_block()
encoderNet = encoder()
decoderNet = decoder()
outNet = seg_out_block()

# Whole NN
net = seg_path(
        in_block=inNet,
        encoder=encoderNet,
        decoder=decoderNet,
        seg_out_block=outNet)

# Objective function and optimization method
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.001)

# The path to save the trained NN
save_path = "./Models/NN1.net"

# Build the trainer
t = trainer(
        net = net,
        criterion = criterion,
        optimizer = optimizer,
        save_path = save_path
    )

# Train
t.train()


# # To read a NN and test
# model = torch.load("./Models/NN1.net")
# model.eval()

# local_path = "./VOC2012" # Root directory of VOC2012
# bs = 4
# dst = pascalVOCLoader(root=local_path, split="train")
# trainloader = data.DataLoader(dst, batch_size=bs)
# print(dst.__len__())
# for i, data in enumerate(trainloader):
#     imgs, labels = data
#     print(imgs.size())
#     print(imgs[0][0][200])
#     print(labels.size())
#     print(labels[0][0][200])
#     outputs = model(imgs)
#     print(outputs.size())
#     print(outputs[0][0][200])
#     break; # ONLY FOR TESTING

