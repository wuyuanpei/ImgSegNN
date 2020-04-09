from torch.utils import data
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from validator import validator

# To read a NN and validate
net = torch.load("./Models/NN1.net")
net.eval()

# Objective function and optimization method
criterion = nn.MSELoss()

# Build the validator
v = validator(
        net = net,
        criterion = criterion
    )

v.validate()
