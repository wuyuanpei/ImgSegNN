import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from decoder import decoder
from encoder import encoder
from in_block import in_block
from loader import pascalVOCLoader
from seg_out_block import seg_out_block
from seg_path import seg_path
from torch.utils import data
from trainer import trainer
from unet import unet
import sys


# Main function for training
def main(argv):
    # Wrong number of arguments
    if len(argv) != 10:
        usage()
        return

    # Read Model
    net = torch.load("./Models/"+argv[1]+".net")

    # Select objective function and optimization method
    if argv[3] == "CE":
        criterion = nn.CrossEntropyLoss()
    else:
        usage()
        return

    if argv[4] == "SGD":
        optimizer = optim.SGD(net.parameters(), lr=float(argv[5]))
    elif argv[4] == "Adam":
        optimizer = optim.Adam(net.parameters(), lr=float(argv[5]))
    else:
        usage()
        return

    # The path to save the trained NN
    save_path = "./Models/" + argv[2] + ".net"

    # Build the trainer
    t = trainer(
        net=net,
        criterion=criterion,
        optimizer=optimizer,
        save_path=save_path,
        rounds=int(argv[6]),
        bs=int(argv[7]),
        num_workers=int(argv[8]),
        pin_memory = (argv[9] == "True")
    )

    # Train
    t.train()

def usage():
    print("usage: python retrain.py fn_r fn_w loss opt lr epoch bs ths pin_m")
    print("\tfn_r:\tthe filename to read the model before training")
    print("\tfn_w:\tthe filename to save the model after training")
    print("\tloss:\tCE (for CrossEntropy)")
    print("\topt:\tSGD or Adam")
    print("\tlr:\tlearning rate")
    print("\tepoch:\tnumber of epochs")
    print("\tbs:\tbatch size (based on your GPU memory)")
    print("\tths:\tnumber of threads (based on your CPU)")
    print("\tpin_m:\tTrue/False to pin your memory for GPU")
    print("e.g.: python retrain.py NN1 NN2 CE Adam 0.001 50 5 4 True")

if __name__ == "__main__":
    main(sys.argv)