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
from torch_poly_lr_decay import PolynomialLRDecay


# Main function for training
def main(argv):
    # Wrong number of arguments
    if len(argv) != 10:
        usage()
        return

    # Select Model
    if argv[1] == "resunet":
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
    elif argv[1] == "unet":
        net = unet(channel_scale=1)
    elif argv[1] == "unet4Chann":
        net = unet()
    elif argv[1] == "unetLeakyR":
        net = unet(func = "L")
    elif argv[1] == "unet3Layer":
        net = unet(layers = 3)
    elif argv[1] == "unet2Chann":
        net = unet(channel_scale=2)
    else:
        usage()
        return

    # Select objective function and optimization method
    if argv[3] == "CE":
        criterion = nn.CrossEntropyLoss()
    else:
        usage()
        return

    scheduler = None
    if argv[4] == "SGD":
        optimizer = optim.SGD(net.parameters(), lr=float(argv[5]))
    elif argv[4] == "Adam":
        optimizer = optim.Adam(net.parameters(), lr=float(argv[5]))
    elif argv[4] == "ASGD":
        optimizer = optim.ASGD(net.parameters(), lr=float(argv[5]))
    elif argv[4] == "SGDp1":
        optimizer = optim.ASGD(net.parameters(), lr=float(argv[5]))
        scheduler = PolynomialLRDecay(optimizer, max_decay_steps=int(argv[6]), end_learning_rate=0.0001, power=1.0)
    elif argv[4] == "SGDp2":
        optimizer = optim.ASGD(net.parameters(), lr=float(argv[5]))
        scheduler = PolynomialLRDecay(optimizer, max_decay_steps=int(argv[6]), end_learning_rate=0.0001, power=2.0)
    elif argv[4] == "Adamp1":
        optimizer = optim.Adam(net.parameters(), lr=float(argv[5]))
        scheduler = PolynomialLRDecay(optimizer, max_decay_steps=int(argv[6]), end_learning_rate=0.0001, power=1.0)
    elif argv[4] == "Adamp2":
        optimizer = optim.Adam(net.parameters(), lr=float(argv[5]))
        scheduler = PolynomialLRDecay(optimizer, max_decay_steps=int(argv[6]), end_learning_rate=0.0001, power=2.0)
    elif argv[4] == "ASGDp1":
        optimizer = optim.ASGD(net.parameters(), lr=float(argv[5]))
        scheduler = PolynomialLRDecay(optimizer, max_decay_steps=int(argv[6]), end_learning_rate=0.0001, power=1.0)
    elif argv[4] == "ASGDp2":
        optimizer = optim.ASGD(net.parameters(), lr=float(argv[5]))
        scheduler = PolynomialLRDecay(optimizer, max_decay_steps=int(argv[6]), end_learning_rate=0.0001, power=2.0)
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
        scheduler=scheduler,
        save_path=save_path,
        rounds=int(argv[6]),
        bs=int(argv[7]),
        num_workers=int(argv[8]),
        pin_memory = (argv[9] == "True")
    )

    # Train
    t.train()

def usage():
    print("usage: python train.py model fn loss opt lr epoch bs ths pin_m")
    print("\tmodel:\tresunet or unet or unet4Chann or unet3Layer or unetLeakyR")
    print("\tfn:\tthe filename to save the model after training")
    print("\tloss:\tCE (for CrossEntropy)")
    print("\topt:\tSGD or Adam or ASGD, can be with p1 or p2 (PLR)")
    print("\tlr:\tlearning rate, or start LR in PLR")
    print("\tepoch:\tnumber of epochs")
    print("\tbs:\tbatch size (based on your GPU memory)")
    print("\tths:\tnumber of threads (based on your CPU)")
    print("\tpin_m:\tTrue/False to pin your memory for GPU")
    print("e.g.: python train.py unet NN1 CE Adam 0.001 50 5 4 True")

if __name__ == "__main__":
    main(sys.argv)