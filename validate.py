from torch.utils import data
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import sys

from validator import validator

def main(argv):
    if len(argv) != 3:
        usage()
        return
    
    # To read a NN and validate
    net = torch.load("./Models/"+ argv[1] +".net")
    net.eval()

    # Objective function and optimization method
    if argv[2] == "CE":
        criterion = nn.CrossEntropyLoss()
    else:
        usage()
        return

    # Build the validator
    v = validator(
            net = net,
            criterion = criterion
        )

    v.validate()


def usage():
    print("usage: python validate.py fn loss")
    print("\tfn:\tthe filename to save the model after training")
    print("\tloss:\tCE (for CrossEntropy)")
    print("e.g.: python validate.py NN1 CE")

if __name__ == "__main__":
    main(sys.argv)
