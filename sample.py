from sampler import sampler
import torch
import sys

def main(argv):
    # Wrong number of arguments
    if len(argv) != 5:
        usage()
        return

    # To read a NN and validate
    net = torch.load("./Models/"+argv[1]+".net")

    # Build the sampler
    s = sampler(
            net = net,
            num_images = int(argv[2]),
            shuffle = (argv[3]=="True"),
            split = argv[4]
        )

    s.sample()



def usage():
    print("usage: python sample.py fn num shffl set")
    print("\tfn:\tthe filename to save the model after training")
    print("\tnum:\tnumber of pictures to print out")
    print("\tshffl:\tTrue/False to shuffle the image set")
    print("\tset:\tval or train, choose data set")
    print("e.g.: python sample.py NN1 2 True val")

if __name__ == "__main__":
    main(sys.argv)