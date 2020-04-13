import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
from PIL import Image
from torchvision import transforms

'''
    This module test a given image given an input image and a NN
'''
def main(argv):
    # Wrong number of arguments
    if len(argv) != 3:
        usage()
        return

    # To read a NN
    net = torch.load("./Models/"+argv[1]+".net")

    # To open an image
    image = Image.open(argv[2])
    image = image.resize((512,512),Image.NEAREST)
    image = transforms.Compose(
                [
                    transforms.ToTensor(),
                ]
            )(image)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # convert it into GPU version
    net.to(device)

    net.eval()

    # Handle both 4 bytes and 3 bytes images
    image = image[0:3].view(1,3,512,512).to(device)
    
    outputs = net(image)

    # Map back from one-hot encoding to index in each entry
    [_, indices] = torch.max(outputs, 1)
    outputs = decode_segmap(
                label_mask = indices[0].cpu().numpy(),
                plot = False)

    # Normalize back and print image
    inputs = image.squeeze()
    inputs = inputs.cpu().transpose(0, 2).transpose(0, 1)


    plt.subplot(1, 2, 1)
    plt.imshow(inputs)
    plt.subplot(1, 2, 2)
    plt.imshow(outputs)
    plt.show()


def get_pascal_labels():
    """Load the mapping that associates pascal classes with label colors
    Returns:
        np.ndarray with dimensions (21, 3)
    """
    return np.asarray(
            [
                [0, 0, 0],
                [128, 0, 0],
                [0, 128, 0],
                [128, 128, 0],
                [0, 0, 128],
                [128, 0, 128],
                [0, 128, 128],
                [128, 128, 128],
                [64, 0, 0],
                [192, 0, 0],
                [64, 128, 0],
                [192, 128, 0],
                [64, 0, 128],
                [192, 0, 128],
                [64, 128, 128],
                [192, 128, 128],
                [0, 64, 0],
                [128, 64, 0],
                [0, 192, 0],
                [128, 192, 0],
                [0, 64, 128],
            ]
        )

def decode_segmap(label_mask, plot=False):
        """Decode segmentation class labels into a color image
        Args:
            label_mask (np.ndarray): an (M,N) array of integer values denoting
              the class label at each spatial location.
            plot (bool, optional): whether to show the resulting color image
              in a figure.
        Returns:
            (np.ndarray, optional): the resulting decoded color image.
        """
        label_colours = get_pascal_labels()
        r = label_mask.copy()
        g = label_mask.copy()
        b = label_mask.copy()
        for ll in range(0, 20):
            r[label_mask == ll] = label_colours[ll, 0]
            g[label_mask == ll] = label_colours[ll, 1]
            b[label_mask == ll] = label_colours[ll, 2]
        rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        if plot:
            plt.imshow(rgb)
            plt.show()
        else:
            return rgb


def usage():
    print("usage: python image_sampler.py fn path")
    print("\tfn:\tthe filename to open the model")
    print("\tpath:\tthe path of the image")
    print("e.g.: python image_sampler.py NN1 ./a.png")


if __name__ == "__main__":
    main(sys.argv)