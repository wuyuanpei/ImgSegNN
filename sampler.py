import torch
from torch.utils import data
import numpy as np
import matplotlib.pyplot as plt

from loader import pascalVOCLoader

'''
    Take sample (from the validation set) and draw the image
'''
class sampler():

    def __init__(
        self,
        net,
        local_path = "./VOC2012",
        num_images = 1,
        shuffle = True      
    ):

        self.net = net
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.local_path = local_path
        self.dst = pascalVOCLoader(root=local_path, split="val")
        self.sampleloader = data.DataLoader(self.dst, batch_size=1, shuffle=shuffle)
        self.num_images = num_images


    def sample(self):

        #convert it into GPU version
        self.net.to(self.device)  

        for i, data in enumerate(self.sampleloader, 0):
            #get the input
            inputs, labels = data

            #convert it into GPU version
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            #forward the inputs
            outputs = self.net(inputs)
            
            inputs = inputs.cpu().numpy()
            
            inputs = np.transpose(inputs, [0,2,3,1])

            # Map back from one-hot encoding
            [_,indices] = torch.max(outputs,1)

            self.dst.decode_segmap(
                label_mask = indices[0].cpu().numpy(),
                plot = True)

            [_,indices] = torch.max(labels,1)

            self.dst.decode_segmap(
                label_mask = indices[0].cpu().numpy(),
                plot = True)

            
            if i == self.num_images - 1:
                break



# To read a NN and validate
net = torch.load("./Models/NN1.net")
net.eval()


# Build the sampler
s = sampler(
        net = net
    )

s.sample()





