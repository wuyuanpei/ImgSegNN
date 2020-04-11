import torch
from torch.utils import data
import torchvision
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
        shuffle = "True",
        local_path = "./VOC2012",
        num_images = 1,
        split = "val"     
    ):

        self.net = net
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.local_path = local_path
        self.dst = pascalVOCLoader(root=local_path, split=split)
        self.sampleloader = data.DataLoader(self.dst, batch_size=1, shuffle=shuffle)
        self.num_images = num_images


    def sample(self):

        #convert it into GPU version
        self.net.to(self.device)

        self.net.eval()  

        images_set = []
        for i, data in enumerate(self.sampleloader, 0):
            #get the input
            inputs, labels = data

            #convert it into GPU version
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            # #forward the inputs
            outputs = self.net(inputs)

            # Map back from one-hot encoding to index in each entry
            [_,indices] = torch.max(outputs,1)
            outputs = self.dst.decode_segmap(
                label_mask = indices[0].cpu().numpy(),
                plot = False)

            labels = self.dst.decode_segmap(
                label_mask = labels[0].cpu().numpy(),
                plot = False)
            
            # Normalize back and print image
            inputs = inputs.squeeze()
            inputs[0] = inputs[0]*0.229+0.485
            inputs[1] = inputs[1]*0.224+0.456
            inputs[2] = inputs[2]*0.225+0.406
            inputs = inputs.cpu().transpose(0,2).transpose(0,1)

            images = [inputs, labels, outputs]
            images_set.append(images)
            if i == self.num_images - 1:
                break
        
        # Draw images
        for i in range(self.num_images):
            plt.subplot(self.num_images,3,i * 3 + 1)
            plt.imshow(images_set[i][0])
            plt.subplot(self.num_images,3,i * 3 + 2)
            plt.imshow(images_set[i][1])
            plt.subplot(self.num_images,3,i * 3 + 3)
            plt.imshow(images_set[i][2])
        plt.show()