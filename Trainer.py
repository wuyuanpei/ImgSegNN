import torch
import torch.nn as nn
import torch.nn.functional

from torch.utils import data
import matplotlib.pyplot as plt

from loader import pascalVOCLoader

class trainer():

    def __init__(
        self,
        optimizer,
        criterion,
        net,
        local_path,
        rounds = 2,
        bs = 4            
    ):

        self.optimizer = optimizer
        self.criterion = criterion
        self.net = net
        self.local_path = local_path
        self.rounds = rounds
        self.bs = bs

        dst = pascalVOCLoader(root=local_path)
        self.trainloader = data.DataLoader(dst, batch_size=bs)


    def train(self):
        
        for epoch in range(self.rounds):  # loop over the dataset multiple times

            running_loss = 0.0
            for i, data in enumerate(self.trainloader, 0):
                #get the input
                inputs, labels = data

                #zero the autograder
                self.optimizer.zero_grad()

                #forward the inputs
                outputs = self.net(inputs)

                #calculate loss, backpropogate, step
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                #print statistics
                running_loss += loss.item()
                if i % 200 == 199:                      #print every 200 pictures
                    print('[%d, %5d] loss: %.3f' %
                        (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0

        print('Finished Training')









