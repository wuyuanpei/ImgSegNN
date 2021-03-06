import torch
from torch.utils import data
import numpy as np

from loader import pascalVOCLoader
from loss import cross_entropy2d

'''
    Train the NN using trainning data
'''
class trainer():

    def __init__(
        self,
        optimizer,
        scheduler,
        criterion,
        net,
        local_path = "./VOC2012",
        rounds = 50,
        bs = 5,
        save_path = None,
        num_workers = 4,
        pin_memory = True,          
    ):

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.net = net
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.local_path = local_path
        self.rounds = rounds
        self.bs = bs
        self.save_path = save_path
        self.dst = pascalVOCLoader(root=local_path, split="train")
        self.trainloader = data.DataLoader(
                                        self.dst, 
                                        batch_size=bs,
                                        num_workers=num_workers,
                                        pin_memory=pin_memory)


    def train(self):
        print("Start Training")
        #convert it into GPU version
        self.net.to(self.device)

        self.net.train()

        for epoch in range(self.rounds):  # loop over the dataset multiple times

            # Update the learning rate
            if self.scheduler is not None:
                    self.scheduler.step(epoch)

            average_loss = 0.0
            for i, data in enumerate(self.trainloader, 0):
                #get the input
                inputs, labels = data

                #convert it into GPU version
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                #zero the autograder
                self.optimizer.zero_grad()

                #forward the inputs
                outputs = self.net(inputs) # outputs is batch_size*21*512*512 (one hot encoding)

                #calculate loss, backpropogate, step
                loss = self.criterion(outputs, labels) # labels is batch_size*512*512 (each entry is 0,1,..,or 20)
                #loss = cross_entropy2d(outputs, labels)
                #print(loss)
                
                loss.backward()
                self.optimizer.step()
                #print statistics
                average_loss += loss.item()
                # if i % 20 == 19:                      #print every 20*self.bs pictures
                #     print('[%d, %4d]\t%.5f' %
                #         (epoch + 1, (i + 1)*self.bs, running_loss / 20))
                #     running_loss = 0.0

            print('[%d]\t%.5f' %
                         (epoch + 1, average_loss / i))
                
        print('Finished Training')

        # Save the trained NN
        if not self.save_path is None:
            torch.save(self.net, self.save_path)
            print('Model Saved')









