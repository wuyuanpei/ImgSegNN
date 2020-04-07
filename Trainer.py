import torch
from torch.utils import data

from loader import pascalVOCLoader

class trainer():

    def __init__(
        self,
        optimizer,
        criterion,
        net,
        local_path = "./VOC2012",
        rounds = 2,
        bs = 4,
        save_path = None          
    ):

        self.optimizer = optimizer
        self.criterion = criterion
        self.net = net
        self.local_path = local_path
        self.rounds = rounds
        self.bs = bs
        self.save_path = save_path
        dst = pascalVOCLoader(root=local_path, split="train")
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
                if i % 20 == 19:                      #print every 20*self.bs pictures
                    print('[%d, %5d] loss: %.3f' %
                        (epoch + 1, (i + 1)*self.bs, running_loss / (20*self.bs)))
                    running_loss = 0.0
                
        print('Finished Training')

        # Save the trained NN
        if not self.save_path is None:
            torch.save(self.net, self.save_path)
            print('Model Saved')









