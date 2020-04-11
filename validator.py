import torch
from torch.utils import data

from loader import pascalVOCLoader

'''
    Validate the NN using validation data
'''
class validator():

    def __init__(
        self,
        criterion,
        net,
        local_path = "./VOC2012",  
        bs = 5      
    ):

        self.criterion = criterion
        self.net = net
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.local_path = local_path
        self.bs = bs
        dst = pascalVOCLoader(root=local_path, split="val")
        self.valloader = data.DataLoader(dst, batch_size=bs)


    def validate(self):
        
        print("Start Validating")

        #convert it into GPU version
        self.net.to(self.device)

        running_loss = 0.0
        average_loss = 0.0

        for i, data in enumerate(self.valloader, 0):
            #get the input
            inputs, labels = data

            #convert it into GPU version
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            #forward the inputs
            outputs = self.net(inputs)

            #calculate loss
            loss = self.criterion(outputs, labels)
            
            #print statistics
            running_loss += loss.item()
            average_loss += loss.item()
            if i % 20 == 19:                      #print every 20*self.bs pictures
                print('[%4d]\t%.5f' %
                    ((i + 1)*self.bs, running_loss / 20))
                running_loss = 0.0
        
        
        print('Finished Validating')
        print('average: %.5f' % (average_loss / i))










