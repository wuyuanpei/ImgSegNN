from torch.utils import data
import matplotlib.pyplot as plt

from loader import pascalVOCLoader 
from encoder import encoder

testNet = encoder(base_filters=3) # test NN

local_path = "./VOC2012" # Root directory of VOC2012
bs = 10
dst = pascalVOCLoader(root=local_path)
trainloader = data.DataLoader(dst, batch_size=bs)

for i, data in enumerate(trainloader):
    imgs, labels = data
    print(imgs.size())
    output = testNet(imgs)
    print(output.size())
    #plt.imshow(labels[0])
    #plt.show()
    break; # Only one loop
