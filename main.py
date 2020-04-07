from torch.utils import data
import matplotlib.pyplot as plt

from loader import pascalVOCLoader 
from in_block import in_block
from encoder import encoder
from decoder import decoder
from seg_out_block import seg_out_block
from seg_path import seg_path

# Different Blocks of the NN
inNet = in_block()
encoderNet = encoder()
decoderNet = decoder()
outNet = seg_out_block()

# Whole NN
net = seg_path(
        in_block=inNet,
        encoder=encoderNet,
        decoder=decoderNet,
        seg_out_block=outNet)


local_path = "./VOC2012" # Root directory of VOC2012
bs = 1
dst = pascalVOCLoader(root=local_path)
trainloader = data.DataLoader(dst, batch_size=bs)

for i, data in enumerate(trainloader):
    imgs, labels = data
    print(imgs.size())
    output = net(imgs)
    print(output.size())
    #plt.imshow(labels[0])
    #plt.show()
    break # Only one loop
