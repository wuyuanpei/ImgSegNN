# ImgSegNN
Image Semantic Segmentation Neuron Networks built by Pytorch with CUDA and trained by PASCAL VOC 2012

## Introduction
Finish implemented:
|Models|Loss fn|Optimizer
|:---|:---|:---|
|U-Net|Cross Entropy|SGD|
|Residual U-Net||Adam|

## Train
To train and save the model. Average loss during each epoch will be printed out in the format **[epoch number] loss**
```
usage: python train.py model fn loss opt lr epoch
    model:  unet or resunet
    fn:     the filename to save the model after training
    loss:   CE (for CrossEntropy)
    opt:    SGD or Adam
    lr:     learning rate
    epoch:  number of epochs
e.g.: python train.py unet NN1 CE Adam 0.001 50
```

## Sample
To sample several images with the model. Print out original images, ground truth, and the outputs of the model.
```
usage: python sample.py fn num shffl set
    fn:     the filename to save the model after training
    num:    number of pictures to print out
    shffl:  True/False to shuffle the image set
    set:    val or train, choose data set
e.g.: python sample.py NN1 2 True val
```

## Validate
To print out the validation error using validation set
```
usage: python validate.py fn loss
    fn:     the filename to save the model after training
    loss:   CE (for CrossEntropy)
e.g.: python validate.py NN1 CE
```
