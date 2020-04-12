# ImgSegNN
Image Semantic Segmentation Neuron Networks built by Pytorch with CUDA and trained by PASCAL VOC 2012

## Introduction
Finish implemented:
|Models|Loss fn|Optimizer
|:---|:---|:---|
|U-Net|Cross Entropy|SGD|
|Residual U-Net||Adam|

## Usage
### Train
To train and save the model. Average loss during each epoch will be printed out in the format **[epoch number] loss**
```
usage: python train.py model fn loss opt lr epoch bs ths pin_m
    model:  unet or resunet
    fn:     the filename to save the model after training
    loss:   CE (for CrossEntropy)
    opt:    SGD or Adam
    lr:     learning rate
    epoch:  number of epochs
    bs:     batch size (based on your GPU memory)
    ths:    number of threads (based on your CPU)
    pin_m:  True/False to pin your memory for GPU
e.g.: python train.py unet NN1 CE Adam 0.001 50 5 4 True
```

### Re-train
To retrain a trained model and save it.
```
usage: python retrain.py fn_r fn_w loss opt lr epoch bs ths pin_m
    fn_r:   the filename to read the model before training
    fn_w:   the filename to save the model after training
    loss:   CE (for CrossEntropy)
    opt:    SGD or Adam
    lr:     learning rate
    epoch:  number of epochs
    bs:     batch size (based on your GPU memory)
    ths:    number of threads (based on your CPU)
    pin_m:  True/False to pin your memory for GPU
e.g.: python retrain.py NN1 NN2 CE Adam 0.001 50 5 4 True
```

### Sample
To sample several images with the model. Print out original images, ground truth, and the outputs of the model.
```
usage: python sample.py fn num shffl set
    fn:     the filename to save the model after training
    num:    number of pictures to print out
    shffl:  True/False to shuffle the image set
    set:    val or train, choose data set
e.g.: python sample.py NN1 2 True val
```

### Validate
To print out the validation error using validation set
```
usage: python validate.py fn loss
    fn:     the filename to save the model after training
    loss:   CE (for CrossEntropy)
e.g.: python validate.py NN1 CE
```

## Examples
|Name|Model|Loss|Opt|LR|Epoch|
|:---|:---|:---|:---|:---|:---|
|NN50epoch|Unet|CE|Adam|0.0005|50|
|NN150epoch|Unet|CE|Adam|0.0005|150|

```
python sample.py NN150epoch 2 True val
```
![NN150epochVal.png](https://github.com/wuyuanpei/ImgSegNN/raw/master/readme-imgs/NN150epochVal.png)

```
python sample.py NN50epoch 2 True train
```
![NN150epochVal.png](https://github.com/wuyuanpei/ImgSegNN/raw/master/readme-imgs/NN50epochTrain.png)
