# ImgSegNN
Image Semantic Segmentation Neural Networks built by Pytorch with CUDA and trained by PASCAL VOC 2012

## Introduction
Model trained with different combinations of structures, optimizers, learning rates, schedulers, number of epochs, etc
![NN041102Val.png](https://github.com/wuyuanpei/ImgSegNN/raw/master/Models/Experiments.png)

Training set includes 2913 images. Validation set includes 300 images. 20 classes of objects plus 1 layer of background.

![color-map.png](https://github.com/wuyuanpei/ImgSegNN/raw/master/readme-imgs/color-map.png)

Example of a trained U-Net:

![TrainedExample.png](https://github.com/wuyuanpei/ImgSegNN/raw/master/readme-imgs/TrainedExample.png)

## Usage
### Train
To train and save the model. Average loss during each epoch will be printed out in the format **[epoch number] loss**
```
usage: python train.py model fn loss opt lr epoch bs ths pin_m
        model:  resunet or unet or unet4Chann or unet3Layer or unetLeakyR
        fn:     the filename to save the model after training
        loss:   CE (for CrossEntropy)
        opt:    SGD or Adam or ASGD, can be with p1 or p2 (PLR)
        lr:     learning rate, or start LR in PLR
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
        fn:     the filename to open the model
        num:    number of pictures to print out
        shffl:  True/False to shuffle the image set
        set:    val or train, choose data set
e.g.: python sample.py NN1 2 True val
```

### Sample an Image
To sample an image outside using a model
```
usage:      python image_sampler.py fn path
    fn:     the filename to open the model
    path:   the path of the image
e.g.: python image_sampler.py NN1 ./a.png
```

### Validate
To print out the validation error using validation set
```
usage: python validate.py fn loss
        fn:     the filename to open the model
        loss:   CE (for CrossEntropy)
e.g.: python validate.py NN1 CE
```
### Examples
```
python sample.py NN041102 2 True val
```
![NN041102Val.png](https://github.com/wuyuanpei/ImgSegNN/raw/master/readme-imgs/NN041102Val.png)

```
python sample.py NN041101 2 True train
```
![NN041101Train.png](https://github.com/wuyuanpei/ImgSegNN/raw/master/readme-imgs/NN041101Train.png)

## Files
- unet.py: used to build U-Net
- decoder.py encoder.py in_block.py res_block.py seg_out_block.py seg_path.py: used to build Res U-Net
- torch_poly_lr_decay.py: the class that implements polynomial learning rate scheduler
- loader.py: the class that loads and does initial treatment to the data
- trainer.py: the class that trains the net input
- train.py: the main function that parses the command line and calls trainer.py
- retrain.py: the main function that parses the command line and calls trainer.py (used to retrain the model)
- sampler.py: the class that samples the model with val or train data set
- sample.py: the main function that parses the command line and call sampler.py
- image_sampler.py: the main function that parses the command line and call sampler.py with a single image specified
- validate.py: the class that validates the net
- validator.py: the main function that parses the command line and calls validator.py
