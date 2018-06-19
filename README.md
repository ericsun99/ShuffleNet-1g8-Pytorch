ShuffleNet-1g8-Pytorch
Introduction

This is a Pytorch implementation of faceplusplus's ShuffleNet-1g8. For details, please read the following papers:
    ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices


Pretrained Models on ImageNet

We provide pretrained ShuffleNet-1g8 models on ImageNet, which achieve nearly accuracy with the original ones reported in the paper.

The top-1/5 accuracy rates by using single center crop (crop size: 224x224, image size: 256xN):
Network 	      Top-1 	Top-5	  Top-1(reported in the paper)
ShuffleNet-1g8 	  67.408 	87.258	  67.60


Evaluate Models
python eval.py -a shufflenet --evaluate ./ShuffleNet_1g8_Top1_67.408_Top5_87.258.pth.tar ./ILSVRC2012/ 

Dataset prepare
Refer to https://github.com/facebook/fb.resnet.torch/blob/master/INSTALL.md#download-the-imagenet-dataset
