## ResNeXt (2017) 

### Basic information
- paper name: [Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/abs/1611.05431)
- author: Saining Xie
- CVPR 2017

### Group Convolution
- The number of parameters can be reduced by 1/9
- When $g = C_{in}$ and $n = C_{in}$ , it will be equivalent to DW Conv.

![GConv](https://github.com/kunlin1013/Classification_Model/blob/main/(2017)%20ResNeXt/img/GConv.png)

### Architecture
**Left**: A block of ResNet

**Right**: A block of ResNeXt with cardinality = 32

![Block of ResNet and ResNeXt](https://github.com/kunlin1013/Classification_Model/blob/main/(2017)%20ResNeXt/img/Block%20of%20ResNet%20and%20ResNeXt.png)

**Left**: ResNet50

**Right**: ResNeXt-50

![Architecture of ResNet and ResNeXt](https://github.com/kunlin1013/Classification_Model/blob/main/(2017)%20ResNeXt/img/Architecture%20of%20ResNet%20and%20ResNeXt.png)

### Novelty
- ResNet combined with the structure of Inception => ResNeXt
- Can improve accuracy without increasing the number of parameters
