## MobileNetV1, MobileNetV2, MobileNetV3 (2019) 

### Basic information
- paper name:
  - [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861)
  - [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)
  - [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244)
- author: Andrew G. Howard
- from: Google Inc.

### Identified problems
Previous models were enlarged at all costs for the sake of improving scores, making them impractical for real-time implementation on mobile devices. To address this issue, Google introduced MobileNet.

### Depthwise separable convolution => Depthwise conv + Pointwise conv => reduce parameters (MobileNetV1)
- Normal convolution => the convolutional layer has a total of 4 filters, each containing 3 kernels, with each kernel size being 3×3. Therefore, the number of parameters in the convolutional layer can be calculated using the following formula:
```
Parameters: 4 x 3 x 3 x 3 = 108
```
![Normal convolution](https://github.com/kunlin1013/Classification_Model/blob/main/(2019)%20MobileNetV3/img/Normal%20convolution.jpg)


### Architecture

### Novelty

