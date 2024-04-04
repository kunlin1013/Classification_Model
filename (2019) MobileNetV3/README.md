# MobileNetV1 (2017)
### Basic information
- paper name: [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861)
- author: Andrew G. Howard
- from: Google Inc.
- CVPR 2017

### Identified problems
Previous models were enlarged at all costs for the sake of improving scores, making them impractical for real-time implementation on mobile devices. To address this issue, Google introduced MobileNet.

### Depthwise separable convolution => Depthwise conv + Pointwise conv => reduce parameters (MobileNetV1)
- Normal Convolution
  - The convolutional layer has a total of 4 filters, each containing 3 kernels, with each kernel size being 3×3.
  - Therefore, the number of parameters in the convolutional layer can be calculated using the following formula:
  ```
  Parameters: 4 x 3 x 3 x 3 = 108
  ```
  ![Normal convolution](https://github.com/kunlin1013/Classification_Model/blob/main/(2019)%20MobileNetV3/img/Normal%20convolution.jpg)

- Separable Convolution = Depthwise Convolution + Pointwise Convolution
  - Depthwise Convolution
    - In Depthwise Convolution, one convolutional kernel is responsible for one channel, and each channel is convolved by only one kernel.
    - A filter contains only one 3×3 sized kernel. The number of parameters for the convolution part is calculated as follows:
    ```
    Parameters: 3 x 3 x 3 = 27
    ```
    ![Depthwise Convolution](https://github.com/kunlin1013/Classification_Model/blob/main/(2019)%20MobileNetV3/img/Depthwise%20Convolution.jpg)
    
  - Pointwise Convolution
    - The size of its convolutional kernel is 1×1×M, where M is the number of channels from the previous layer, and the number of convolutional kernels determines the number of output feature maps.
    - Since the 1×1 convolution method is used, the number of parameters can be calculated as follows:
    ```
    Parameters: 1 x 1 x 3 x 4 = 12
    ```
    ![Pointwise Convolution](https://github.com/kunlin1013/Classification_Model/blob/main/(2019)%20MobileNetV3/img/Pointwise%20Convolution.jpg)
    
- Compare parameters
  ```
  Normal Convolution: 108
  Depthwise separable convolution: 27 + 12 = 39
  ```
  Given the same input and obtaining 4 feature maps as the output, the number of parameters in depthwise separable convolution is approximately **1/3** of that in a normal convolution.

### Architecture
![MobileNetV1 Architecture](https://github.com/kunlin1013/Classification_Model/blob/main/(2019)%20MobileNetV3/img/MobileNetV1%20Architecture.jpg)

### Novelty
- Depthwise separable convolution => This significantly reduces the computational cost and the number of parameters.
- Increase the hyperparameters $\alpha$ and $\rho$
  - $\alpha$ : Width Multiplier, control the number of convolution kernels
  - $\rho$ : Resolution Multiplier, image size

# MobileNetV2 (2018)
### Basic information
- paper name: [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)
- author: Andrew G. Howard
- from: Google Inc.
- Compared to MobileNetV1, MobileNetV2 has **higher accuracy** and a **smaller model size**.
- CVPR 2018

### Identified problems
In MobileNetV1, many people found that after training, some of the parameters of the depthwise convolution kernels are zero.
=> The depthwise convolution did not function effectively.
=> MobileNetV2 will improve on this issue.

### Inverted Residuals
(a) Residual block
  - 1 x 1 convolution for dimension reduction
  - 3 x 3 convolution
  - 1 x 1 convolution for dimension increase

(b)   residual block
  - 1 x 1 convolution for dimension increase
  - 3 x 3 convolution
  - 1 x 1 convolution for dimension reduction
  - activation function: **Relu6** = $min(max(x, 0), 6)$
![Inverted Residuals](https://github.com/kunlin1013/Classification_Model/blob/main/(2019)%20MobileNetV3/img/Inverted%20Residuals.jpg)
![Bottleneck Block](https://github.com/kunlin1013/Classification_Model/blob/main/(2019)%20MobileNetV3/img/Bottleneck%20Block.png)

### Linear Bottlenecks
In Inverted Residuals, the last convolutional layer uses a **linear activation function** to avoid information loss.
![Mobilenet V2](https://github.com/kunlin1013/Classification_Model/blob/main/(2019)%20MobileNetV3/img/Mobilenet%20V2.jpg)
**A shortcut connection is only present when stride=1 and the shape of the input feature matrix is the same as the output feature matrix.**

### Architecture
![MobileNetV2 Architecture](https://github.com/kunlin1013/Classification_Model/blob/main/(2019)%20MobileNetV3/img/MobileNetV2%20Architecture.jpg)
- $t$ : expansion factor
- $c$ : output channels
- $n$ : the number of times the bottleneck is repeated.
- $s$ : stride (for the first layer, others are 1)

### Novelty
- Inverted Residuals
- Linear Bottlenecks

# MobileNetV3 (2019) 
### Basic information
- paper name: [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244)
- author: Andrew G. Howard
- from: Google Inc.
