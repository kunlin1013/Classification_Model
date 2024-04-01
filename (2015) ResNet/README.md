## ResNet (2015) 

### Basic information
- paper name: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- author: Kaiming He
- from: Microsoft Research
- ILSVRC (ImageNet Large Scale Visual Recognition Challenge) 2015 Classification, localization, detection Task first runner-up :1st_place_medal:
- Coco (Common Objects in Context) 2015 detection, segmentation Task first runner-up :1st_place_medal:

### Identified problems
- Found that a deeper network does not necessarily lead to better performance, which can lead to the following two issues:
  - vanishing/exploding gradients
  - model degradation
![degradation](https://github.com/kunlin1013/Classification_Model/blob/main/(2015)%20ResNet/img/degradation.png)
As the depth increases, accuracy reaches saturation and then rapidly declines.

### Architecture
![residual block](https://github.com/kunlin1013/Classification_Model/blob/main/(2015)%20ResNet/img/residual%20block.png)

Assuming input $x$ passes through two layers, the feature we expect it to learn is $H(x)$ (desired underlying mapping).
In reality, after passing through two layers, $x$ may not fully learn $H(x)$. We refer to the gap between the ideal and the actual learned outcome as the Residual.

$$
Residual = H(x) - x
$$

We refer to the Residual as $F(x)$ (residual mapping).

$$
F(x) = H(x) - x
$$

$$
H(x) = F(x) + x
$$

So now, the goal of the model has become to **learn the residual mapping**.

![architecture](https://github.com/kunlin1013/Classification_Model/blob/main/(2015)%20ResNet/img/architecture.png)
Different layers of ResNet architecture.

### Novelty
- Ultra-deep network structure (over 1000 layers)
- Propose the residual block
- Use **batch normalization** to accelerate training and discard Dropout
  - after each convolution and before activation
  - If a BN (Batch Normalization) layer is added after the convolutional layer, then the bias is not needed

### Residual network
1. Easy to optimize
2. Solve the degradation problem
3. The network can be very deep, significantly improving accuracy
4. The shortcut connection neither introduces additional parameters nor increases computational complexity
