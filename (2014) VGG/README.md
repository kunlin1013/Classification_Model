## VGG (2014) 
- Since the model depth of AlexNet is greater than that of LeNet, leading to improved accuracy, the idea behind VGG is to further increase the network depth.

![VGG](https://github.com/kunlin1013/Classification_Model/blob/main/(2014)%20VGG/img/VGG.png)

### Basic information
- paper name: [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
- author: Andrew Zisserman (Teacher), Karen Simonyan (Student)
- from: Visual Geometry Group, Department of Engineering Science, University of Oxford
- ILSVRC (ImageNet Large Scale Visual Recognition Challenge) 2014 Localization Task Champion :1st_place_medal:, Classification Task first runner-up :2nd_place_medal:

### Architecture
- D is VGG16, and E is VGG19.
![Architecture](https://github.com/kunlin1013/Classification_Model/blob/main/(2014)%20VGG/img/Architecture.png)

### Novelty
1. By **stacking multiple 3x3** convolutional kernels instead of using large-sized convolutional kernels => the approach reduces the number of required parameters
   - e.g. two 3x3 convolutional kernels can be stacked to replace a single 5x5 convolutional kernel, and stacking three 3x3 convolutional kernels can replace a 7x7 kernel (providing the **same receptive field**)
2. It was discovered that LRN (Local Response Normalization) **does not enhance performance** but rather increases computational complexity.
3. Replacing the two 3x3 Conv layers in Model B with a single 5x5 Conv (having the same receptive field) resulted in a 7% increase in the top-1 error rate. **This confirms that deep networks with smaller filters perform better than shallower networks with larger filters.**

#### After the ILSVRC competition, further experiments were conducted with VGG, and it was observed that VGG outperformed GoogLeNet in terms of the effectiveness of a single model.
![error](https://github.com/kunlin1013/Classification_Model/blob/main/(2014)%20VGG/img/error.png)

#### While it appears that smaller and deeper models are preferable, by the time of VGG19, the depth was nearing saturation due to `Gradient Vanishing`. Consequently, ResNet (2015) proposed a solution to address this issue.
