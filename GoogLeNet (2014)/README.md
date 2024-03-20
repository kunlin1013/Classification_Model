## GoogLeNet (2014) 
- In GoogLeNet, the capital "L" is a tribute to LeNet.
- Ideal: Regardless of the size of the object within the image, Inception can extract features using different convolutional kernels.
![GoogLeNet](https://github.com/kunlin1013/Classification_Model/blob/main/GoogLeNet%20(2014)/img/GoogLeNet.png)

### Basic information
- paper name: [Going deeper with convolutions](https://arxiv.org/abs/1409.4842)
- author: Christian Szegedy
- from: Google Inc
- ILSVRC (ImageNet Large Scale Visual Recognition Challenge) 2014 Classification Task first runner-up :1st_place_medal: , Detection Task first runner-up :1st_place_medal:

### Hebbian principle
Currently, using deeper networks to enhance representation power and thereby improve accuracy results in an explosive increase in the number of parameters that need to be updated, leading to two serious problems.
1. When the dataset is incomplete, the network is more prone to overfitting.
2. A large number of parameters requiring updates leads to the need for substantial computational resources, resulting in high hardware costs.

One way to address the aforementioned issues is to establish a sparse neural network structure, which involves replacing fully connected layers with **sparse layers**.

When dealing with sparse networks, we can group neurons with high similarity in their activation patterns together by examining which neurons' activity patterns are similar to each other. This approach aligns with the Hebbian principle, which states "neurons that fire together, wire together."

However, this approach introduces another problem: hardware acceleration for sparse networks is quite unfriendly. Therefore, to implement sparsity while ensuring computational efficiency, Inception extracts features using kernels of different sizes and then merges them. This method allows features with high correlation to be grouped together (e.g., one group for 1x1, another for 3x3, and yet another for 5x5 kernels).

### Network In Network
GoogLeNet incorporates two concepts from Network In Network (NIN):
1. 1x1 Convolution
   - Dimension reduction
   - Reduce the amount of parameters and computations
   - Increase model depth, enhancing the capability for non-linear expression
2. Global Average Pooling (GAP)
   - Represent a channel with a single average value
   - Replace the fully connected layer to reduce the number of parameters
   - Improved the top-1 accuracy by 0.6%

### Architecture
![Architecture](https://github.com/kunlin1013/Classification_Model/blob/main/GoogLeNet%20(2014)/img/Architecture.png)
The **3x3 reduce refers** to the 1x1 convolution before the 3x3 convolution.
The **pool proj** corresponds to the 1x1 convolution after pooling.

![inception](https://github.com/kunlin1013/Classification_Model/blob/main/GoogLeNet%20(2014)/img/inception.png)
The right image is an improved version of the Inception module, using 1x1 convolution for dimensionality reduction.

### Novelty
1. Introduce the **Inception structure** (integrating feature information of different scales).
2. Use **1x1 convolutional kernels**.
   - Why? Because the problem with the naive Inception is that as the number of channels increases, it leads to an explosion in computation.
4. Add two auxiliary classifiers to aid in training.
   - As the network deepens, it may encounter the problem of **gradient vanishing.** Therefore, the idea is to use intermediate layers to assist in prediction.
   - The loss function during training : 
     -  $L = L_{final{\kern 3pt}layer} + 0.3 * L_{auxiliary{\kern 3pt}classifier1} + 0.3 * L_{auxiliary{\kern 3pt}classifier2}$
   - It will remove auxiliary classifiers in the testing part.
   - The auxiliary classifiers were proven to be of little use, so the authors removed them in the Inception v2/v3 paper.
6. Discard the fully connected layers and use a global average pooling layer (significantly reducing the model parameters).

