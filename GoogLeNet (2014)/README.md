## GoogLeNet (2014) 
- In GoogLeNet, the capital "L" is a tribute to LeNet.
- Ideal: Regardless of the size of the object within the image, Inception can extract features using different convolutional kernels.

### Basic information
- paper name: [Going deeper with convolutions](https://arxiv.org/abs/1409.4842)
- author: Christian Szegedy
- from: Google Inc
- ILSVRC (ImageNet Large Scale Visual Recognition Challenge) 2014 Classification Task first runner-up :1st_place_medal:

### Architecture


### Novelty
1. Introduce the **Inception structure** (integrating feature information of different scales).
2. Use **1x1 convolutional kernels**.
   - Why? Because the problem with the naive Inception is that as the number of channels increases, it leads to an explosion in computation.
   - Dimension reduction
   - Reduce the amount of parameters and computations
   - Increase model depth, enhancing the capability for non-linear expression
4. Add two auxiliary classifiers to aid in training.
   - As the network deepens, it may encounter the problem of **gradient vanishing.** Therefore, the idea is to use intermediate layers to assist in prediction.
   - The loss function during training : 
     -  $L = L_{final_layer} + 0.3 * L_{auxiliary_classifier1} + 0.3 * L_{auxiliary_classifier2}$
6. Discard the fully connected layers and use a global average pooling layer (significantly reducing the model parameters).

