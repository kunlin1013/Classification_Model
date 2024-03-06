## AlexNet (2012) - the epoch-making ancestor of Deep Learning
- First demonstration that learned features are better than manually designed ones.

### Basic information
- paper name: [ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper_files/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html)
- author: Hinton (Teacher), Alex Krizhevsky (Student)
- ISLVRC (ImageNet Large Scale Visual Recognition Challenge) 2012 Champion :trophy:
- The classification accuracy has been improved from the traditional 70%+ to 80%+.

### Novelty
1. First use of **GPU** for accelerated network training.
2. Use the **Relu** for activation function. Traditionally, sigmoid or tanh are used as activation functions, but these functions can cause the gradient to vanish as the model depth increases.
3. Use the **Local Response Normalization (LRN)**.
4. Use **Dropout** in the first two layers of the fully connected layers to prevent overfitting.

### Suppress overfitting
1. Data Augmentation
   - After image translations and horizontal reflections, a 224x224 image is randomly cropped from the 256x256 image to serve as the image for training.
   - Use PCA to randomly alter the color channels.
2. Dropout
   - This will cause the training speed to slow down.

### Local Response Normalization (LRN)

![AlexNet](https://github.com/kunlin1013/Classification_Model/blob/main/AlexNet%20(2012)/img/Architecture.png)
