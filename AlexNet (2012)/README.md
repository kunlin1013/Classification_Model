## AlexNet (2012) - the epoch-making ancestor of Deep Learning
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
