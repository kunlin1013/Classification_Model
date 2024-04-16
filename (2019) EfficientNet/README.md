# EfficientNet (2019)
### Basic information
- paper name: [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)
- author: Mingxing Tan
- from: Google
- EfficientNet-B7 achieved the highest accuracy of the year at 84.3% on ImageNet top-1. 
- Compared to the previous highest accuracy model, GPipe, EfficientNet-B7 has only 1/8.4 the number of parameters and a 6.1 times faster inference speed.
- It also explores the impact of **input resolution**, **network depth**, and **network width**.
![Model](https://github.com/kunlin1013/Classification_Model/blob/main/(2019)%20EfficientNet/img/Model.png)

### Architecture
![Block](https://github.com/kunlin1013/Classification_Model/blob/main/(2019)%20EfficientNet/img/Block.png)

- Width refers to the number of channels used in the feature matrix.
- The method proposed by EfficientNet involves simultaneously increasing the width, depth, and resolution.

![EfficientNet-B0 Architecture](https://github.com/kunlin1013/Classification_Model/blob/main/(2019)%20EfficientNet/img/EfficientNet-B0%20Architecture.png)

- In the table, it is assumed that after the convolutional layers, Batch Normalization (BN) and Swish activation functions are used by default.
- #Layers represents how many times the layers are repeated.

### Novelty
