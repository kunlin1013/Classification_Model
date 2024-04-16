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
- Use NAS (Neural Architecture Search) to search for a rational configuration of the three parameters: image input resolution, network depth, and channel width.
- Increasing the depth of the network can yield richer and more complex features, but excessive depth may lead to gradient vanishing and make training difficult.
- Increasing the width of the network can capture more fine-grained features and is easier to train, but networks with large widths and shallow depths often struggle to learn deeper-level features.
- Increasing the image resolution input to the network can potentially capture more fine-grained feature templates, but with very high input resolutions, the gain in accuracy diminishes. Additionally, high-resolution images increase the computational load.
  - The clearer the image, the more details can be seen.

![Scaling Up with Different Methods](https://github.com/kunlin1013/Classification_Model/blob/main/(2019)%20EfficientNet/img/Scaling%20Up%20with%20Different%20Methods.png)

**Scaling Up EfficientNet-B0 with Different Methods.**
