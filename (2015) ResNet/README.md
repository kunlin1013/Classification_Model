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

### Architecture

### Novelty
- Ultra-deep network structure (over 1000 layers)
- Propose the residual block
- Use **batch normalization** to accelerate training and discard Dropout
