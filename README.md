## Implement classification tasks with various models

### Dataset link
[Flower classification](https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz)

- Briefly describe: Need to use **.\lib\split_data.py** to split data.

### Data augmentation
- ratio: 0.8
- Type: Affine, PerspectiveTransform, AddToHueAndSaturation, Flipud, GaussianBlur, GaussianNoise, SpeckleNoise

### Parameter settings
- Batch size: 32
- Learning rate: 0.0001
- Learning rate decay factor: 0.5
- Learning rate decay patience: 3
- Early stopping patience: 8
- Epoches: 100

### Model performance
|     Model   | Parameters  | early stopping  epoch |   train acc (%)   |   val acc (%)   |   test acc (%)   |
|    :----:   |   :----:    |       :----:          |      :----:       |     :----:      |      :----:      |
|    AlexNet  |   58.30M    |         34            |       86.09       |     76.33       |      79.84       |  
|      VGG    |  139.59M    |         31            |       71.87       |     66.35       |      65.59       |  
|   GoogLeNet |    5.98M    |         77            |       80.48       |     76.74       |      77.96       |  
|    ResNet18 |   11.18M    |         46            |       88.43       |     78.66       |      80.91       |  
| MobileNetV2 |    2.23M    |         62            |       73.16       |      69.9       |      73.12       | 
| MobileNetV3 |    4.21M    |         47            |       76.04       |     74.42       |      76.61       | 
 






