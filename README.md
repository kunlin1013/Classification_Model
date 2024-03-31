## Implement classification tasks with various models

### Dataset link
[Flower classification](https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz)

- Briefly describe: Need to use **.\lib\split_data.py** to split data.

### Data augmentation
- ratio: 0.3
- Type: Affine, PerspectiveTransform, AddToHueAndSaturation, Flipud, GaussianBlur, GaussianNoise, SpeckleNoise

### Parameter settings
- Batch size: 32
- Learning rate: 0.0001
- Learning rate decay factor: 0.5
- Learning rate decay patience: 3
- Early stopping patience: 8
- Epoches: 100

### Model performance
|    Model   | early stopping  epoch |   train acc (%)   |   val acc (%)   |   test acc (%)   |
|   :----:   |       :----:          |      :----:       |     :----:      |      :----:      |
|  GoogLeNet |         32            |       83.99       |     78.39       |                  |  
 






