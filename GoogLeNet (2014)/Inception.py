from tensorflow.keras import layers, Sequential

class Inception(layers.Layer):
    # In the GoogLeNet paper, 'red' means 'reduction'
    def __init__(self, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj, **kwargs):
        super(Inception, self).__init__(**kwargs)
        
        # Single 1x1 convolution layer
        self.branch1 = layers.Conv2D(ch1x1, kernel_size=1, activation="relu")

        # A 1x1 convolution layer followed by a 3x3 convolution layer, 
        # using padding='SAME' to maintain the output size unchanged
        self.branch2 = Sequential([layers.Conv2D(ch3x3red, kernel_size=1, activation="relu"),
                                   layers.Conv2D(ch3x3, kernel_size=3, padding="SAME", activation="relu")])         # output_size= input_size

        # A 1x1 convolution layer followed by a 5x5 convolution layer, 
        # also using padding='SAME'
        self.branch3 = Sequential([layers.Conv2D(ch5x5red, kernel_size=1, activation="relu"),
                                   layers.Conv2D(ch5x5, kernel_size=5, padding="SAME", activation="relu")])         # output_size= input_size

        # A max pooling layer followed by a 1x1 convolution layer, 
        # using padding='SAME' and strides=1 to maintain the output size
        self.branch4 = Sequential([layers.MaxPool2D(pool_size=3, strides=1, padding="SAME"),                     
                                   layers.Conv2D(pool_proj, kernel_size=1, activation="relu")])                     # output_size= input_size

    def call(self, inputs, **kwargs):
        branch1 = self.branch1(inputs)
        branch2 = self.branch2(inputs)
        branch3 = self.branch3(inputs)
        branch4 = self.branch4(inputs)
        outputs = layers.concatenate([branch1, branch2, branch3, branch4])                                          # Concatenate the outputs of all branches along the channel dimension
        return outputs