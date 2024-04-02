from tensorflow.keras import layers, Model, Sequential
from tensorflow.keras import regularizers

class BasicBlock(layers.Layer):
    expansion = 1

    def __init__(self, out_channel, strides=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__(**kwargs)
        self.conv1 = layers.Conv2D(out_channel, kernel_size=3, strides=strides,
                                   padding="SAME", use_bias=False, kernel_regularizer=regularizers.l2(0.01))
        self.bn1 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        # -----------------------------------------
        self.conv2 = layers.Conv2D(out_channel, kernel_size=3, strides=1,
                                   padding="SAME", use_bias=False, kernel_regularizer=regularizers.l2(0.01))
        self.bn2 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        # -----------------------------------------
        self.downsample = downsample
        self.relu = layers.ReLU()
        self.add = layers.Add()

    def call(self, inputs, training=False):
        identity = inputs
        if self.downsample is not None:
            identity = self.downsample(inputs)

        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)

        x = self.add([identity, x])
        x = self.relu(x)

        return x

class Bottleneck(layers.Layer):
    """
    Note: In the original paper, on the main branch of the dashed residual structure, the stride of the first 1x1 convolution layer is 2, and the stride of the second 3x3 convolution layer is 1.
    However, in the official PyTorch implementation, the stride of the first 1x1 convolution layer is 1, and the stride of the second 3x3 convolution layer is 2.
    The advantage of doing this is that it can improve the top-1 accuracy by about 0.5%.
    For reference, see ResNet v1.5 at https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch
    """
    expansion = 4   # The number of channels in the last layer is four times that of the previous layers.

    def __init__(self, out_channel, strides=1, downsample=None, **kwargs):
        super(Bottleneck, self).__init__(**kwargs)
        self.conv1 = layers.Conv2D(out_channel, kernel_size=1, use_bias=False, kernel_regularizer=regularizers.l2(0.01), name="conv1")
        self.bn1 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name="conv1/BatchNorm")
        # -----------------------------------------
        self.conv2 = layers.Conv2D(out_channel, kernel_size=3, use_bias=False,
                                   strides=strides, padding="SAME", kernel_regularizer=regularizers.l2(0.01), name="conv2")
        self.bn2 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name="conv2/BatchNorm")
        # -----------------------------------------
        self.conv3 = layers.Conv2D(out_channel * self.expansion, kernel_size=1, use_bias=False, kernel_regularizer=regularizers.l2(0.01), name="conv3")
        self.bn3 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name="conv3/BatchNorm")
        # -----------------------------------------
        self.relu = layers.ReLU()
        self.downsample = downsample
        self.add = layers.Add()

    def call(self, inputs, training=False):
        identity = inputs
        if self.downsample is not None:
            identity = self.downsample(inputs)

        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x, training=training)

        x = self.add([x, identity])
        x = self.relu(x)

        return x

def _make_layer(block, in_channel, channel, block_num, name, strides=1):
    downsample = None
    # In models after ResNet50, the shortcut connection needs to be dimensionally increased, 
    # changing the number of channels to match the output channels.
    if strides != 1 or in_channel != channel * block.expansion:
        downsample = Sequential([
            layers.Conv2D(channel * block.expansion, kernel_size=1, strides=strides,
                          use_bias=False, kernel_regularizer=regularizers.l2(0.01), name="conv1"),
            layers.BatchNormalization(momentum=0.9, epsilon=1.001e-5, name="BatchNorm")
        ], name="shortcut")

    layers_list = []
    layers_list.append(block(channel, downsample=downsample, strides=strides, name="unit_1"))

    for index in range(1, block_num):
        layers_list.append(block(channel, name="unit_" + str(index + 1)))

    return Sequential(layers_list, name=name)

def _resnet(block, blocks_num, input_shape=(224,224,3), nclass=1000, include_top=True):
    input_ = layers.Input(shape=input_shape, dtype="float32")
    x = layers.Conv2D(filters=64, kernel_size=7, strides=2,
                      padding="SAME", use_bias=False, kernel_regularizer=regularizers.l2(0.01), name="conv1")(input_)
    x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name="conv1/BatchNorm")(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D(pool_size=3, strides=2, padding="SAME")(x)

    # In block1, the height and width of the features do not change, so strides=1.
    x = _make_layer(block, x.shape[-1], 64, blocks_num[0], name="block1")(x)
    x = _make_layer(block, x.shape[-1], 128, blocks_num[1], strides=2, name="block2")(x)
    x = _make_layer(block, x.shape[-1], 256, blocks_num[2], strides=2, name="block3")(x)
    x = _make_layer(block, x.shape[-1], 512, blocks_num[3], strides=2, name="block4")(x)

    if include_top:
        x = layers.GlobalAvgPool2D()(x)  # pool + flatten
        x = layers.Dense(nclass, name="logits")(x)
        predict = layers.Softmax()(x)
    else:
        predict = x

    model = Model(inputs=input_, outputs=predict)

    return model

def resnet18(input_shape=(224,224,3), nclass=1000, include_top=True):
    return _resnet(BasicBlock, [2, 2, 2, 2], input_shape, nclass, include_top)

def resnet34(input_shape=(224,224,3), nclass=1000, include_top=True):
    return _resnet(BasicBlock, [3, 4, 6, 3], input_shape, nclass, include_top)

def resnet50(input_shape=(224,224,3), nclass=1000, include_top=True):
    return _resnet(Bottleneck, [3, 4, 6, 3], input_shape, nclass, include_top)

def resnet101(input_shape=(224,224,3), nclass=1000, include_top=True):
    return _resnet(Bottleneck, [3, 4, 23, 3], input_shape, nclass, include_top)

def resnet152(input_shape=(224,224,3), nclass=1000, include_top=True):
    return _resnet(Bottleneck, [3, 8, 36, 3], input_shape, nclass, include_top)

if __name__ == '__main__':
    model = resnet152()
    model.summary()