from Inception import Inception, InceptionAux
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Flatten, Dense, Lambda, AvgPool2D, Dropout
from tensorflow.keras.models import Model


def GoogLeNet(input_shape=(224,224,3), nclass=1000):
    input_ = Input(shape = input_shape)                                                                     # output(None, 224, 224, 3)
    x = Conv2D(64, kernel_size=7, strides=2, padding="SAME", activation="relu", name="conv2d_1")(input_)    # output(None, 112, 112, 64)
    x = MaxPool2D(pool_size=3, strides=2, padding="SAME", name="maxpool_1")(x)                              # output(None, 56, 56, 64)
    x = Lambda(tf.nn.local_response_normalization)(x)                                                       # Local Response Normalization
    x = Conv2D(64, kernel_size=1, activation="relu", name="conv2d_2")(x)                                    # output(None, 56, 56, 64)
    x = Conv2D(192, kernel_size=3, padding="SAME", activation="relu", name="conv2d_3")(x)                   # output(None, 56, 56, 192)
    x = Lambda(tf.nn.local_response_normalization)(x)                                                       # Local Response Normalization
    x = MaxPool2D(pool_size=3, strides=2, padding="SAME", name="maxpool_2")(x)                              # output(None, 28, 28, 192)

    # Inception part
    x = Inception(64, 96, 128, 16, 32, 32, name="inception_3a")(x)                                          # output(None, 28, 28, 256)
    x = Inception(128, 128, 192, 32, 96, 64, name="inception_3b")(x)                                        # output(None, 28, 28, 480)
    x = MaxPool2D(pool_size=3, strides=2, padding="SAME", name="maxpool_3")(x)                              # output(None, 14, 14, 480)
    x = Inception(192, 96, 208, 16, 48, 64, name="inception_4a")(x)                                         # output(None, 14, 14, 512)

    # first auxiliary classifier
    aux1 = InceptionAux(nclass, name="aux_1")(x)

    x = Inception(160, 112, 224, 24, 64, 64, name="inception_4b")(x)                                        # output(None, 14, 14, 512)
    x = Inception(128, 128, 256, 24, 64, 64, name="inception_4c")(x)                                        # output(None, 14, 14, 512)
    x = Inception(112, 144, 288, 32, 64, 64, name="inception_4d")(x)                                        # output(None, 14, 14, 528)

    # second auxiliary classifier
    aux2 = InceptionAux(nclass, name="aux_2")(x)

    x = Inception(256, 160, 320, 32, 128, 128, name="inception_4e")(x)                                      # output(None, 14, 14, 832)
    x = MaxPool2D(pool_size=3, strides=2, padding="SAME", name="maxpool_4")(x)                              # output(None, 7, 7, 832)
    x = Inception(256, 160, 320, 32, 128, 128, name="inception_5a")(x)                                      # output(None, 7, 7, 832)
    x = Inception(384, 192, 384, 48, 128, 128, name="inception_5b")(x)                                      # output(None, 7, 7, 1024)
    x = AvgPool2D(pool_size=7, strides=1, name="avgpool_1")(x)                                              # output(None, 1, 1, 1024)

    x = Flatten(name="output_flatten")(x)                                                                   # output(None, 1*1*1024)
    x = Dropout(rate=0.4, name="output_dropout")(x)
    aux3 = Dense(nclass, activation='softmax', name='aux_3')(x)                                             # output(None, nclass)


    model = Model(inputs=input_, outputs=[aux1, aux2, aux3])
    model.summary()
    return model

if __name__ == '__main__':
    net_final = GoogLeNet()