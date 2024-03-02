import tensorflow as tf
from tensorflow.keras.layers import Input, ZeroPadding2D, Conv2D, MaxPool2D, Flatten, Dropout, Dense, Lambda, Activation
from tensorflow.keras.models import Model

def AlexNet(input_shape=(224,224,3), nclass=1000):
    '''
    LRN (Local Response Normalization) is applied only in the first and second convolutional layers.
    Dropout = 0.5 (default)
    '''
    
    input_ = Input(shape = input_shape)                                     # output(None, 224, 224, 3)
    x = ZeroPadding2D(((1, 2), (1, 2)))(input_)                             # output(None, 227, 227, 3)
    
    # conv 1
    x = Conv2D(96, kernel_size=11, strides=4)(x)                            # output(None, 55, 55, 96)
    x = Lambda(tf.nn.local_response_normalization)(x)
    x = Activation('relu')(x)
    x = MaxPool2D(3, strides=2)(x)                                          # output(None, 27, 27, 96)
    
    # conv 2
    x = Conv2D(256, kernel_size=5, padding="same")(x)                       # output(None, 27, 27, 256)
    x = Lambda(tf.nn.local_response_normalization)(x)
    x = Activation('relu')(x)
    x = MaxPool2D(3, strides=2)(x)                                          # output(None, 13, 13, 256)
    
    # conv 3
    x = Conv2D(384, kernel_size=3, padding="same")(x)                       # output(None, 13, 13, 384)
    x = Activation('relu')(x)
    
    # conv 4
    x = Conv2D(384, kernel_size=3, padding="same")(x)                       # output(None, 13, 13, 384)
    x = Activation('relu')(x)
    
    # conv 5
    x = Conv2D(256, kernel_size=3, padding="same")(x)                       # output(None, 13, 13, 256)
    x = Activation('relu')(x)
    x = MaxPool2D(3, strides=2)(x)                                          # output(None, 6, 6, 256)

    x = Flatten()(x)                                                        # output(None, 6*6*256)
    x = Dense(4096, activation="relu")(x)                                   # output(None, 4096)
    x = Dropout(0.5)(x)     
    x = Dense(4096, activation="relu")(x)                                   # output(None, 4096)
    x = Dropout(0.5)(x)
    output_ = Dense(nclass, activation='softmax')(x)                        # output(None, nclass)

    model = Model(inputs=input_, outputs=output_)
    model.summary()
    return model

if __name__ == '__main__':
    net_final = AlexNet()