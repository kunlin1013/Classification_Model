from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Flatten, Dense, Activation
from tensorflow.keras.models import Model

def VGG16(input_shape=(224,224,3), nclass=1000):
    
    input_ = Input(shape = input_shape)                                         # output(None, 224, 224, 3)
    
    # block1
    x = Conv2D(64, kernel_size=3, padding='same', name='block1_conv1')(input_)  # output(None, 224, 224, 64)
    x = Activation('relu')(x)
    x = Conv2D(64, kernel_size=3, padding='same', name='block1_conv2')(x)       # output(None, 224, 224, 64)
    x = Activation('relu')(x)
    x = MaxPool2D(2, strides=2, name='block1_pool')(x)                          # output(None, 112, 112, 64)
    
    # block2
    x = Conv2D(128, kernel_size=3, padding='same', name='block2_conv1')(x)      # output(None, 112, 112, 128)
    x = Activation('relu')(x)
    x = Conv2D(128, kernel_size=3, padding='same', name='block2_conv2')(x)      # output(None, 112, 112, 128)
    x = Activation('relu')(x)
    x = MaxPool2D(2, strides=2, name='block2_pool')(x)                          # output(None, 56, 56, 128)
    
    # block3
    x = Conv2D(256, kernel_size=3, padding='same', name='block3_conv1')(x)      # output(None, 56, 56, 256)
    x = Activation('relu')(x)
    x = Conv2D(256, kernel_size=3, padding='same', name='block3_conv2')(x)      # output(None, 56, 56, 256)
    x = Activation('relu')(x)
    x = Conv2D(256, kernel_size=3, padding='same', name='block3_conv3')(x)      # output(None, 56, 56, 256)
    x = Activation('relu')(x)
    x = MaxPool2D(2, strides=2, name='block3_pool')(x)                          # output(None, 28, 28, 128)
    
    # block4
    x = Conv2D(512, kernel_size=3, padding='same', name='block4_conv1')(x)      # output(None, 28, 28, 512)
    x = Activation('relu')(x)
    x = Conv2D(512, kernel_size=3, padding='same', name='block4_conv2')(x)      # output(None, 28, 28, 512)
    x = Activation('relu')(x)
    x = Conv2D(512, kernel_size=3, padding='same', name='block4_conv3')(x)      # output(None, 28, 28, 512)
    x = Activation('relu')(x)
    x = MaxPool2D(2, strides=2, name='block4_pool')(x)                          # output(None, 14, 14, 512)
    
    # block5
    x = Conv2D(512, kernel_size=3, padding='same', name='block5_conv1')(x)      # output(None, 14, 14, 512)
    x = Activation('relu')(x)
    x = Conv2D(512, kernel_size=3, padding='same', name='block5_conv2')(x)      # output(None, 14, 14, 512)
    x = Activation('relu')(x)
    x = Conv2D(512, kernel_size=3, padding='same', name='block5_conv3')(x)      # output(None, 14, 14, 512)
    x = Activation('relu')(x)
    x = MaxPool2D(2, strides=2, name='block5_pool')(x)                          # output(None, 7, 7, 512)

    x = Flatten(name='flatten')(x)                                              # output(None, 7*7*512)
    x = Dense(4096, activation="relu", name='fc1')(x)                           # output(None, 4096)
    x = Dense(4096, activation="relu", name='fc2')(x)                           # output(None, 4096)
    output_ = Dense(nclass, activation='softmax', name='predictions')(x)        # output(None, nclass)

    model = Model(inputs=input_, outputs=output_)
    model.summary()
    return model

def VGG19(input_shape=(224,224,3), nclass=1000):
    
    input_ = Input(shape = input_shape)                                         # output(None, 224, 224, 3)
    
    # block1
    x = Conv2D(64, kernel_size=3, padding='same', name='block1_conv1')(input_)  # output(None, 224, 224, 64)
    x = Activation('relu')(x)
    x = Conv2D(64, kernel_size=3, padding='same', name='block1_conv2')(x)       # output(None, 224, 224, 64)
    x = Activation('relu')(x)
    x = MaxPool2D(2, strides=2, name='block1_pool')(x)                          # output(None, 112, 112, 64)
    
    # block2
    x = Conv2D(128, kernel_size=3, padding='same', name='block2_conv1')(x)      # output(None, 112, 112, 128)
    x = Activation('relu')(x)
    x = Conv2D(128, kernel_size=3, padding='same', name='block2_conv2')(x)      # output(None, 112, 112, 128)
    x = Activation('relu')(x)
    x = MaxPool2D(2, strides=2, name='block2_pool')(x)                          # output(None, 56, 56, 128)
    
    # block3
    x = Conv2D(256, kernel_size=3, padding='same', name='block3_conv1')(x)      # output(None, 56, 56, 256)
    x = Activation('relu')(x)
    x = Conv2D(256, kernel_size=3, padding='same', name='block3_conv2')(x)      # output(None, 56, 56, 256)
    x = Activation('relu')(x)
    x = Conv2D(256, kernel_size=3, padding='same', name='block3_conv3')(x)      # output(None, 56, 56, 256)
    x = Activation('relu')(x)
    x = Conv2D(256, kernel_size=3, padding='same', name='block3_conv4')(x)      # output(None, 56, 56, 256)
    x = Activation('relu')(x)
    x = MaxPool2D(2, strides=2, name='block3_pool')(x)                          # output(None, 28, 28, 128)
    
    # block4
    x = Conv2D(512, kernel_size=3, padding='same', name='block4_conv1')(x)      # output(None, 28, 28, 512)
    x = Activation('relu')(x)
    x = Conv2D(512, kernel_size=3, padding='same', name='block4_conv2')(x)      # output(None, 28, 28, 512)
    x = Activation('relu')(x)
    x = Conv2D(512, kernel_size=3, padding='same', name='block4_conv3')(x)      # output(None, 28, 28, 512)
    x = Activation('relu')(x)
    x = Conv2D(512, kernel_size=3, padding='same', name='block4_conv4')(x)      # output(None, 28, 28, 512)
    x = Activation('relu')(x)
    x = MaxPool2D(2, strides=2, name='block4_pool')(x)                          # output(None, 14, 14, 512)
    
    # block5
    x = Conv2D(512, kernel_size=3, padding='same', name='block5_conv1')(x)      # output(None, 14, 14, 512)
    x = Activation('relu')(x)
    x = Conv2D(512, kernel_size=3, padding='same', name='block5_conv2')(x)      # output(None, 14, 14, 512)
    x = Activation('relu')(x)
    x = Conv2D(512, kernel_size=3, padding='same', name='block5_conv3')(x)      # output(None, 14, 14, 512)
    x = Activation('relu')(x)
    x = Conv2D(512, kernel_size=3, padding='same', name='block5_conv4')(x)      # output(None, 14, 14, 512)
    x = Activation('relu')(x)
    x = MaxPool2D(2, strides=2, name='block5_pool')(x)                          # output(None, 7, 7, 512)

    x = Flatten(name='flatten')(x)                                              # output(None, 7*7*512)
    x = Dense(4096, activation="relu", name='fc1')(x)                           # output(None, 4096)
    x = Dense(4096, activation="relu", name='fc2')(x)                           # output(None, 4096)
    output_ = Dense(nclass, activation='softmax', name='predictions')(x)        # output(None, nclass)

    model = Model(inputs=input_, outputs=output_)
    model.summary()
    return model

if __name__ == '__main__':
    net_final = VGG19()