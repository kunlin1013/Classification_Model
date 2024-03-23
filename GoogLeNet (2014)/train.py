import matplotlib.pyplot as plt 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras import optimizers, losses
from model import GoogLeNet

import sys
sys.path.append("..")
from lib.split_data import split_data
from lib.load_data import DataGenerator_train

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

def replicate_labels(image, label):
    return image, (label, label, label)

# Set training hyperparameters 
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
LEARNING_RATE_DECAY_FACTOR = 0.08
LEARNING_RATE_DECAY_PATIENCE = 3
EARLY_STOPPING_PATIENCE = 8
EPOCHS = 100
CSVPATH = r"..\..\Dataset\sports.csv"

if __name__ == '__main__':
    # Load training data and divide into two parts: training set and testing set 
    split_ratio = 0.3
    train_data, val_data = split_data(CSVPATH, split_ratio=split_ratio)

    # Use DataGenerator to generate train batch and val batch
    train, train_count = DataGenerator_train(dir='train', data_dict=train_data, IsAugmentation=True, batch_size=BATCH_SIZE)
    val, val_count = DataGenerator_train(dir='val', data_dict=val_data, IsAugmentation=True, batch_size=BATCH_SIZE)
    # use the map method to apply label replication
    train = train.map(replicate_labels)
    val = val.map(replicate_labels)
    
    # Set callback function
    Reduce = ReduceLROnPlateau(monitor='val_loss',
                               factor=LEARNING_RATE_DECAY_FACTOR,
                               patience=LEARNING_RATE_DECAY_PATIENCE,
                               verbose=EARLY_STOPPING_PATIENCE,
                               mode='min')

    early_stopping = EarlyStopping(monitor='val_loss', 
                                   patience=EARLY_STOPPING_PATIENCE, 
                                   verbose=1, 
                                   mode='min') 

    # filepath = "GoogLeNet-{epoch:02d}-{val_loss:.3f}.h5"
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filepath = "GoogLeNet-" + timestamp + "-{epoch:02d}-{val_loss:.3f}.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    callbacks_list = [checkpoint, Reduce, early_stopping] 

    # Load model and set complie
    net_final = GoogLeNet(input_shape=(224,224,3), nclass=100)
    net_final.compile(optimizer=optimizers.Adam(lr=LEARNING_RATE),
                      loss=[losses.categorical_crossentropy,
                            losses.categorical_crossentropy,
                            losses.categorical_crossentropy],
                      loss_weights=[0.4, 0.3, 0.3],
                      metrics=['accuracy'])

    # Prepare dataset from train and val and calculate train/val step
    train_dataset = train.repeat()
    val_dataset = val.repeat()
    train_steps = train_count // BATCH_SIZE
    val_steps = val_count // BATCH_SIZE

    # Train model
    history = net_final.fit(train_dataset,
                            steps_per_epoch=train_steps,
                            validation_data=val_dataset,
                            validation_steps=val_steps,
                            epochs=EPOCHS,
                            callbacks=callbacks_list
                            )

    # Save acc/loss figure
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['aux_3_accuracy'], label='accuracy')
    plt.plot(history.history['val_aux_3_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('accuracy')
    # plt.ylim([0, 1])
    plt.legend(loc='lower right')
    # test_loss, test_acc = net_final.evaluate([X_test],Y_test, verbose=2)
    print('accuracy=',history.history['aux_3_accuracy'][-1],"   ","val_accuracy=",history.history['val_aux_3_accuracy'][-1])
    plt.savefig(r'.\GoogLeNet_acc.png')
    
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    #plt.ylim([0.3, 1])
    plt.legend(loc='upper right')
    # test_loss, test_acc = net_final.evaluate([X_test],Y_test, verbose=2)
    print('loss=',history.history['loss'][-1],"   ","val_loss=",history.history['val_loss'][-1])
    plt.savefig(r'.\GoogLeNet_loss.png')

tf.config.experimental.set_memory_growth(gpus[0],True)