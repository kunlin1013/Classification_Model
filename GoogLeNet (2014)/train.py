import matplotlib.pyplot as plt 
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from model import GoogLeNet

import sys
sys.path.append("..")
from lib.split_data import split_data
from lib.load_data import DataGenerator_train
from collections import defaultdict

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

@tf.function
def train_step(model, batch, loss_object, optimizer, train_accuracy , train_loss):
    image, label_true = batch
    with tf.GradientTape() as tape:
        aux1, aux2, output = model(image, training=True)
        loss1 = loss_object(label_true, aux1)
        loss2 = loss_object(label_true, aux2)
        loss3 = loss_object(label_true, output)
        loss = loss1*0.3 + loss2*0.3 + loss3
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_accuracy(label_true, output)
    train_loss(loss)
    
    return loss
    
@tf.function
def val_step(model, batch, loss_object, val_accuracy, val_loss):
    image, label_true = batch
    _, _, output = model(image, training=False)
    v_loss = loss_object(label_true, output)

    val_loss(v_loss)
    val_accuracy(label_true, output)
    return v_loss

# Set training hyperparameters 
BATCH_SIZE = 32
LEARNING_RATE = 0.01
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
    train_ds, train_count = DataGenerator_train(dir='train', data_dict=train_data, IsAugmentation=False, batch_size=BATCH_SIZE)
    val_ds, val_count = DataGenerator_train(dir='val', data_dict=val_data, IsAugmentation=False, batch_size=BATCH_SIZE)
    
    model = GoogLeNet(input_shape=(224,224,3), nclass=100, aux_logits=True)
    model.summary()
    
    # using keras low level api for training
    loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
    
    val_loss = tf.keras.metrics.Mean(name='val_loss')
    val_accuracy = tf.keras.metrics.CategoricalAccuracy(name='val_accuracy')

    best_test_loss = float('inf')
    history = defaultdict(list)
    patience_counter_learningrate = 0
    patience_counter_earlystopping = 0
    for epoch in range(EPOCHS):
        start_time = time.time()
        print("Epoch {}/{}".format(epoch+1, EPOCHS))
        
        # Optimize the model using the training data
        train_loss.reset_states()        # clear history info
        train_accuracy.reset_states()    # clear history info
        i_step = 0
        for batch in train_ds:
            loss = train_step(model, batch, loss_object, optimizer, train_accuracy, train_loss)
            print("\rStep {}, loss: {:.6f} ".format(i_step, tf.reduce_mean(loss)), end='')
            i_step += 1
        
        print(', loss (epoch): {:.6f}, acc (epoch): {:.2f}% '.format(train_loss.result(), train_accuracy.result()*100))
        history['loss'].append(train_loss.result())
        history['accuracy'].append(train_accuracy.result())
            
        # Evaluate on the validation data
        val_loss.reset_states()         # clear history info
        val_accuracy.reset_states()     # clear history info
        i_step = 0
        for batch in val_ds:
            loss = val_step(model, batch, loss_object, val_accuracy, val_loss)
            print("\rStep {}, loss: {:.6f} ".format(i_step, tf.reduce_mean(loss)), end='')
            i_step += 1
        
        print(', loss (epoch): {:.6f}, acc (epoch): {:.2f}% '.format(val_loss.result(), val_accuracy.result()*100))
        history['val_loss'].append(val_loss.result())
        history['val_accuracy'].append(val_accuracy.result())
        
        end_time = time.time()
        print("Time taken: {:.2f} s".format(end_time - start_time))
        
        if val_loss.result() < best_test_loss:
            model_savepath = f'model_weights_{epoch+1}_{val_loss.result():.3f}.h5'
            model.save_weights(model_savepath)
            print(f'Epoch {epoch+1:05d}: val_acc improved from {best_test_loss:.5f} to {val_loss.result():.5f}, saving model {model_savepath}')
            best_test_loss = val_loss.result()
            patience_counter_learningrate = 0
            patience_counter_earlystopping = 0
        else:
            patience_counter_learningrate += 1
            patience_counter_earlystopping += 1
        
        if patience_counter_learningrate >= LEARNING_RATE_DECAY_PATIENCE:
            new_lr = optimizer.learning_rate * LEARNING_RATE_DECAY_FACTOR
            print(f'Reducing learning rate to {new_lr:.6f}.')
            optimizer.learning_rate = new_lr
            patience_counter_learningrate = 0
        
        if patience_counter_earlystopping >= EARLY_STOPPING_PATIENCE:
            print("Early stopping...")
            break
        
    # plot training history
    plt.figure(dpi=300)
    plt.plot(history['accuracy'], label='accuracy')
    plt.plot(history['val_accuracy'], label = 'val_accuracy')  
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('accuracy_history.png')

    plt.figure(dpi=300)
    plt.plot(history['loss'], label='loss')
    plt.plot(history['val_loss'], label = 'val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss_history.png')

tf.config.experimental.set_memory_growth(gpus[0],True)