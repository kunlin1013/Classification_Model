import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from model_v2 import MobileNetV2
import numpy as np
import sys
sys.path.append("..")
from lib.load_data import get_data_dict, DataGenerator_test  

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
DATASETPATH = r"..\..\Dataset\flower_photos"
CLASSINDEX = r"..\..\Dataset\flower_photos/class_index.json"
WEIGHTS_PATH = r"./save_weights_v2/model_weights_54_0.788.h5"

if __name__ == '__main__':
    # Load training data and divide into two parts: training set and testing set 
    test_data =  get_data_dict(DATASETPATH, "test", CLASSINDEX)
    groundtruth = test_data['class id']
 
    # Use DataGenerator to generate train batch and val batch
    test_ds, test_count = DataGenerator_test(dir='test', data_dict=test_data, batch_size=BATCH_SIZE)
    
    model = MobileNetV2(input_shape=(224,224,3), nclass=5)
    model.summary()
    model.load_weights(WEIGHTS_PATH, by_name=True)
    
    result = np.squeeze(model.predict(test_ds))
    predict_class = np.argmax(result, axis=1)
    
    accuracy = np.mean(predict_class == groundtruth)
    print(f"Accuracy: {accuracy*100:.2f} %")
    
tf.config.experimental.set_memory_growth(gpus[0],True)