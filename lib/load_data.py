import imgaug.augmenters as iaa
import tensorflow as tf
import os
import json

def get_data_dict(datasetpath: str, dir: str, classindex: str):
    fulldatapath = os.path.join(datasetpath, dir)
    assert os.path.exists(fulldatapath), "path '{}' does not exist.".format(fulldatapath)
    
    # Read the class index file
    with open(classindex, 'r') as f:
        class_index = json.load(f)
    
    filepaths = []
    class_ids = []

    for class_name in os.listdir(fulldatapath):
        class_folder = os.path.join(fulldatapath, class_name)
        if os.path.isdir(class_folder):
            for img_filename in os.listdir(class_folder):
                img_path = os.path.join(class_folder, img_filename)
                filepaths.append(img_path)
                class_ids.append(class_index[class_name])
    
    # output for dict
    result_dict = {
        'filepaths': filepaths,
        'class id': class_ids
    }
    
    return result_dict
                
def DataGenerator_train(dir: str, data_dict: dict, IsAugmentation: bool = True, batch_size: int = 32):
    
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    IMG_SIZE = [224, 224]
    
    def load_and_preprocess_img(path, label):
        image_string = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image_string, channels=3)
        image = tf.image.resize(image, IMG_SIZE)
        image = image / 255.0
        return image, label
    
    def image_augmentation(img, label):
        def sequential_aug(img):
            img = tf.cast(img * 255, tf.uint8)
            
            sometimes = lambda aug: iaa.Sometimes(0.8, aug)  # apply operations on 50% of input data
            seq = iaa.Sequential([sometimes(iaa.SomeOf(1, 
                                                        [iaa.Affine(scale={'x': (0.9, 1.1), 'y': (0.9, 1.1)},
                                                                    translate_percent={'x': (-0.1, 0.1), 'y': (-0.1, 0.1)},
                                                                    rotate=(-30, 30), order=3, cval=0),
                                                        iaa.PerspectiveTransform(scale=(0.01, 0.15))]
                                                        )
                                            ),
                                    sometimes(iaa.AddToHueAndSaturation((-50, 50), per_channel=True)),
                                    iaa.Flipud(0.3), # horizontally flip 30% of the images
                                    sometimes(iaa.SomeOf(1,
                                                        [iaa.GaussianBlur(sigma=(0.5, 3.0)),
                                                          iaa.imgcorruptlike.GaussianNoise(severity=2),
                                                          iaa.imgcorruptlike.SpeckleNoise(severity=2)]
                                                        )
                                              )
                                ])
            
            seq = seq.to_deterministic()
            img = seq.augment_image(img.numpy())

            return img
        
        img = tf.py_function(sequential_aug, [img], (tf.float32))
        img.set_shape([IMG_SIZE[0], IMG_SIZE[1], 3])
        img = tf.cast(img, tf.float32) / 255.0
        
        return img, label
    
    # Get path to all files
    img_path_list = data_dict['filepaths']
    label_list = data_dict['class id']
    
    print('Total samples:', len(label_list))
    
    X = img_path_list
    Y = tf.one_hot(tf.constant(label_list), depth=len(set(label_list)))
    
    # Construct tf.data.Dataset
    data = tf.data.Dataset.from_tensor_slices((X, Y))
    data = data.map(lambda x, y: load_and_preprocess_img(x, y), AUTOTUNE)
    # cache(): Allows the read data to be stored in cache memory for repeated use thereafter.
    data = data.cache()
    if dir == "train" and IsAugmentation:
        data = data.map(lambda x, y: image_augmentation(x, y), AUTOTUNE) # augment only the training dataset
    
    # Add all the settings
    # prefetch(): During training, simultaneously read the next batch of data and perform transformations.
    data = data.shuffle(len(label_list))
    data = data.batch(batch_size)
    data = data.prefetch(AUTOTUNE)
    
    total_data = len(img_path_list)
    
    return data, total_data

def DataGenerator_test(dir: str, data_dict: dict, batch_size: int = 32):
    
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    IMG_SIZE = [224, 224]
    
    def load_and_preprocess_img(path, label):
        image_string = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image_string, channels=3)
        image = tf.image.resize(image, IMG_SIZE)
        image = image / 255.0
        return image, label
    
    # Get path to all files
    img_path_list = data_dict['filepaths']
    label_list = data_dict['class id']
    
    print('Total samples:', len(label_list))
    
    X = img_path_list
    Y = tf.one_hot(tf.constant(label_list), depth=len(set(label_list)))
    
    # Construct tf.data.Dataset
    data = tf.data.Dataset.from_tensor_slices((X, Y))
    data = data.map(lambda x, y: load_and_preprocess_img(x, y), AUTOTUNE)
    
    # Add all the settings
    # cache(): Allows the read data to be stored in cache memory for repeated use thereafter.
    # prefetch(): During training, simultaneously read the next batch of data and perform transformations.
    data = data.cache()
    data = data.batch(batch_size)
    data = data.prefetch(AUTOTUNE)
    
    total_data = len(img_path_list)
    
    return data, total_data

# # ========================== to test if the program works ========================== 
# import matplotlib.pyplot as plt

# DATASETPATH = r"..\..\Dataset\flower_photos"
# CLASSINDEX = r"..\..\Dataset\flower_photos/class_index.json"
# train_data =  get_data_dict(DATASETPATH, "train", CLASSINDEX)
# BATCH_SIZE = 32

# train, train_count = DataGenerator_train(dir='train', data_dict=train_data, IsAugmentation=True, batch_size=BATCH_SIZE)

# # get the first batch from a data generator
# for img_batch, label_batch in train.take(1):
#     first_img = img_batch[0].numpy()
#     first_label_one_hot = label_batch[0]
    
#     # convert One-Hot encoded labels back to numerical labels
#     first_label = tf.argmax(first_label_one_hot).numpy()

#     # show image
#     plt.figure(figsize=(6, 6))
#     plt.imshow(first_img)
#     plt.title(f"Label: {first_label}")
#     plt.show()
#     break 