import imgaug.augmenters as iaa
import tensorflow as tf
import numpy as np

def DataGenerator_train(dir: str, data_dict: dict, IsAugmentation: bool = True, batch_size: int = 32):
    
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    IMG_SIZE = [224, 224]
    
    def load_and_preprocess_img(path, label):
        def load_numpy_file(file_path):
            file_path = file_path.numpy()
            file_path_str = file_path.decode()
            numpy_array = np.load(file_path_str)
            return numpy_array.astype("float32")

        image_string = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image_string, channels=3)
        image = tf.image.resize(image, IMG_SIZE)
        image = image / 255.0
        return image, label
    
    def image_augmentation(img, label):
        def sequential_aug(img):
            img = tf.cast(img * 255, tf.uint8)
            
            sometimes = lambda aug: iaa.Sometimes(0.3, aug)  # apply operations on 50% of input data
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
    img_path_list = []
    for path in data_dict['filepaths']:
        img_path_list.append(r"..\..\Dataset" + "\\" + path.replace('/', '\\'))
    label_list = data_dict['class id']
    
    print('Total samples:', len(label_list))
    
    X = img_path_list
    Y = tf.one_hot(tf.constant(label_list), depth=len(set(label_list)))
    
    # Construct tf.data.Dataset
    data = tf.data.Dataset.from_tensor_slices((X, Y))
    data = data.shuffle(len(label_list))
    
    def load_dataset(x, y):
        return tf.data.Dataset.from_tensors(load_and_preprocess_img(x, y))
    data = data.map(lambda x, y: load_and_preprocess_img(x, y), AUTOTUNE)
    if dir == "train" and IsAugmentation:
        data = data.map(lambda x, y: image_augmentation(x, y), AUTOTUNE) # augment only the training dataset
    
    # Add all the settings
    data = data.cache()
    data = data.batch(batch_size)
    data = data.prefetch(AUTOTUNE)
    
    total_data = len(img_path_list)
    
    return data, total_data


# # ========================== to test if the program works ========================== 
# import sys
# import matplotlib.pyplot as plt
# sys.path.append("..")
# from lib.split_data import split_data

# CSVPATH = r"..\..\Dataset\sports.csv"
# train_data, val_data = split_data(CSVPATH)
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