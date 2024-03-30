import pandas as pd
from sklearn.model_selection import train_test_split
from shutil import copy, rmtree
import os
import random

def mk_file(file_path: str):
    if os.path.exists(file_path):
        # If the folder exists, delete the original folder and then recreate it
        rmtree(file_path)
    os.makedirs(file_path)

FULLDATAPATH = r"..\..\Dataset\flower_photos\FullDataSet"
SAVEPATH = r"..\..\Dataset\flower_photos"

def main():
    random.seed(0)
    
    # Define the split ratio.
    train_rate = 0.7
    val_rate = 0.2
    
    assert os.path.exists(FULLDATAPATH), "path '{}' does not exist.".format(FULLDATAPATH)
    
    flower_class = [cla for cla in os.listdir(FULLDATAPATH)
                    if os.path.isdir(os.path.join(FULLDATAPATH, cla))]
    
    # Create folders to save the training set, validation set, and test set
    train_root = os.path.join(SAVEPATH, "train")
    val_root = os.path.join(SAVEPATH, "val")
    test_root = os.path.join(SAVEPATH, "test")
    mk_file(train_root)
    mk_file(val_root)
    mk_file(test_root)
    for cla in flower_class:
        mk_file(os.path.join(train_root, cla))
        mk_file(os.path.join(val_root, cla))
        mk_file(os.path.join(test_root, cla))
    
    for cla in flower_class:
        cla_path = os.path.join(FULLDATAPATH, cla)
        images = os.listdir(cla_path)
        num = len(images)
        random.shuffle(images)
        train_num = int(num * train_rate)
        val_num = int(num * val_rate)
        
        for i, image in enumerate(images):
            image_path = os.path.join(cla_path, image)
            if i < train_num:
                # 分配至训练集
                new_path = os.path.join(train_root, cla)
            elif i < train_num + val_num:
                # 分配至验证集
                new_path = os.path.join(val_root, cla)
            else:
                # 分配至测试集
                new_path = os.path.join(test_root, cla)
            copy(image_path, new_path)
            print("\r[{}] processing [{}/{}]".format(cla, i+1, num), end="")  # processing bar
        print()
    
    print("Processing done!")

if __name__ == '__main__':
    main()