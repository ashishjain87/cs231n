# Moves images into train, val, test
import os
import sys
# import shutil
import numpy as np

DATA_DIR_ORIG = './data/images_original/'
DATA_DIR_NEW = './data/images/'
LABELS_DIR_ORIG = './data/labels_original/'
LABELS_DIR_NEW = './data/labels/'
TRAIN_PATH = './train_split.csv'
VAL_PATH = './val_split.csv'
TEST_PATH = './test_split.csv'

def get_filename_from_id(image_id):
    str_id = str(image_id)
    return "0"*(6-len(str_id)) + str_id

def move_data(ids, orig_dir, new_dir, split_name="train", extension=".png"):
    for image_id in ids:
        image_name = get_filename_from_id(image_id)
        old_image_filename = os.path.join(os.path.dirname(__file__), orig_dir + image_name + extension)
        new_image_filename = os.path.join(os.path.dirname(__file__), new_dir + split_name + "/" + image_name + extension)
        os.rename(old_image_filename, new_image_filename)

if __name__ == "__main__":
    if not os.path.exists(DATA_DIR_NEW + 'train'):
        os.mkdir(DATA_DIR_NEW + 'train')
    if not os.path.exists(DATA_DIR_NEW + 'val'):
        os.mkdir(DATA_DIR_NEW + 'val')
    if not os.path.exists(DATA_DIR_NEW + 'test'):
        os.mkdir(DATA_DIR_NEW + 'test')
    if not os.path.exists(LABELS_DIR_NEW + 'train'):
        os.mkdir(LABELS_DIR_NEW + 'train')
    if not os.path.exists(LABELS_DIR_NEW + 'val'):
        os.mkdir(LABELS_DIR_NEW + 'val')
    if not os.path.exists(LABELS_DIR_NEW + 'test'):
        os.mkdir(LABELS_DIR_NEW + 'test')

    train_ids = np.genfromtxt(TRAIN_PATH, delimiter=",", dtype=None, encoding=None)
    val_ids = np.genfromtxt(VAL_PATH, delimiter=",", dtype=None, encoding=None)
    test_ids = np.genfromtxt(TEST_PATH, delimiter=",", dtype=None, encoding=None)
    
    move_data(train_ids, DATA_DIR_ORIG, DATA_DIR_NEW, split_name="train")
    move_data(val_ids, DATA_DIR_ORIG, DATA_DIR_NEW, split_name="val")
    move_data(test_ids, DATA_DIR_ORIG, DATA_DIR_NEW, split_name="test")
    move_data(train_ids, LABELS_DIR_ORIG, LABELS_DIR_NEW, split_name="train", extension=".txt")
    move_data(val_ids, LABELS_DIR_ORIG, LABELS_DIR_NEW, split_name="val", extension=".txt")
    move_data(test_ids, LABELS_DIR_ORIG, LABELS_DIR_NEW, split_name="test", extension=".txt")
