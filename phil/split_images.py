# Moves images into train, val, test
import os
import sys
# import shutil
import numpy as np

DATA_DIR_ORIG = '../image_2/'
DATA_DIR_NEW = '../raw_kitti/images/'
LABELS_DIR_ORIG = '../yolo_labels/'
LABELS_DIR_NEW = '../raw_kitti/labels/'
TRAIN_PATH = '../train_split.csv'
VAL_PATH = '../val_split.csv'
TEST_PATH = '../test_split.csv'

def get_filename_from_id(image_id):
    str_id = str(image_id)
    return "0"*(6-len(str_id)) + str_id

def move_data(ids, orig_dir, new_dir, split_name="train"):
    for image_id in ids:
        image_name = get_filename_from_id(image_id)
        old_image_filename = os.path.join(os.path.dirname(__file__), orig_dir + image_name + ".png")
        new_image_filename = os.path.join(os.path.dirname(__file__), new_dir + split_name + "/" + image_name + ".png")
        os.rename(old_image_filename, new_image_filename)

if __name__ == "__main__":
    if not os.path.exists('../raw_kitti/images/train'):
        os.mkdir('../raw_kitti/images/train')
    if not os.path.exists('../raw_kitti/images/val'):
        os.mkdir('../raw_kitti/images/val')
    if not os.path.exists('../raw_kitti/images/test'):
        os.mkdir('../raw_kitti/images/test')
    if not os.path.exists('../raw_kitti/labels/train'):
        os.mkdir('../raw_kitti/labels/train')
    if not os.path.exists('../raw_kitti/labels/val'):
        os.mkdir('../raw_kitti/labels/val')
    if not os.path.exists('../raw_kitti/labels/test'):
        os.mkdir('../raw_kitti/labels/test')

    train_ids = np.genfromtxt(TRAIN_PATH, delimiter=",", dtype=None, encoding=None)
    val_ids = np.genfromtxt(VAL_PATH, delimiter=",", dtype=None, encoding=None)
    test_ids = np.genfromtxt(TEST_PATH, delimiter=",", dtype=None, encoding=None)
    
    move_data(train_ids, DATA_DIR_ORIG, DATA_DIR_NEW, split_name="train")
    move_data(val_ids, DATA_DIR_ORIG, DATA_DIR_NEW, split_name="val")
    move_data(test_ids, DATA_DIR_ORIG, DATA_DIR_NEW, split_name="test")
    move_data(train_ids, LABELS_DIR_ORIG, LABELS_DIR_NEW, split_name="train")
    move_data(val_ids, LABELS_DIR_ORIG, LABELS_DIR_NEW, split_name="val")
    move_data(test_ids, LABELS_DIR_ORIG, LABELS_DIR_NEW, split_name="test")
