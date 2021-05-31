# Moves images into train, val, test
import os
from phil.constants import NUM_PER_SPLIT
import numpy as np
from constants import SPLIT_NAMES

DATA_DIR_ORIG = './data/images_original/'
DATA_DIR_NEW = './data/images/'
YOLO_LABELS_DIR_ORIG = './data/labels_original/'
YOLO_LABELS_DIR_NEW = './data/labels/'
KITTI_LABELS_DIR_ORIG = './data/kitti_labels_original/'
KITTI_LABELS_DIR_NEW = './data/kitti_labels/'


def get_filename_from_id(image_id):
    str_id = str(image_id)
    return "0"*(6-len(str_id)) + str_id

def move_data(ids, orig_dir, new_dir, split_name, extension):
    for image_id in ids:
        image_name = get_filename_from_id(image_id)
        old_image_filename = os.path.join(os.path.dirname(__file__), orig_dir + image_name + extension)
        new_image_filename = os.path.join(os.path.dirname(__file__), new_dir + split_name + "/" + image_name + extension)
        os.rename(old_image_filename, new_image_filename)

def create_original_directory():
    split_ids_paths = {}
    split_ids = {}
    for split_name in SPLIT_NAMES:
        split_ids_paths[split_name] = os.path.join(os.path.dirname(__file__), './' + split_name + '_split.csv')
        
        if not os.path.exists(DATA_DIR_NEW + split_name):
            os.makedirs(DATA_DIR_NEW + split_name)
        if not os.path.exists(YOLO_LABELS_DIR_NEW + split_name):
            os.makedirs(YOLO_LABELS_DIR_NEW + split_name)
        if not os.path.exists(KITTI_LABELS_DIR_ORIG + split_name):
            os.makedirs(KITTI_LABELS_DIR_ORIG + split_name)
        
        split_ids[split_name] = np.genfromtxt(split_ids_paths[split_name], delimiter=",", dtype=None, encoding=None)
    
        move_data(split_ids[split_name], DATA_DIR_ORIG, DATA_DIR_NEW, split_name, ".png")
        move_data(split_ids[split_name], YOLO_LABELS_DIR_ORIG, YOLO_LABELS_DIR_NEW, split_name, ".txt")
        move_data(split_ids[split_name], KITTI_LABELS_DIR_ORIG, KITTI_LABELS_DIR_NEW, split_name, ".txt")

# def create_split_ids(num_per_split=NUM_PER_SPLIT):
#     label_dir = os.path.join(os.path.dirname(__file__), PROJ_DIR)   # Labels kept at proj directory level
#     train_ids, val_ids, test_ids = None, None, None
    
#     all_ids = np.array(range(NUM_LABELED))
#     train_ids = np.sort(np.random.choice(all_ids, size=num_train, replace=False))
#     non_train_ids = np.setdiff1d(all_ids, train_ids)
#     val_ids = np.sort(np.random.choice(non_train_ids, size=num_val, replace=False))
#     non_train_val_ids = np.setdiff1d(non_train_ids, val_ids)
#     test_ids = np.sort(np.random.choice(non_train_val_ids, size=num_test, replace=False))
    
#     np.savetxt(os.path.join(label_dir, "train_split.csv"), train_ids, delimiter=",", fmt='%06d')
#     np.savetxt(os.path.join(label_dir, "val_split.csv"), val_ids, delimiter=",", fmt='%06d')
#     np.savetxt(os.path.join(label_dir, "test_split.csv"), test_ids, delimiter=",", fmt='%06d')

#     assert train_ids.shape[0] == num_train
#     assert val_ids.shape[0] == num_val
#     assert test_ids.shape[0] == num_test

#     return train_ids, val_ids, test_ids

if __name__ == "__main__":
    # create_split_ids(NUM_PER_SPLIT)
    create_original_directory()

    