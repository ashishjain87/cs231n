import time 
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.patches as patches
import pandas as pd 
import os 
import pickle 
from util import *
from args import get_train_test_args
from constants import *


def config():
    args = get_train_test_args()
    args.labels_pickle = 'labels_pickle' if not args.new_labels else None

    plt.rcParams["figure.figsize"] = (10.0, 8.0)  # Set default size of plots.
    plt.rcParams["image.interpolation"] = "nearest"
    return args

# Just get one 
def get_yolo_label(yolo_label_path):
    yolo_label = np.genfromtxt(yolo_label_path, delimiter=" ", dtype=float, encoding=None)
    return clean_yolo(yolo_label) 

def get_yolo_labels(labels_pickle=None, rel_yolo_dir=LABELS_DIR_YOLO):
    """
    Returns
    y: Python dict of labels, indexed by their label_id.

    All values (numerical or strings) are separated via spaces,
    each row corresponds to one object. The 15 columns represent:

    #Values    Name      Description
    ----------------------------------------------------------------------------
    1       class       'Car': 0, 'Van': 1, 'Truck': 2,
                            'Pedestrian': 3, 'Person_sitting': 4, 'Cyclist': 5,
                            'Tram': 6, 'Misc': 7, 'DontCare': 8,
                            'PlasticBag': 10,
    1       x           x_center of object as % of image width (i.e. normalised 0-1)
    1       y           y_center of object as % of image height (i.e. normalised 0-1)
    1       width       object_bbox_width / image_width (i.e. normalised 0-1)
    1       height      object_bbox_height / image_height (i.e. normalised 0-1)
    """
    y = {}
    
    label_dir_yolo = os.path.join(os.path.dirname(__file__), rel_yolo_dir)
    for label_file in os.scandir(label_dir_yolo):
        label_id = get_id_from_filename(label_file)

        yolo_label = np.genfromtxt(label_file, delimiter=" ", dtype=float, encoding=None)
        yolo_label = clean_yolo(yolo_label)
        y[label_id] = yolo_label

    
    return y

def get_kitti(labels):
    """
    Returns
    y: Python dict of labels, indexed by their label_id.

    All values (numerical or strings) are separated via spaces,
    each row corresponds to one object. The 15 columns represent:

    #Values    Name      Description
    ----------------------------------------------------------------------------
    1       class       'Car': 0, 'Van': 1, 'Truck': 2,
                            'Pedestrian': 3, 'Person_sitting': 4, 'Cyclist': 5,
                            'Tram': 6, 'Misc': 7, 'DontCare': 8
    1       truncated   Float from 0 (non-truncated) to 1 (truncated), where
                            truncated refers to the object leaving image boundaries
    1       occluded    Integer (0,1,2,3) indicating occlusion state:
                            0 = fully visible, 1 = partly occluded
                            2 = largely occluded, 3 = unknown
    1       alpha       Observation angle of object, ranging [-pi..pi]
    4       bbox            2D bounding box of object in the image (0-based index):
                            contains left, top, right, bottom pixel coordinates
    3       dimensions  3D object dimensions: height, width, length (in meters)
    3       location    3D object location x,y,z in camera coordinates (in meters)
    1       rotation_y  Rotation ry around Y-axis in camera coordinates [-pi..pi]
    1       score       Only for results: Float, indicating confidence in
                            detection, needed for p/r curves, higher is better.
    """
    pass

def get_labels(labels_pickle=None, rel_yolo_dir=LABELS_DIR_YOLO, rel_kitti_dir=LABELS_DIR_KITTI):
    """
    Returns
    y: Python dict of labels, indexed by their label_id.

    All values (numerical or strings) are separated via spaces,
    each row corresponds to one object. The 15 columns represent:

    #Values    Name      Description
    ----------------------------------------------------------------------------
    1       class       'Car': 0, 'Van': 1, 'Truck': 2,
                            'Pedestrian': 3, 'Person_sitting': 4, 'Cyclist': 5,
                            'Tram': 6, 'Misc': 7, 'DontCare': 8
    ---- KITTI ----
    1       truncated   Float from 0 (non-truncated) to 1 (truncated), where
                            truncated refers to the object leaving image boundaries
    1       occluded    Integer (0,1,2,3) indicating occlusion state:
                            0 = fully visible, 1 = partly occluded
                            2 = largely occluded, 3 = unknown
    1       alpha       Observation angle of object, ranging [-pi..pi]
    4       bbox            2D bounding box of object in the image (0-based index):
                            contains left, top, right, bottom pixel coordinates
    3       dimensions  3D object dimensions: height, width, length (in meters)
    3       location    3D object location x,y,z in camera coordinates (in meters)
    1       rotation_y  Rotation ry around Y-axis in camera coordinates [-pi..pi]
    1       score       Only for results: Float, indicating confidence in
                            detection, needed for p/r curves, higher is better.
    ---- YOLO ----
    1       x           x_center of object as % of image width (i.e. normalised 0-1)
    1       y           y_center of object as % of image height (i.e. normalised 0-1)
    1       width       object_bbox_width / image_width (i.e. normalised 0-1)
    1       height      object_bbox_height / image_height (i.e. normalised 0-1)
    ---- UNDER CONSTRUCTION: Modal estimate ----
    1       x           x_center of modal object as % of image width (i.e. normalised 0-1)
    1       y           y_center of modal object as % of image height (i.e. normalised 0-1)
    1       width       object_bbox_width / image_width (i.e. normalised 0-1)
    1       height      object_bbox_height / image_height (i.e. normalised 0-1)
    """
    y = {}
    
    if labels_pickle is not None:
        y = load_pickle(labels_pickle)
    else:
        label_dir_yolo = os.path.join(os.path.dirname(__file__), rel_yolo_dir)
        for label_file in os.scandir(label_dir_yolo):
            label_id = get_id_from_filename(label_file)

            kitti_label_file = get_filename_from_name(rel_kitti_dir, get_name_from_id(label_id), ".txt")
            kitti_label = np.array(np.genfromtxt(kitti_label_file, delimiter=" ", dtype=None, encoding=None))
            kitti_label = clean_kitti(kitti_label)
            yolo_label = np.genfromtxt(label_file, delimiter=" ", dtype=float, encoding=None)
            yolo_label = clean_yolo(yolo_label)

            label = np.concatenate((yolo_label[:,0].reshape(-1,1), kitti_label, yolo_label[:,1:]), axis=1)
            y[label_id] = label

        store_pickle('labels_pickle', y)
    
    return y

def get_split_ids(generate_new_data_split, num_train=NUM_TRAIN, num_val=NUM_VAL, num_test=NUM_TEST):
    label_dir = os.path.join(os.path.dirname(__file__), PROJ_DIR)   # Labels kept at proj directory level
    train_ids, val_ids, test_ids = None, None, None
    
    if generate_new_data_split:
        all_ids = np.array(range(NUM_LABELED))
        train_ids = np.sort(np.random.choice(all_ids, size=num_train, replace=False))
        non_train_ids = np.setdiff1d(all_ids, train_ids)
        val_ids = np.sort(np.random.choice(non_train_ids, size=num_val, replace=False))
        non_train_val_ids = np.setdiff1d(non_train_ids, val_ids)
        test_ids = np.sort(np.random.choice(non_train_val_ids, size=num_test, replace=False))
        
        np.savetxt(os.path.join(label_dir, "train_split.csv"), train_ids, delimiter=",", fmt='%06d')
        np.savetxt(os.path.join(label_dir, "val_split.csv"), val_ids, delimiter=",", fmt='%06d')
        np.savetxt(os.path.join(label_dir, "test_split.csv"), test_ids, delimiter=",", fmt='%06d')
    else: 
        train_ids = np.genfromtxt(os.path.join(label_dir, "train_split.csv"), delimiter=",", dtype=None, encoding=None)
        val_ids = np.genfromtxt(os.path.join(label_dir, "val_split.csv"), delimiter=",", dtype=None, encoding=None)
        test_ids = np.genfromtxt(os.path.join(label_dir, "test_split.csv"), delimiter=",", dtype=None, encoding=None)

    assert train_ids.shape[0] == num_train
    assert val_ids.shape[0] == num_val
    assert test_ids.shape[0] == num_test

    return train_ids, val_ids, test_ids

def get_split_data(ids, data_dir, show_image=False):
    X = np.empty((len(ids), IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH))
    for idx, image_id in enumerate(ids):
        if idx % 200 == 0:
            print(f"Processed images: {idx}")

        image_filename = get_filename_from_name(data_dir, get_name_from_id(image_id), ".png")
        image = np.array(plt.imread(image_filename))
        image = image[:IMG_HEIGHT, :IMG_WIDTH, :]
        X[idx] = image
        
        if show_image and idx == 0:
            plt.imshow(image)
            plt.show()
    
    return X

def get_data(train_ids, val_ids, test_ids, num_train=NUM_TRAIN, num_val=NUM_VAL, num_test=NUM_TEST, rel_data_dir=TRAIN_DIR): # TODO: preprocessing subtract_mean=True
    '''
    Does not work with pickle: size too large.
    '''
    X_train, X_val, X_test = None, None, None 
    
    # if data_pickle is not None:
    #     X_train = load_pickle(data_pickle + "_train")
    #     X_val = load_pickle(data_pickle + "_val")
    #     X_test = load_pickle(data_pickle + "_test")
    # else:

    X_train = get_split_data(train_ids, rel_data_dir)
    X_val = get_split_data(val_ids, rel_data_dir)
    X_test = get_split_data(test_ids, rel_data_dir)

        # store_pickle('data_pickle' + '_train', X_train)
        # store_pickle('data_pickle' + '_val', X_val)
        # store_pickle('data_pickle' + '_test', X_test)

    return X_train, X_val, X_test



if __name__ == "__main__":
    args = config()

    # CONSTANTS FOR LOCAL FLAG (my poor baby laptop)
    num_train = 1000 if args.local else NUM_TRAIN
    num_val = 100 if args.local else NUM_VAL
    num_test = 100 if args.local else NUM_TEST

    # EXAMPLE GET/CREATE LABELS
    y = get_yolo_labels(args.labels_pickle)
    # y = get_labels(args.labels_pickle)
    # show_bbox_kitti(y, image_id=6757)

    # EXAMPLE GET DATA SPLIT
    train_ids, val_ids, test_ids = get_split_ids(args.new_data_split, num_train, num_val, num_test)
    
    # EXAMPLE GET DATA
    X_train, X_val, X_test = get_data(train_ids, val_ids, test_ids, num_train, num_val, num_test)



'''
TODO
- Count how many images don't have annotations.
- Display YOLO image fn (check yolov5 repo)
    Input: txt files with YOLO entries + augmented image
    Show labels
- Intersections between all bounding boxes
- kitti_aug file
- Trim images
'''