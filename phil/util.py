import os
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.patches as patches
import math
import setup
import time 
import pickle 
from constants import *
from args import get_train_test_args

def store_pickle(pickle_filename, data):
    pickle_file = open(pickle_filename, 'ab')
    pickle.dump(data, pickle_file)
    pickle_file.close()
    print("Stored in:", "../" + pickle_filename)

def load_pickle(pickle_filename):
    pickle_file = open(pickle_filename, 'rb')     
    data = pickle.load(pickle_file)
    pickle_file.close()
    return data

def get_name_from_id(image_id):
    str_id = str(image_id)
    return "0"*(6-len(str_id)) + str_id

def get_id_from_filename(dir_entry_filename):
    return int(dir_entry_filename.name[:-4])

def get_filename_from_name(dir_name, image_name="000001", extension=".txt"):
    return os.path.join(os.path.dirname(__file__), dir_name + image_name + extension)

def yolo_to_kitti(annotation):
    x, y, width, height = annotation[1], annotation[2], annotation[3], annotation[4]
    new_width = math.floor(width * IMG_WIDTH)
    new_height = math.floor(height * IMG_HEIGHT)
    left = math.floor(x * IMG_WIDTH) - new_width//2
    top = math.floor(y * IMG_HEIGHT) - new_height//2
    return (left, top, new_width, new_height)
    

EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), "../examples/")
def show_bbox_yolo(label=None, image_name="006145", rel_data_dir=EXAMPLES_DIR, extension=".jpeg"):
    image_filename = os.path.join(rel_data_dir, image_name) + extension
    image = np.array(plt.imread(image_filename))
    image = image[:IMG_HEIGHT, :IMG_WIDTH, :]
    
    fig, ax = plt.subplots()
    ax.imshow(image)

    # Draw each bounding box
    for annotation in label:
        (left, top, new_width, new_height) = yolo_to_kitti(annotation)
        rect = patches.Rectangle((left, top), new_width, new_height, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    plt.show()

def show_bbox_kitti(labels=None, image_id=6145, rel_data_dir=TRAIN_DIR):    
    image_filename = get_filename_from_name(rel_data_dir, get_name_from_id(image_id), ".png")
    image = np.array(plt.imread(image_filename))
    image = image[:IMG_HEIGHT, :IMG_WIDTH, :]
    
    fig, ax = plt.subplots()
    ax.imshow(image)

    # Draw each bounding box
    label = labels[image_id]
    for entry in label:
        width = entry[6]-entry[4]
        height = entry[7]-entry[5]
        rect = patches.Rectangle((entry[4], entry[5]), width, height, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    plt.show()

def show_bbox_tl_br(image_path, top_left_coords, bottom_right_coords):
    (xTopLeft, yTopLeft), (xBottomRight, yBottomRight) = top_left_coords, bottom_right_coords

    image = np.array(plt.imread(image_path))
    image = image[:IMG_HEIGHT, :IMG_WIDTH, :]
    
    fig, ax = plt.subplots()
    ax.imshow(image)

    # Draw each bounding box
    width = xBottomRight - xTopLeft
    height = yBottomRight - yTopLeft
    rect = patches.Rectangle((xTopLeft, yTopLeft), width, height, linewidth=1, edgecolor='r', facecolor='none', label='LABEL')
    ax.add_patch(rect)

    plt.show()

def crop_images(rel_data_dir=TRAIN_DIR, extension=".png", height=IMG_HEIGHT, width=IMG_WIDTH):
    data_dir = os.path.join(os.path.dirname(__file__), rel_data_dir)
    for idx, image_file in enumerate(os.scandir(data_dir)):
        if idx % 200 == 0:
            print(f"Processed images: {idx}")
        
        dot_idx = image_file.name.find(".")
        image_filename = get_filename_from_name(data_dir, image_file.name[:dot_idx], extension)
        image = np.array(plt.imread(image_filename))
        image = image[:IMG_HEIGHT, :IMG_WIDTH, :]
        plt.imsave(image_filename, image)

def clean_kitti(kitti_label):
    kitti_label = np.array(kitti_label)
    if kitti_label.shape == ():
        kitti_label = np.array([kitti_label])
    kitti_label = list(filter(lambda entry: entry[0] != 'DontCare', kitti_label))
    kitti_label = list(filter(lambda entry: entry[4] <= IMG_WIDTH and entry[5] <= IMG_HEIGHT, kitti_label))
    kitti_label = [list(entry)[1:] for entry in kitti_label]
    return np.array(kitti_label)

def clean_yolo(yolo_label):
    if len(yolo_label.shape) == 1:  # if there's only one label, shape will be (5,) which is 1-d, this fixes it.
        yolo_label = np.array([yolo_label])
    return yolo_label

# Concat two txt files (original and augmented annotations)
def concat_txts(txt1, txt2):
    pass

    

if __name__ == "__main__":
    # EXAMPLE BBOX
    # image_name = "000002_10.annotated"
    # label_file = get_filename_from_name(EXAMPLES_DIR, image_name, ".txt")
    # yolo_label = np.genfromtxt(label_file, delimiter=" ", dtype=float, encoding=None)
    # yolo_label = setup.clean_yolo(yolo_label)
    # show_bbox_yolo(yolo_label, image_name)

    # CROP IMAGES
    # crop_images("../data_object_image/training/first20/images/")
    pass