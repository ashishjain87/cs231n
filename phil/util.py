import os
from typing import List
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.patches as patches
import math
import setup
import time 
import pickle 
from constants import *
from args import get_train_test_args
from pathlib import Path

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


def get_filename_without_ext(path):
    filename_with_ext = os.path.basename(path)
    list = os.path.splitext(filename_with_ext)
    return list[0]

def get_filename_and_extension(path): 
    path = Path(path)
    return (path.stem, path.suffix)

def rel_to_absolute_label(annotation):
    x, y, width, height = annotation[1], annotation[2], annotation[3], annotation[4]
    new_width = math.floor(width * IMG_WIDTH)
    new_height = math.floor(height * IMG_HEIGHT)
    left = math.floor(x * IMG_WIDTH) - new_width//2
    top = math.floor(y * IMG_HEIGHT) - new_height//2
    return (left, top, new_width, new_height)
    

EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), "./data/")
def show_bbox_yolo(label=None, image_name="006145", rel_data_dir=EXAMPLES_DIR, extension=".jpeg"):
    image_filename = os.path.join(rel_data_dir, 'images/train/' + image_name) + extension
    image = np.array(plt.imread(image_filename))
    image = image[:IMG_HEIGHT, :IMG_WIDTH, :]
    
    fig, ax = plt.subplots()
    ax.imshow(image)

    # Draw each bounding box

    for annotation in label:
        (left, top, new_width, new_height) = rel_to_absolute_label(annotation)
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


def get_top_left_bottom_right_coordinates(background_annotations, index):
    annotation = background_annotations[index]
    (xTopLeft, yTopLeft, width, height) = rel_to_absolute_label(annotation)
    xBottomRight = xTopLeft + width
    yBottomRight = yTopLeft + height
    return ((xTopLeft, yTopLeft), (xBottomRight, yBottomRight))


def show_bboxes(image_path, label):
    image = np.array(plt.imread(image_path))
    # image = image[:IMG_HEIGHT, :IMG_WIDTH, :]
    
    fig, ax = plt.subplots()
    ax.imshow(image)
    for idx, annotation in enumerate(label):
        ((xTopLeft, yTopLeft), (xBottomRight, yBottomRight)) = get_top_left_bottom_right_coordinates(label, idx)
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
    kitti_label = [[get_class_idx_from_entry(entry)] + list(entry)[1:] for entry in kitti_label]
    return np.array(kitti_label)

def clean_yolo(yolo_label):
    if len(yolo_label.shape) == 1:  # if there's only one label, shape will be (5,) which is 1-d, this fixes it.
        yolo_label = np.array([yolo_label])
    return yolo_label

def get_class_idx_from_entry(entry):
    return Classes[entry[0]]

def annotated_yolo_to_kitti(yolo_label: np.ndarray) -> np.ndarray:
    kitti_label = []
    for annotation in yolo_label:
        x, y, width, height = annotation[1:5]
        new_width = math.floor(width * IMG_WIDTH)
        new_height = math.floor(height * IMG_HEIGHT)
        left = math.floor(x * IMG_WIDTH) - new_width//2
        top = math.floor(y * IMG_HEIGHT) - new_height//2
        right = left + new_width
        bottom = top + new_height
        kitti_annotation = [annotation[0], -1, 0, -1, left, top, right, bottom, -1,-1,-1,-1,-1,-1]
        kitti_label.append(kitti_annotation)
    
    return np.array(kitti_label)



def convertToYoloBBox(bbox):
    # Yolo uses bounding bbox coordinates and size relative to the image size.
    # This is taken from https://pjreddie.com/media/files/voc_label.py .
    dw = 1. / IMG_WIDTH
    dh = 1. / IMG_HEIGHT
    
    if bbox[0] > IMG_WIDTH or bbox[1] > IMG_HEIGHT:
        return None
    right = min(bbox[2], 1224)
    bottom = min(bbox[3], 370)
    x = (bbox[0] + right) / 2.0
    y = (bbox[1] + bottom) / 2.0
    w = right - bbox[0]
    h = bottom - bbox[1]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def kitti_to_yolo(kitti_label: np.ndarray) -> np.ndarray:
    yolo_label = []
    for annotation in kitti_label:
        kitti_bbox = annotation[4:8]
        yolo_bbox = convertToYoloBBox(kitti_bbox)
        if yolo_bbox is None:
            continue
        (x, y, w, h) = yolo_bbox
        yolo_annotation = [annotation[0], x, y, w, h]
        yolo_label.append(yolo_annotation)
    
    return np.array(yolo_label)

if __name__ == "__main__":
    # EXAMPLE BBOX
    # image_name = "000002_10.annotated"
    # label_file = get_filename_from_name(EXAMPLES_DIR, image_name, ".txt")
    # yolo_label = np.genfromtxt(label_file, delimiter=" ", dtype=float, encoding=None)
    # show_bbox_yolo(yolo_label, image_name)
    
    image_name = "006145"
    label_file = os.path.join(EXAMPLES_DIR, 'labels/train/' + image_name + ".txt")
    yolo_label = np.genfromtxt(label_file, delimiter=" ", dtype=float, encoding=None)
    yolo_label = setup.clean_yolo(yolo_label)
    show_bbox_yolo(yolo_label, extension=".png")

    # CROP IMAGES
    # crop_images("./augmented20/images/")