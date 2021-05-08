import numpy as np
import os
import random
from util import *

DATA_DIR = os.path.join(os.path.dirname(__file__), "./first20/")
LABELS_DIR = DATA_DIR + "labels/"
IMAGES_DIR = DATA_DIR + "images/"

#######################################
# Given a path, read the annotation file
def read_background_annotation(yolo_label_path: str) -> np.ndarray:
    yolo_label = np.genfromtxt(yolo_label_path, delimiter=" ", dtype=float, encoding=None)
    return clean_yolo(yolo_label) 

def randomly_choose_object_of_interest(num_annotations) -> int: # returns index
    return random.randint(0,num_annotations-1)


def get_top_left_bottom_right_coordinates(background_annotations, index):
    annotation = background_annotations[index]
    (xTopLeft, yTopLeft, width, height) = yolo_to_kitti(annotation)
    xBottomRight = xTopLeft + width
    yBottomRight = yTopLeft + height
    return ((xTopLeft, yTopLeft), (xBottomRight, yBottomRight))


def example_usage():
    image_name = "000001"
    yolo_label_path = LABELS_DIR + image_name + ".txt"
    yolo_image_path = IMAGES_DIR + image_name + ".png"

    yolo_label = read_background_annotation(yolo_label_path)
    rand_idx = randomly_choose_object_of_interest(yolo_label.shape[0])    
    ((xTopLeft, yTopLeft), (xBottomRight, yBottomRight)) = get_top_left_bottom_right_coordinates(yolo_label, rand_idx)
    show_bbox_tl_br(yolo_image_path, (xTopLeft, yTopLeft), (xBottomRight, yBottomRight))

if __name__ == "__main__":
    example_usage()



