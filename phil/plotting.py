"""
Plotting Functions
"""
import os
from typing import List
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.patches as patches
import math
from constants import *
from pathlib import Path
from annotation_processing import *
from util import *


def show_bbox_yolo_paths(label_path: str, image_path: str):
    image = np.array(plt.imread(image_path))
    image = image[:IMG_HEIGHT, :IMG_WIDTH, :]
    label = read_background_annotation_yolo(label_path)

    fig, ax = plt.subplots()
    ax.imshow(image)

    # Draw each bounding box
    for idx, annotation in enumerate(label):
        (left, top, new_width, new_height) = rel_to_absolute_label(annotation)
        rect = patches.Rectangle((left, top), new_width, new_height, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    plt.show()


if __name__ == "__main__":
    # Show YOLO
    image_filepath = os.path.join(os.path.dirname(__file__), './intermediate_data/images/train/orig/005029.png')
    label_filepath = os.path.join(os.path.dirname(__file__), './intermediate_data/labels/train/orig/amodal/005029.txt')
    show_bbox_yolo_paths(label_filepath, image_filepath)