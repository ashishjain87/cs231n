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

    _, ax = plt.subplots()
    ax.imshow(image)

    # Draw each bounding box
    for idx, annotation in enumerate(label):
        (left, top, new_width, new_height) = rel_to_absolute_label(annotation)
        rect = patches.Rectangle((left, top), new_width, new_height, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.text(left, top, idx, backgroundcolor='w', fontsize='xx-small')
    plt.show()

def show_bbox_yolo_modal_amodal(amodal_label_path: str, modal_label_path: str, image_path: str):
    image = np.array(plt.imread(image_path))
    image = image[:IMG_HEIGHT, :IMG_WIDTH, :]
    amodal_label = read_background_annotation_yolo(amodal_label_path)
    modal_label = read_background_annotation_yolo(modal_label_path)

    _, ax = plt.subplots(2)
    ax[0].imshow(image)
    ax[1].imshow(image)

    # Draw each bounding box
    for idx in range(len(amodal_label)):
        amodal_annotation = amodal_label[idx]
        (aleft, atop, anew_width, anew_height) = rel_to_absolute_label(amodal_annotation)
        arect = patches.Rectangle((aleft, atop), anew_width, anew_height, linewidth=1, edgecolor='r', facecolor='none')
        ax[0].add_patch(arect)
        ax[0].text(aleft, atop, idx, backgroundcolor='w', fontsize='xx-small')
        
        modal_annotation = modal_label[idx]
        (mleft, mtop, mnew_width, mnew_height) = rel_to_absolute_label(modal_annotation)
        mrect = patches.Rectangle((mleft, atop), mnew_width, mnew_height, linewidth=1, edgecolor='b', facecolor='none')
        ax[1].add_patch(mrect)
        ax[1].text(mleft, mtop, idx, backgroundcolor='w', fontsize='xx-small')

    plt.show()


if __name__ == "__main__":
    # Show YOLO
    # image_filepath = os.path.join(os.path.dirname(__file__), './intermediate_data/images/train-SideAffixerSameClassSameImage/aug/Basketball/000008.28.png')
    # label_filepath = os.path.join(os.path.dirname(__file__), './intermediate_data/labels/train-SideAffixerSameClassSameImage/aug/Basketball/amodal/000008.annotated.28.txt')
    # show_bbox_yolo_paths(label_filepath, image_filepath)
    
    image_filepath = os.path.join(os.path.dirname(__file__), './intermediate_data/images/train-SideAffixerSameClassSameImage/aug/Basketball/000008.28.png')
    amodal_label_filepath = os.path.join(os.path.dirname(__file__), './intermediate_data/labels/train-SideAffixerSameClassSameImage/aug/Basketball/amodal/000008.annotated.28.txt')
    modal_label_filepath = os.path.join(os.path.dirname(__file__), './intermediate_data/labels/train-SideAffixerSameClassSameImage/aug/Basketball/modal/000008.annotated.28.txt')
    show_bbox_yolo_modal_amodal(amodal_label_filepath, modal_label_filepath, image_filepath)