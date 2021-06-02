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
    TEXT_OFFSET = -10
    for idx in range(len(amodal_label)):
        amodal_annotation = amodal_label[idx]
        (aleft, atop, anew_width, anew_height) = rel_to_absolute_label(amodal_annotation)
        arect = patches.Rectangle((aleft, atop), anew_width, anew_height, linewidth=1, edgecolor='r', facecolor='none')
        ax[0].add_patch(arect)
        ax[0].text(aleft+TEXT_OFFSET, atop+TEXT_OFFSET, idx, backgroundcolor='w', fontsize='xx-small')
        ax[0].set_title('Amodal Bounding Boxes')
        
        modal_annotation = modal_label[idx]
        (mleft, mtop, mnew_width, mnew_height) = rel_to_absolute_label(modal_annotation)
        mrect = patches.Rectangle((mleft, mtop), mnew_width, mnew_height, linewidth=1, edgecolor='b', facecolor='none')
        ax[1].add_patch(mrect)
        ax[1].text(mleft+TEXT_OFFSET, mtop+TEXT_OFFSET, idx, backgroundcolor='w', fontsize='xx-small')
        ax[1].set_title('Modal Bounding Boxes')

    plt.show()


if __name__ == "__main__":
    # Show YOLO
    # image_filepath = os.path.join(os.path.dirname(__file__), './intermediate_data/images/train-SideAffixerSameClassSameImage/aug/Basketball/000008.28.png')
    # label_filepath = os.path.join(os.path.dirname(__file__), './intermediate_data/labels/train-SideAffixerSameClassSameImage/aug/Basketball/amodal/000008.annotated.28.txt')
    # show_bbox_yolo_paths(label_filepath, image_filepath)
    
    # image_filepath = os.path.join(os.path.dirname(__file__), './intermediate_data/images/train-SideAffixerSameClassSameImage/aug/Basketball/000008.28.png')
    # amodal_label_filepath = os.path.join(os.path.dirname(__file__), './intermediate_data/labels/train-SideAffixerSameClassSameImage/aug/Basketball/amodal/000008.annotated.28.txt')
    # modal_label_filepath = os.path.join(os.path.dirname(__file__), './intermediate_data/labels/train-SideAffixerSameClassSameImage/aug/Basketball/modal/000008.annotated.28.txt')
    # show_bbox_yolo_modal_amodal(amodal_label_filepath, modal_label_filepath, image_filepath)
    
    # image_filepath = os.path.join(os.path.dirname(__file__), './intermediate_data/images/train-SideAffixerSameClassSameImage/orig/000010.png')
    # amodal_label_filepath = os.path.join(os.path.dirname(__file__), './intermediate_data/labels/train-SideAffixerSameClassSameImage/orig/amodal/000010.txt')
    # modal_label_filepath = os.path.join(os.path.dirname(__file__), './intermediate_data/labels/train-SideAffixerSameClassSameImage/orig/modal/000010.txt')
    
    # image_filepath = os.path.join(os.path.dirname(__file__), './001792.18000603.png')
    # amodal_label_filepath = os.path.join(os.path.dirname(__file__), './001792.annotated.18000603.txt')
    # modal_label_filepath = os.path.join(os.path.dirname(__file__), '../001792.annotated.18000603.txt')
    
    # image_filepath = os.path.join(os.path.dirname(__file__), './train_test/004234.1016435.png')
    # amodal_label_filepath = os.path.join(os.path.dirname(__file__), './train_test/amodal/004234.annotated.1016435.txt')
    # modal_label_filepath = os.path.join(os.path.dirname(__file__), './train_test/modal/004234.annotated.1016435.txt')
    
    # image_filepath = '/Users/philipmateopfeffer/Downloads/cs231n_class_project/final/images/train/005003.3006671.png'
    # amodal_label_filepath = '/Users/philipmateopfeffer/Downloads/cs231n_class_project/final/labels/train/amodal/005003.3006671.txt'
    # modal_label_filepath = '/Users/philipmateopfeffer/Downloads/cs231n_class_project/final/labels/train/modal/005003.3006671.txt'
    
    # image_filepath = '/Users/philipmateopfeffer/Downloads/cs231n_class_project/final/images/train/001247.3003243.png'
    # amodal_label_filepath = '/Users/philipmateopfeffer/Downloads/cs231n_class_project/final/labels/train/amodal/001247.3003243.txt'
    # modal_label_filepath = '/Users/philipmateopfeffer/Downloads/cs231n_class_project/final/labels/train/modal/001247.3003243.txt'

    # image_filepath = '/Users/philipmateopfeffer/Downloads/cs231n_class_project/final/images/train/003767.2003886.png'
    # amodal_label_filepath = '/Users/philipmateopfeffer/Downloads/cs231n_class_project/final/labels/train/amodal/003767.2003886.txt'
    # modal_label_filepath = '/Users/philipmateopfeffer/Downloads/cs231n_class_project/final/labels/train/modal/003767.2003886.txt'
    
    # image_filepath = '/Users/philipmateopfeffer/Downloads/cs231n_class_project/cs231n/phil/intermediate_data/images/val-side-affixer-different-class/aug/Kite/000003.17000700.png'
    # amodal_label_filepath = '/Users/philipmateopfeffer/Downloads/cs231n_class_project/cs231n/phil/intermediate_data/labels/val-side-affixer-different-class/aug/Kite/amodal/000003.annotated.17000700.txt'
    # modal_label_filepath = '/Users/philipmateopfeffer/Downloads/cs231n_class_project/cs231n/phil/intermediate_data/labels/val-side-affixer-different-class/aug/Kite/modal/000003.annotated.17000700.txt'
    
    image_filepath = '/Users/philipmateopfeffer/Downloads/cs231n_class_project/cs231n/phil/fixed_final/images/val-side-affixer-different-class/000042.17000238.png'
    amodal_label_filepath = '/Users/philipmateopfeffer/Downloads/cs231n_class_project/cs231n/phil/fixed_final/labels/val-side-affixer-different-class/amodal/000042.17000238.txt'
    modal_label_filepath = '/Users/philipmateopfeffer/Downloads/cs231n_class_project/cs231n/phil/fixed_final/labels/val-side-affixer-different-class/modal/000042.17000238.txt'
    show_bbox_yolo_modal_amodal(amodal_label_filepath, modal_label_filepath, image_filepath)