import os
from constants import PeopleInts, VehicleInts
from util import *
from annotation_processing import *
from constants import *
import enum
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.patches as patches
import copy
from typing import Tuple

def plot_box(ax, entry, color='r',):
    width = entry[6]-entry[4]
    height = entry[7]-entry[5]
    rect = patches.Rectangle((entry[4], entry[5]), width, height, linewidth=1, edgecolor=color, facecolor='none')
    ax.add_patch(rect)

def show_all_labels(image_path, amodal_labels, modal_labels):
    image = np.array(plt.imread(image_path))
    fig, ax = plt.subplots()
    ax.imshow(image)
    
    # Draw each bounding box
    assert len(modal_labels) == len(amodal_labels)
    for label_idx in range((len(modal_labels))):
        modal_entry = modal_labels[label_idx]
        amodal_entry = amodal_labels[label_idx]
        if list(modal_entry) != list(amodal_entry):
            plot_box(ax, amodal_entry, 'r')
        plot_box(ax, modal_entry, 'b')
    
    plt.show()
    
def filter_full_overlap(modal_labels: np.ndarray) -> np.ndarray:
    modal_labels = modal_labels[modal_labels[:,4] != -1]
    return modal_labels

def label2_infront(label1, label2):
    obj1_class, _, obj1_occlusion, _, left1, top1, right1, bottom1 = label1[:8]
    obj2_class, _, obj2_occlusion, _, left2, top2, right2, bottom2 = label2[:8]

    if obj1_occlusion == OcclusionState['Unknown'] or obj2_occlusion == OcclusionState['Unknown']:
        return False

    # if int(obj1_class) == Classes['Van'] or int(obj2_class) == Classes['Van']:
    #     print(obj2_occlusion, obj1_occlusion)
    obj1_more_occluded = (obj2_occlusion < obj1_occlusion)

    # label1_area = (right1-left1)*(bottom1-top1)
    # label2_area = (right2-left2)*(bottom2-top2)
    # obj2_larger_area = (label2_area > label1_area)

    same_class = (obj1_class == obj2_class)
    similar_class = (obj1_class in PeopleInts and obj2_class in PeopleInts) or (obj1_class in VehicleInts and obj2_class in VehicleInts)

    obj2_taller_same_class = same_class and ((bottom2-top2) > (bottom1-top1))
    obj2_taller_similar_class = similar_class and ((bottom2-top2) > (bottom1-top1))
    return obj1_more_occluded or obj2_taller_similar_class
    
def generate_modal_label(amodal_kitti_label: np.ndarray, modal_amodal_different_count: int) -> Tuple[np.ndarray, int]:
    modal_labels = copy.deepcopy(amodal_kitti_label)
    num_labels = len(modal_labels)

    for label1_idx in range(num_labels):
        for label2_idx in range(num_labels):
            if label1_idx != label2_idx:
                label1 = modal_labels[label1_idx]
                label2 = modal_labels[label2_idx]
                label1_occlusion = label1[2]
                label2_occlusion = label2[2]
                label1_coords = label1[4:8]
                left1, top1, right1, bottom1 = label1_coords
                label2_coords = label2[4:8]
                left2, top2, right2, bottom2 = label2_coords
                
                # We assume that label 2 is the larger object
                modal_label1_coords = copy.deepcopy(label1_coords)
                if (left2 <= left1) and (right2 >= right1) and (top2 <= top1) and (bottom2 >= bottom1) and label2_occlusion < 0 and label1_occlusion >= 0:
                    # Occlusion fully covers non-occlusion object (if another object fully covers, we cannot be sure which is in front)
                    modal_label1_coords = [-1,-1,-1,-1]

                elif ((left2 <= left1) and (right2 >= right1)):
                    # Amodal -> Modal horizontal
                    if label2_infront(label1, label2):
                        new_top = bottom2 if (top1 < bottom2 and top1 > top2) else top1
                        new_bottom = top2 if (bottom1 < bottom2 and bottom1 > top2) else bottom1
                        modal_label1_coords[1] = new_top
                        modal_label1_coords[3] = new_bottom

                elif ((bottom2 >= bottom1) and (top2 <= top1)):
                    # Amodal -> Modal veritcal
                    if label2_infront(label1, label2):
                        new_left = right2 if (left1 > left2 and left1 < right2) else left1
                        new_right = left2 if (right1 > left2 and right1 < right2) else right1
                        modal_label1_coords[0] = new_left
                        modal_label1_coords[2] = new_right

                if list(modal_label1_coords) != list(label1_coords):
                    # TODO: append all, record how many are different
                    modal_amodal_different_count += 1

                modal_label = list(label1[:4]) + list(modal_label1_coords) + list(label1[8:])
                modal_labels[label1_idx] = modal_label
    
    modal_labels = filter_full_overlap(modal_labels)
    return modal_labels, modal_amodal_different_count

def write_modal_label_to_file(modal_label, filepath):
    np.savetxt(filepath, modal_label, delimiter=" ")

def create_all_modal_labels(images_dir, labels_dir, show_images=False, show_different_modal_amodal=False):
    modal_amodal_different_counts = {}  # { "train": Int, "val": Int, "test": Int }

    for dataset_split in os.scandir(labels_dir):
        if dataset_split.name == ".DS_Store":
            continue

        modal_amodal_different_count = 0
        for kitti_label_path in os.scandir(dataset_split):
            if kitti_label_path.name == ".DS_Store":
                continue

            if show_different_modal_amodal and kitti_label_path.name.find("006757.txt") == -1:
                continue
        
            amodal_kitti_label = read_background_annotation_kitti(kitti_label_path)
            modal_labels, modal_amodal_different_count = generate_modal_label(amodal_kitti_label, modal_amodal_different_count)
            
            image_path = os.path.join(os.path.dirname(__file__), images_dir + '/' + dataset_split.name + '/' + get_filename_without_ext(kitti_label_path) + '.png')
            
            if show_images:
                print(kitti_label_path.name)
                show_all_labels(image_path, amodal_kitti_label, modal_labels)

        modal_amodal_different_counts[dataset_split.name] = modal_amodal_different_count
    
    return modal_amodal_different_counts

def test_filter_full_overlap():
    modal_labels = np.array([[0., 0., 0., -2.23, 728.25, 173.97, 791.57, 200.2, 1.62, 1.72, 4.35, 9.71, 1.73, 47.16, -2.03],
                    [0., -1., 0., -1., 1001, 301, 1200, 320., -1., -1., -1., -1., -1., -1., -1.],
                    [0., -1., 0., -1., -1, -1, -1, -1, -1., -1., -1., -1., -1., -1., -1.],
                    [0., -1., 0., -1., 0, 0, 1000, 300., -1., -1., -1., -1., -1., -1., -1.]])
    modal_labels = filter_full_overlap(modal_labels)
    assert modal_labels.shape[0] == 3


def test_generate_modal_label():
    modal_labels = np.array([[0., 0., 0., -2.23, 728.25, 173.97, 791.57, 200.2, 1.62, 1.72, 4.35, 9.71, 1.73, 47.16, -2.03],
                    [0., -1., 0., -1., 1001, 301, 1200, 320., -1., -1., -1., -1., -1., -1., -1.],
                    [0., -1., -10, -1., 0, 0, 1000, 300., -1., -1., -1., -1., -1., -1., -1.]])
    modal_labels, _ = generate_modal_label(modal_labels, 0)
    assert modal_labels.shape[0] == 2


def run_create_modal_boxes_tests():
    test_filter_full_overlap()
    test_generate_modal_label()

if __name__ == "__main__":
    # run_create_modal_boxes_tests()
    # labels_dir = os.path.join(os.path.dirname(__file__), "../../cs231_project/data_object_image/training/label_2/")
    labels_dir = os.path.join(os.path.dirname(__file__), "./first20/kitti_labels/")
    images_dir = os.path.join(os.path.dirname(__file__), "./first20/images/")
    # labels_dir = os.path.join(os.path.dirname(__file__), "./data/kitti_labels/")
    # images_dir = os.path.join(os.path.dirname(__file__), "./data/images/")
    modal_amodal_different_counts = create_all_modal_labels(images_dir, labels_dir, show_images=True, show_different_modal_amodal=False)
    print(modal_amodal_different_counts)
