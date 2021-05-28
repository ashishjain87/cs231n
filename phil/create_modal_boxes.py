import os
from util import *
from annotation_processing import *
import enum
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.patches as patches
import copy

def plot_box(ax, entry, color='r',):
    width = entry[5]-entry[3]
    height = entry[6]-entry[4]
    rect = patches.Rectangle((entry[3], entry[4]), width, height, linewidth=1, edgecolor=color, facecolor='none')
    ax.add_patch(rect)

def show_all_labels(image_path, amodal_labels, modal_labels):
    image = np.array(plt.imread(image_path))
    
    fig, ax = plt.subplots()
    ax.imshow(image)

    # Draw each bounding box
    print(len(modal_labels))
    print(modal_labels)
    print(len(amodal_labels))
    assert len(modal_labels) == len(amodal_labels)
    for label_idx in range((len(modal_labels))):
        modal_entry = modal_labels[label_idx]
        amodal_entry = amodal_labels[label_idx]
        if list(modal_entry) != list(amodal_entry):
            plot_box(ax, amodal_entry, 'r')
        plot_box(ax, modal_entry, 'b')
    
    plt.show()
    pass


# Using enum class create enumerations
class OcclusionState(enum.Enum):
   FullyVisible = 0
   PartlyOccluded = 1
   LargelyOccluded = 2
   Unknown = 3

def label2_infront(label1, label2):
    # Default to not occluded if we don't know or if occlusion state is the same
    obj1_occlusion = label1[2]
    obj2_occlusion = label2[2]
    
    if obj1_occlusion == OcclusionState.Unknown or obj2_occlusion == OcclusionState.Unknown:
        return False

    return obj2_occlusion < obj1_occlusion
    
def generate_modal_labels(amodal_kitti_label, modal_amodal_different_count):
    modal_labels = copy.deepcopy(amodal_kitti_label)

    for label1_idx in range(len(modal_labels)):
        for label2_idx in range(label1_idx+1, len(modal_labels)):
            label1 = modal_labels[label1_idx]
            label2 = modal_labels[label2_idx]
            
            label1_coords = label1[3:7]
            left1, top1, right1, bottom1 = label1_coords
            label2_coords = label2[3:7]
            left2, top2, right2, bottom2 = label2_coords
            
            # We assume that label 2 is the larger object
            modal_label1_coords = copy.deepcopy(label1_coords)
            if ((left2 <= left1) and (right2 >= right1)):
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

            modal_label = list(label1[:3]) + list(modal_label1_coords) + list(label1[7:])
            modal_labels[label1_idx] = modal_label

    return modal_labels, modal_amodal_different_count

def create_modal_labels(images_dir, labels_dir, show_images=False):
    modal_amodal_different_counts = {}  # { "train": Int, "val": Int, "test": Int }

    for dataset_split in os.scandir(labels_dir):
        if dataset_split.name == ".DS_Store":
            continue

        modal_amodal_different_count = 0
        for kitti_label_path in os.scandir(dataset_split):
            if kitti_label_path.name == ".DS_Store":
                continue

            if kitti_label_path.name.find("006757.txt") == -1:
                continue
        
            amodal_kitti_label = read_background_annotation_kitti(kitti_label_path)
            modal_labels, modal_amodal_different_count = generate_modal_labels(amodal_kitti_label, modal_amodal_different_count)
            
            image_path = os.path.join(os.path.dirname(__file__), images_dir + '/' + dataset_split.name + '/' + get_filename_without_ext(kitti_label_path) + '.png')
            
            if show_images:
                show_all_labels(image_path, amodal_kitti_label, modal_labels)

        modal_amodal_different_counts[dataset_split.name] = modal_amodal_different_count
    
    return modal_amodal_different_counts

if __name__ == "__main__":
    # labels_dir = os.path.join(os.path.dirname(__file__), "../../cs231_project/data_object_image/training/label_2/")
    labels_dir = os.path.join(os.path.dirname(__file__), "./first20/kitti_labels/")
    images_dir = os.path.join(os.path.dirname(__file__), "./first20/images/")
    modal_amodal_different_counts = create_modal_labels(images_dir, labels_dir, show_images=True)
    print(modal_amodal_different_counts)
