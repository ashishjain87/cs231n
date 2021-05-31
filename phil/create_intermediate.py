"""
Creates an intermediate dataset that preserves all information

------------------
Input Expectation:
------------------
augmented_dir
    train
        PlasticBag
            images
            labels
        PaperBag
	        images
            labels
    val
        PlasticBag
            images
            labels
        PaperBag
	        images
            labels
    test
        PlasticBag
            images
            labels
        PaperBag
	        images
            labels

original_dir
    images
        train
        val
        test
    yolo_labels
        train
        val
        test
    kitti_labels
        train
        val
        test

------------------
Output:
------------------
intermediate_dir
    images
        train
            aug
                PlasticBag
                PaperBag      
	        orig      
        val
            aug
                PlasticBag
                PaperBag      
	        orig      
        test
            aug
                PlasticBag
                PaperBag      
	        orig      
    labels
        train
            aug
                PlasticBag
                    modal
                    amodal
                PaperBag      
                    modal
                    amodal
	        orig      
                modal
                amodal 
        val
            aug
                PlasticBag
                    modal
                    amodal
                PaperBag      
                    modal
                    amodal
	        orig      
                modal
                amodal
        test
            aug
                PlasticBag
                    modal
                    amodal
                PaperBag      
                    modal
                    amodal
	        orig      
                modal
                amodal
   
"""
import os
from util import *
import shutil
from datetime import datetime
import numpy as np
import argparse
from pathlib import Path
from create_modal_boxes import *


def make_orig_dirs(intermediate_dir):
    for split in ['train', 'val', 'test']:
        new_dir = intermediate_dir + '/images/' + split + '/orig/'
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
        new_dir = intermediate_dir + '/images/' + split + '/orig/'
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)

    for split in ['train', 'val', 'test']:
        new_dir = intermediate_dir + '/labels/' + split + '/orig/amodal/'
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
        new_dir = intermediate_dir + '/labels/' + split + '/orig/modal/'
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)

def make_aug_dirs(intermediate_dir):
    for split in ['train', 'val', 'test']:
        new_dir = intermediate_dir + '/images/' + split + '/aug/'
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
        new_dir = intermediate_dir + '/images/' + split + '/aug/'
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)

    for split in ['train', 'val', 'test']:
        new_dir = intermediate_dir + '/labels/' + split + '/aug/'
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
        new_dir = intermediate_dir + '/labels/' + split + '/aug/'
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)

def check_modal_amodal(image_name, modal_yolo_label, amodal_yolo_label):
    assert amodal_yolo_label.shape[0] == modal_yolo_label.shape[0], f"Modal and Amodal labels have different number of annotations for image {image_name}"
    if modal_yolo_label.size != 0:
        assert ~np.any(modal_yolo_label[:,1:5] < 0), f"Modal box has negative value for image {image_name}."

def get_split_type(split: str) -> str:
    if split.find("train") != -1: return "train"
    if split.find("val") != -1: return "val"
    if split.find("test") != -1: return "test"


def copy_orig(orig_data_dir, intermediate_dir, aug_data_dir_path):
    # make_orig_dirs(intermediate_dir)

    orig_images_path = orig_data_dir + "images/"
    orig_labels_path_yolo = orig_data_dir + "yolo_labels/"
    orig_labels_path_kitti = orig_data_dir + "kitti_labels/"

    orig_labels_yolo = os.listdir(orig_labels_path_yolo)
    if '.DS_Store' in orig_labels_yolo:
        orig_labels_yolo.remove('.DS_Store')
    
    print(f"[{datetime.now()}] Starting original images")
    modal_amodal_different_counts = {}  # { "train": List[int, int], "val": List[int, int], "test": List[int, int] }
    for split in os.listdir(aug_data_dir_path):
        if split.find(".DS_Store") != -1:
            continue
        
        img_count = 0
        print(f"[{datetime.now()}] Starting {split}")
        split_type = get_split_type(split)

        orig_images_split_path = orig_images_path + split_type + '/'
        # orig_labels_yolo_split_path = orig_labels_path_yolo + split + '/'
        orig_labels_kitti_split_path = orig_labels_path_kitti + split_type + '/'
        
        modal_amodal_different_count = [0,0]
        image_filenames = os.listdir(orig_images_split_path)
        for image_filename in image_filenames: 
            if image_filename.find(".DS_Store") != -1:
                continue

            # Copy image
            s_img = os.path.join(os.path.dirname(__file__), orig_images_split_path + image_filename)
            d_dir = os.path.join(intermediate_dir, "images/" + split + "/orig/")
            if not os.path.exists(d_dir):
                os.makedirs(d_dir)
            shutil.copy2(s_img, d_dir)

            # Create and write amodal YOLO label
            image_name, _ = get_filename_and_extension(image_filename)
            amodal_kitti_label_path = os.path.join(os.path.dirname(__file__), orig_labels_kitti_split_path + image_name + '.txt')
            amodal_kitti_label = read_background_annotation_kitti(amodal_kitti_label_path)
            modal_kitti_label, amodal_kitti_label, modal_amodal_different_count = generate_modal_label(amodal_kitti_label, modal_amodal_different_count)
            modal_yolo_label = kitti_label_to_yolo_label(modal_kitti_label)
            amodal_yolo_label = kitti_label_to_yolo_label(amodal_kitti_label)
            
            modal_label_dir = os.path.join(intermediate_dir, "labels/" + split + "/orig/modal/")
            if not os.path.exists(modal_label_dir):
                os.makedirs(modal_label_dir)
            modal_label_filepath = modal_label_dir + image_name + '.txt'
            
            amodal_label_dir = os.path.join(intermediate_dir, "labels/" + split + "/orig/amodal/")
            if not os.path.exists(amodal_label_dir):
                os.makedirs(amodal_label_dir)
            amodal_label_filepath = amodal_label_dir + image_name + '.txt'
            write_label_to_file(modal_yolo_label, modal_label_filepath)
            write_label_to_file(amodal_yolo_label, amodal_label_filepath)

            check_modal_amodal(image_name, modal_yolo_label, amodal_yolo_label)
            img_count += 1
            if img_count % 100 == 0:
                print(f"[{datetime.now()}] Finished {img_count} images of {len(image_filenames)}")
            
        modal_amodal_different_counts[split] = modal_amodal_different_count
        
    print(f"[{datetime.now()}] Finished original images")
    print(f"[{datetime.now()}] Original modal-amodal-different Count: ", modal_amodal_different_counts)




def copy_aug(aug_data_dir_path, intermediate_dir, orig_data_dir):
    print(f"[{datetime.now()}] Starting to copy augmented images")
    modal_amodal_different_counts = {}  # { "train": List[int, int], "val": List[int, int], "test": List[int, int] }

    for split in os.listdir(aug_data_dir_path):
        if split.find('.DS_Store') != -1:
            continue

        print(f"[{datetime.now()}] Starting {split}")
        split_path = aug_data_dir_path + split + '/'
        modal_amodal_different_count = [0,0]
        img_count = 0

        occlusion_classes = os.listdir(split_path)
        for occlusion_class in occlusion_classes:
            if occlusion_class.find(".DS_Store") != -1:
                continue

            print(f"[{datetime.now()}] Starting {occlusion_class}")
            occlusion_class_path = split_path + occlusion_class + '/'
            images_dir_path = occlusion_class_path + 'images/'
            labels_dir_path = occlusion_class_path + 'labels/'

            image_filenames = os.listdir(images_dir_path)
            for image_filename in image_filenames:
                if image_filename.find(".DS_Store") != -1:
                    continue

                ##############
                # Copy image
                ##############
                s_img = os.path.join(images_dir_path, image_filename)
                d_img_dir = os.path.join(intermediate_dir, "images/" + split + '/aug/' + occlusion_class + '/')
                if not os.path.exists(d_img_dir):
                    os.makedirs(d_img_dir)
                d_img = os.path.join(d_img_dir, image_filename)
                shutil.copy2(s_img, d_img)

                ##############
                # Copy label
                ##############
                # Get occlusion yolo path
                image_name, _ = get_filename_and_extension(image_filename)
                image_id, occlusion_id = tuple(image_name.split('.'))
                occlusion_yolo_label_path = os.path.join(labels_dir_path, image_id + '.annotated.' + occlusion_id + '.txt')

                # Get occlusion and original kitti labels
                split_type = get_split_type(split)
                original_kitti_label_filepath = orig_data_dir + 'kitti_labels/' + split_type + '/' + image_id + '.txt'
                original_amodal_kitti_label = read_background_annotation_kitti(original_kitti_label_filepath)

                occlusion_yolo_label = read_background_annotation_yolo(occlusion_yolo_label_path)
                occlusion_amodal_kitti_label = annotated_yolo_to_kitti(occlusion_yolo_label)
                num_occlusions = occlusion_amodal_kitti_label.shape[0]

                # Combine occlusion with original label
                concat_amodal_kitti_label = np.concatenate((original_amodal_kitti_label, occlusion_amodal_kitti_label))

                # Get modal bboxes
                concat_modal_kitti_label, concat_amodal_kitti_label, modal_amodal_different_count = generate_modal_label(concat_amodal_kitti_label, modal_amodal_different_count)
                modal_kitti_label = concat_modal_kitti_label[:-num_occlusions,:]    # Discard occluder labels
                amodal_kitti_label = concat_amodal_kitti_label[:-num_occlusions,:]    # Discard occluder labels
                modal_yolo_label = kitti_label_to_yolo_label(modal_kitti_label)
                amodal_yolo_label = kitti_label_to_yolo_label(amodal_kitti_label)

                # Write amodal YOLO label
                d_label_amodal_dir = os.path.join(intermediate_dir, 'labels/' + split + '/aug/' + occlusion_class + '/amodal/')
                if not os.path.exists(d_label_amodal_dir):
                    os.makedirs(d_label_amodal_dir)
                d_label_amodal_filepath = os.path.join(d_label_amodal_dir, image_id + '.annotated.' + occlusion_id + '.txt')
                write_label_to_file(amodal_yolo_label, d_label_amodal_filepath)

                # Write modal YOLO label
                d_label_modal_dir = os.path.join(intermediate_dir, "labels/" + split + '/aug/' + occlusion_class + '/modal/')
                if not os.path.exists(d_label_modal_dir):
                    os.makedirs(d_label_modal_dir)
                d_label_modal_filepath = os.path.join(d_label_modal_dir, image_id + '.annotated.' + occlusion_id + '.txt')
                write_label_to_file(modal_yolo_label, d_label_modal_filepath)

                check_modal_amodal(image_name, modal_yolo_label, amodal_yolo_label)
                img_count += 1
                if img_count % 100 == 0:
                    print(f"[{datetime.now()}] Finished {img_count} images of {len(image_filenames)}")

        modal_amodal_different_counts[split] = modal_amodal_different_count
    
    print(f"[{datetime.now()}] Augmented modal-amodal-different Count: ", modal_amodal_different_counts)
    print(f"[{datetime.now()}] Finished augmented images")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--orig-root', type=str, help="top-level dir where original data is stored")
    parser.add_argument('--aug-root', type=str, help="top-level dir where augmented data is stored")
    parser.add_argument('--inter-root', type=str, help="top-level dir where intermediate data should be stored")
    return parser.parse_args()


if __name__ == "__main__":
    # Get args
    args = get_args()
    orig_path = os.path.join(os.path.dirname(__file__), args.orig_root + '/')
    aug_path = os.path.join(os.path.dirname(__file__), args.aug_root + '/')
    inter_path = os.path.join(os.path.dirname(__file__), args.inter_root + '/')

    # Create output dirs
    if not os.path.exists(inter_path):
        os.makedirs(inter_path)
    
    # Copy original images into output dir
    copy_orig(orig_path, inter_path, aug_path)

    # Copy augmented images into output dir
    copy_aug(aug_path, inter_path, orig_path)

"""
Example Usage:
python create_intermediate.py --orig-root first20_pre_intermediate --aug-root augmented20 --inter-root intermediate_data
python create_intermediate.py --orig-root data --aug-root augmented20 --inter-root intermediate_data
"""
