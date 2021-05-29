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

# Mix data sets
import os
from util import *
import shutil
import glob
import random
from datetime import datetime
from create_modal_boxes import *

AUG_DIR = os.path.join(os.path.dirname(__file__), './augmented20/') # Augmented data
ORIG_DIR = os.path.join(os.path.dirname(__file__), './first20_pre_intermediate/') # Original data
INTERMEDIATE_DIR = os.path.join(os.path.dirname(__file__), './intermediate_data/')




def make_orig_dirs(intermediate_dir):
    for split in ['train', 'val', 'test']:
        new_dir = INTERMEDIATE_DIR + '/images/' + split + '/orig/'
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
        new_dir = INTERMEDIATE_DIR + '/images/' + split + '/orig/'
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)

    for split in ['train', 'val', 'test']:
        new_dir = INTERMEDIATE_DIR + '/labels/' + split + '/orig/amodal/'
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
        new_dir = INTERMEDIATE_DIR + '/labels/' + split + '/orig/modal/'
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)




def copy_orig(orig_data_dir=ORIG_DIR, intermediate_dir=INTERMEDIATE_DIR):
    make_orig_dirs(intermediate_dir)

    orig_images_path = orig_data_dir + "images/"
    orig_labels_path_yolo = orig_data_dir + "yolo_labels/"
    orig_labels_path_kitti = orig_data_dir + "kitti_labels/"

    orig_labels_yolo = os.listdir(orig_labels_path_yolo)
    if '.DS_Store' in orig_labels_yolo:
        orig_labels_yolo.remove('.DS_Store')
    
    print("Starting original images")
    img_count = 0
    modal_amodal_different_counts = {}  # { "train": Int, "val": Int, "test": Int }
    for split in os.listdir(orig_images_path):
        print(f"[{datetime.now()}] Starting {split}")
        
        if split.find(".DS_Store") != -1:
            continue
        
        orig_images_split_path = orig_images_path + split + '/'
        orig_labels_yolo_split_path = orig_labels_path_yolo + split + '/'
        orig_labels_kitti_split_path = orig_labels_path_kitti + split + '/'
        
        modal_amodal_different_count = 0
        for image_filename in os.listdir(orig_images_split_path):
            if image_filename.find(".DS_Store") != -1:
                continue
            # if image_filename.find("006757") == -1:
            #     continue
            # Copy image
            s_img = os.path.join(os.path.dirname(__file__), orig_images_split_path + image_filename)
            d_img = os.path.join(intermediate_dir, "images/" + split + "/orig/")
            shutil.copy2(s_img, d_img)

            # Copy YOLO label
            image_name, _ = get_filename_and_extension(image_filename)
            s_label = os.path.join(os.path.dirname(__file__), orig_labels_yolo_split_path + image_name + '.txt')
            d_label = os.path.join(intermediate_dir, "labels/" + split + "/orig/amodal/")
            shutil.copy2(s_label, d_label)
            
            # Create and write amodal KITTI label
            amodal_kitti_label_path = os.path.join(os.path.dirname(__file__), orig_labels_kitti_split_path + image_name + '.txt')
            amodal_kitti_label = read_background_annotation_kitti(amodal_kitti_label_path)
            modal_kitti_label, modal_amodal_different_count = generate_modal_labels(amodal_kitti_label, modal_amodal_different_count)
            modal_yolo_label = kitti_to_yolo(modal_kitti_label)
            d_label_modal_filepath = os.path.join(intermediate_dir, "labels/" + split + "/orig/modal/" + image_name + '.txt')
            write_modal_label_to_file(modal_yolo_label, d_label_modal_filepath)

            img_count += 1
            if img_count % 100 == 0:
                print(f"Finished {img_count} images")
            
        modal_amodal_different_counts[split] = modal_amodal_different_count
        
    print("Finishes original images")
    print("Original modal-amodal-different Count: ", modal_amodal_different_counts)




def copy_aug(aug_data_dir_path=AUG_DIR, intermediate_dir=INTERMEDIATE_DIR):
    splits = os.listdir(aug_data_dir_path)
    if '.DS_Store' in splits:
        splits.remove('.DS_Store')

    img_count = 0
    print("Starting to copy augmented images")
    modal_amodal_different_counts = {}  # { "train": Int, "val": Int, "test": Int }

    for split in splits:
        print(f"[{datetime.now()}] Starting {split}")
        split_path = aug_data_dir_path + split + '/'
        occlusion_classes = os.listdir(split_path)
        if '.DS_Store' in occlusion_classes:
            occlusion_classes.remove('.DS_Store')

        modal_amodal_different_count = 0
        for occlusion_class in occlusion_classes:
            occlusion_class_path = split_path + occlusion_class + '/'
            images_dir_path = occlusion_class_path + 'images/'
            labels_dir_path = occlusion_class_path + 'labels/'

            image_filenames = os.listdir(images_dir_path)
            if '.DS_Store' in image_filenames:
                image_filenames.remove('.DS_Store')
            
            for image_filename in image_filenames:
                # Copy image
                s_img = os.path.join(images_dir_path, image_filename)
                d_img_dir = os.path.join(intermediate_dir, "images/" + split + '/aug/' + occlusion_class + '/')
                if not os.path.exists(d_img_dir):
                    os.makedirs(d_img_dir)
                d_img = os.path.join(d_img_dir, image_filename)
                shutil.copy2(s_img, d_img)

                # Copy label
                image_name, _ = get_filename_and_extension(image_filename)
                image_id, occlusion_id = tuple(image_name.split('.'))
                s_label = os.path.join(labels_dir_path, image_id + '.annotated.' + occlusion_id + '.txt')
                d_label_dir = os.path.join(intermediate_dir, "labels/" + split + '/aug/' + occlusion_class + '/amodal/')
                if not os.path.exists(d_label_dir):
                    os.makedirs(d_label_dir)
                d_label = os.path.join(d_label_dir, image_id + '.annotated.' + occlusion_id + '.txt')
                shutil.copy2(s_label, d_label)
            
                # Create and write amodal KITTI label
                yolo_label = read_background_annotation_yolo(d_label)
                amodal_kitti_label = annotated_yolo_to_kitti(yolo_label)
                modal_kitti_label, modal_amodal_different_count = generate_modal_labels(amodal_kitti_label, modal_amodal_different_count)
                modal_yolo_label = kitti_to_yolo(modal_kitti_label)

                d_label_modal_dir = os.path.join(intermediate_dir, "labels/" + split + '/aug/' + occlusion_class + '/modal/')
                if not os.path.exists(d_label_modal_dir):
                    os.makedirs(d_label_modal_dir)
                d_label_modal_filepath = os.path.join(d_label_modal_dir, image_id + '.annotated.' + occlusion_id + '.txt')
                write_modal_label_to_file(modal_yolo_label, d_label_modal_filepath)

                img_count += 1
                if img_count % 100 == 0:
                    print(f"Finished {img_count} images")

        modal_amodal_different_counts[split] = modal_amodal_different_count
    
    print("Augmented modal-amodal-different Count: ", modal_amodal_different_counts)
    print("Finished augmented images")




if __name__ == "__main__":
    # Create output dirs
    if not os.path.exists(INTERMEDIATE_DIR):
        os.makedirs(INTERMEDIATE_DIR)
    
    # Copy original images into output dir
    copy_orig(ORIG_DIR, INTERMEDIATE_DIR)

    # Copy augmented images into output dir
    copy_aug(AUG_DIR, INTERMEDIATE_DIR)