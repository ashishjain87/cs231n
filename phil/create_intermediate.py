# Creates an intermediate dataset that preserves all information
"""
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
    labels
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
   
"""

# Mix data sets
import os
from util import *
import shutil
import glob
import random


AUG_DIR = os.path.join(os.path.dirname(__file__), './augmented20/') # Augmented data
ORIG_DIR = os.path.join(os.path.dirname(__file__), './first20/') # Original data
INTERMEDIATE_DIR = os.path.join(os.path.dirname(__file__), './intermediate_data/')

def make_orig_dirs(intermediate_dir):
    for data_type in ['images', 'labels']:
        for split in ['train', 'val', 'test']:
            new_dir = INTERMEDIATE_DIR + '/' + data_type + '/' + split + '/orig/'
            if not os.path.exists(new_dir):
                os.makedirs(new_dir)

def copy_orig(orig_data_dir=ORIG_DIR, intermediate_dir=INTERMEDIATE_DIR):
    make_orig_dirs(intermediate_dir)

    orig_images_path = orig_data_dir + "images/"
    orig_labels_path = orig_data_dir + "labels/"

    orig_images = os.listdir(orig_images_path)
    orig_labels = os.listdir(orig_labels_path)
    if '.DS_Store' in orig_images:
        orig_images.remove('.DS_Store')

    if '.DS_Store' in orig_labels:
        orig_labels.remove('.DS_Store')

    for split in orig_images:
        orig_images_split_path = orig_images_path + split + '/'
        orig_labels_split_path = orig_labels_path + split + '/'
        
        for image_filename in os.listdir(orig_images_split_path):
            # Copy image
            s_img = os.path.join(os.path.dirname(__file__), orig_images_split_path + image_filename)
            d_img = os.path.join(intermediate_dir, "images/" + split + "/orig/")
            shutil.copy2(s_img, d_img)

            # Copy label
            image_name, _ = get_filename_and_extension(image_filename)
            s_label = os.path.join(os.path.dirname(__file__), orig_labels_split_path + image_name + '.txt')
            d_label = os.path.join(intermediate_dir, "labels/" + split + "/orig/")
            shutil.copy2(s_label, d_label)


def copy_aug(aug_data_dir_path=AUG_DIR, intermediate_dir=INTERMEDIATE_DIR):
    splits = os.listdir(aug_data_dir_path)
    if '.DS_Store' in splits:
        splits.remove('.DS_Store')


    for split in splits:
        split_path = aug_data_dir_path + split + '/'
        occlusion_classes = os.listdir(split_path)
        if '.DS_Store' in occlusion_classes:
            occlusion_classes.remove('.DS_Store')

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
                s_label = os.path.join(labels_dir_path, image_name + '.txt')
                d_label_dir = os.path.join(intermediate_dir, "labels/" + split + '/aug/' + occlusion_class + '/')
                if not os.path.exists(d_label_dir):
                    os.makedirs(d_label_dir)
                d_label = os.path.join(d_label_dir, image_name + '.txt')
                shutil.copy2(s_label, d_label)


if __name__ == "__main__":
    # Create output dirs
    if not os.path.exists(INTERMEDIATE_DIR):
        os.makedirs(INTERMEDIATE_DIR)
    
    # Copy original images into output dir
    copy_orig(ORIG_DIR, INTERMEDIATE_DIR)

    # Copy augmented images into output dir
    copy_aug(AUG_DIR, INTERMEDIATE_DIR)