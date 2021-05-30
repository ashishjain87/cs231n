# Mix data sets
import os
from util import *
import shutil
import glob
import random

AUG_KEEP_FRACTION = 0.4
ORIG_IMAGES_DIR = os.path.join(os.path.dirname(__file__), './first20/images_orig/train') 
ORIG_LABELS_DIR = os.path.join(os.path.dirname(__file__), './first20/labels_orig/train') 
AUG_IMAGES_DIR = os.path.join(os.path.dirname(__file__), './augmented20/images/train')
AUG_LABELS_DIR = os.path.join(os.path.dirname(__file__), './augmented20/labels/train')
NEW_IMAGES_DIR = os.path.join(os.path.dirname(__file__), './mixed_data/images/train')
NEW_LABELS_DIR = os.path.join(os.path.dirname(__file__), './mixed_data/labels/train')

def copy_orig(orig_data_dir=ORIG_IMAGES_DIR, orig_labels_dir=ORIG_LABELS_DIR, new_data_dir=NEW_IMAGES_DIR, new_labels_dir=NEW_LABELS_DIR):
    orig_files = os.listdir(orig_data_dir)
    if '.DS_Store' in orig_files:
        orig_files.remove('.DS_Store')
    
    for file_name in orig_files:
        # Copy image
        s_img = os.path.join(orig_data_dir, file_name)
        d_img = os.path.join(new_data_dir, file_name)
        shutil.copy2(s_img, d_img)

        # Copy label
        image_name, _ = get_filename_and_extension(file_name)
        s_label = os.path.join(orig_labels_dir, image_name + '.txt')
        d_label = os.path.join(new_labels_dir, image_name + '.txt')
        shutil.copy2(s_label, d_label)

def copy_aug(orig_data_dir=ORIG_IMAGES_DIR, orig_labels_dir=ORIG_LABELS_DIR, 
                    aug_data_dir=AUG_IMAGES_DIR, aug_labels_dir=AUG_LABELS_DIR, 
                    new_data_dir=NEW_IMAGES_DIR, new_labels_dir=NEW_LABELS_DIR, aug_keep_fraction=AUG_KEEP_FRACTION):

    aug_files = os.listdir(aug_data_dir)
    if '.DS_Store' in aug_files:
        aug_files.remove('.DS_Store')
    
    num_aug = round(aug_keep_fraction * len(aug_files))
    chosen_aug_files = random.sample(aug_files, num_aug)

    for file_name in chosen_aug_files:
        # Copy image
        s_img = os.path.join(aug_data_dir, file_name)
        d_img = os.path.join(new_data_dir, file_name)
        shutil.copy2(s_img, d_img)

        # Copy label
        # NOTE: USES ORIGINAL DATA LABEL
        image_name, _ = get_filename_and_extension(file_name)
        orig_image_id, occlusion_id = tuple(image_name.split('.'))
        s_label = os.path.join(orig_labels_dir, orig_image_id + '.txt')
        d_label = os.path.join(new_labels_dir, image_name + '.txt')
        shutil.copy2(s_label, d_label)

    

if __name__ == "__main__":
    # Create output dirs
    if not os.path.exists(NEW_IMAGES_DIR):
        os.makedirs(NEW_IMAGES_DIR)
    
    if not os.path.exists(NEW_LABELS_DIR):
        os.makedirs(NEW_LABELS_DIR)
    
    # Copy original images into output dir
    copy_orig(ORIG_IMAGES_DIR, ORIG_LABELS_DIR, NEW_IMAGES_DIR, NEW_LABELS_DIR)

    # Copy augmented images into output die
    copy_aug(ORIG_IMAGES_DIR, ORIG_LABELS_DIR, AUG_IMAGES_DIR, AUG_LABELS_DIR, NEW_IMAGES_DIR, NEW_LABELS_DIR)