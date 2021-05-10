import numpy as np
import os
import random
import glob
from util import *
from pathlib import Path

DATA_DIR = os.path.join(os.path.dirname(__file__), "./output/")
LABELS_DIR = DATA_DIR + "labels/"
IMAGES_DIR = DATA_DIR + "images/"



#######################################
# Given a path, read the annotation file
def read_background_annotation(yolo_label_path: str) -> np.ndarray:
    yolo_label = np.genfromtxt(yolo_label_path, delimiter=" ", dtype=float, encoding=None)
    return clean_yolo(yolo_label) 

def randomly_choose_object_of_interest(num_annotations) -> int: # returns index
    return random.randint(0,num_annotations-1)



def example_usage():
    image_name = "000001"
    yolo_label_path = LABELS_DIR + image_name + ".txt"
    yolo_image_path = IMAGES_DIR + image_name + ".png"

    yolo_label = read_background_annotation(yolo_label_path)
    rand_idx = randomly_choose_object_of_interest(yolo_label.shape[0])    
    ((xTopLeft, yTopLeft), (xBottomRight, yBottomRight)) = get_top_left_bottom_right_coordinates(yolo_label, rand_idx)
    show_bbox_tl_br(yolo_image_path, (xTopLeft, yTopLeft), (xBottomRight, yBottomRight))

def show_all():
    path_dir = IMAGES_DIR
    valid_extensions = ["png", "PNG"]
    image_paths = []
    for valid_extension in valid_extensions:
        search_path = path_dir + "/" + "*." + valid_extension
        for file_path in glob.glob(search_path):
            image_paths.append(file_path)

    for image_path in image_paths:
        image_filename, extension = get_filename_and_extension(image_path)

        orig_image_id, occlusion_id = tuple(image_filename.split('.'))
        target_annotations_file_name = '%s.annotated.%s.%s' % (orig_image_id, occlusion_id, 'txt') 
        target_original_file_name = '%s.%s' % (orig_image_id, 'txt') 
    
        target_path_annotations_file = os.path.join(LABELS_DIR, target_annotations_file_name)
        target_original_file = os.path.join(LABELS_DIR, target_original_file_name)
        annotation_label = read_background_annotation(target_path_annotations_file)
        orig_label = read_background_annotation(target_original_file)
        
        label = np.concatenate((orig_label, annotation_label), axis=0)
        show_bboxes(image_path, label)

if __name__ == "__main__":
    show_all()
