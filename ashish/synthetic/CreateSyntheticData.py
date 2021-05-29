import argparse
import ast 
import confuse
import csv
import Occlusions as occlusions
import glob
import json
import logging
import logging.config
import math
import ntpath
import numpy as np
import os
import random
import sys
import tqdm
import yaml
import SyntheticConfig as synthetic_config
import Startup

from ImageTransformer import ImageTransformer
from NoOpImageTransformer import NoOpImageTransformer
from RandomRotator import RandomRotator
from RandomHorizontalFlipper import RandomHorizontalFlipper
from RandomVerticalFlipper import RandomVerticalFlipper
from DecoratorImageTransformer import DecoratorImageTransformer
from Affixer import Affixer
from OriginalAffixer import OriginalAffixer
from ImagePath import *
from Annotation import *
from SyntheticImage import *

from PIL import Image, ImageOps
from pathlib import Path

def randomly_sample_new_image_size(fg_image_size, bg_image_size):
    logger = logging.getLogger(__name__)

    (fg_width, fg_height) = fg_image_size
    (bg_width, bg_height) = bg_image_size

    logger.debug('fg_size: (%i, %i)', fg_width, fg_height) 
    logger.debug('bg_size: (%i, %i)', bg_width, bg_height) 

    low = 0.80 # TODO: Move to config
    high = 2.8 # TODO: Move to config

    # We are working with wide images. Therefore, worrying about the height is sufficient.
    min_height = math.floor(0.10*bg_height)
    max_height = math.floor(0.60*bg_height)

    low = min_height*1.0/fg_height
    high = max_height*1.0/fg_height

    logger.debug('min_height: %i, max_height: %i, low: %f, high: %f', min_height, max_height, low, high) 

    # We preserve the aspect ratio for the foreground image. Therefore, scale remains the same across width and height 
    scale = random.uniform(low, high) # [low, high]
    
    scaled_fg_width = int(scale * fg_width)
    scaled_fg_height = int(scale * fg_height)

    logger.debug('scale: %f, scaled_fg_size: (%i, %i)', scale, scaled_fg_width, scaled_fg_height) 

    return (scaled_fg_width, scaled_fg_height)

def resize_image_randomly(pil_image_original, pil_image_background):
    logger = logging.getLogger(__name__)

    new_size = randomly_sample_new_image_size(pil_image_original.size, pil_image_background.size)
    pil_image_resized = ImageOps.fit(pil_image_original, new_size, Image.ANTIALIAS)

    head, tail = ntpath.split(pil_image_original.filename)
    original_filename_with_extension = tail or ntpath.basename(head)
    original_filename, file_extension = os.path.splitext(original_filename_with_extension)

    (new_width, new_height) = new_size
    pil_image_resized.filename = '%s_width_%d_height_%d%s' % (original_filename, new_width, new_height, file_extension)
    logger.debug('From image %s created new image in memory with name %s', original_filename_with_extension, pil_image_resized.filename) 

    return pil_image_resized

def determine_new_image_size(fg_image_size, bg_image_size, resize_scale):
    logger = logging.getLogger(__name__)

    (fg_width, fg_height) = fg_image_size
    (bg_width, bg_height) = bg_image_size

    logger.debug('fg_size: (%i, %i)', fg_width, fg_height) 
    logger.debug('bg_size: (%i, %i)', bg_width, bg_height) 
    
    scaled_fg_width = int(resize_scale * fg_width)
    scaled_fg_height = int(resize_scale * fg_height)

    if scaled_fg_height >= bg_height:
        logger.error('scaled_fg_height is gte bg_height specifically %i is gte %i', scaled_fg_height, bg_height) 

    if scaled_fg_width >= bg_width:
        logger.error('scaled_fg_width is gte bg_width specifically %i is gte %i', scaled_fg_width, bg_width) 

    logger.debug('scale: %f, scaled_fg_size: (%i, %i)', resize_scale, scaled_fg_width, scaled_fg_height) 

    return (scaled_fg_width, scaled_fg_height)

def resize_image(pil_image_original, pil_image_background, resize_scale):
    logger = logging.getLogger(__name__)

    new_size = determine_new_image_size(pil_image_original.size, pil_image_background.size, resize_scale)
    pil_image_resized = ImageOps.fit(pil_image_original, new_size, Image.ANTIALIAS)

    head, tail = ntpath.split(pil_image_original.filename)
    original_filename_with_extension = tail or ntpath.basename(head)
    original_filename, file_extension = os.path.splitext(original_filename_with_extension)

    (new_width, new_height) = new_size
    pil_image_resized.filename = '%s_width_%d_height_%d%s' % (original_filename, new_width, new_height, file_extension)
    logger.debug('From image %s created new image in memory with name %s', original_filename_with_extension, pil_image_resized.filename) 

    return pil_image_resized

def randomly_sample_point_within_circle(xCenter, yCenter, radius):
    logger = logging.getLogger(__name__)
    logger.debug('xCenter is %s, yCenter is %s, feasible radius is %s', xCenter, yCenter, radius)

    # Determine angle
    angle_degrees = random.uniform(0, 360) # inclusive of min, max 
    angle_radians = np.radians(angle_degrees) 

    # Determine radius
    radius = random.uniform(0, radius) # inclusive of min, max 

    # Determine x, y
    x_relative = radius * math.cos(angle_radians)
    y_relative = radius * math.sin(angle_radians)

    x_relative = math.floor(x_relative)
    y_relative = math.floor(y_relative)

    x_absolute = xCenter + x_relative
    y_absolute = yCenter - y_relative

    logger.debug('random sample angle degrees %s, radius %s, relative (%s, %s), absolute (%s, %s)', angle_degrees, radius, x_relative, y_relative, x_absolute, y_absolute)

    return (x_absolute, y_absolute)

def compute_target_path(target_dir, path_background_image, cur_image_id):
    # Compute target path
    (file_name, file_ext) = get_filename_and_extension(path_background_image)
    file_name = '%s.%s%s' % (file_name, cur_image_id, file_ext) 
    target_path = os.path.join(target_dir, file_name)
    return (file_name, target_path)

def get_image_info(synthetic_image, file_name, image_id):
    (width, height) = synthetic_image.size

    dict = {}
    dict['file_name'] = file_name 
    dict['license'] = 1 # hardcoded
    dict['width'] = width 
    dict['height'] = height 
    dict['id'] = image_id 

    return dict

def process_resized_occlusion_image(
    occlusion_image,
    occlusion_name,
    target_dir,
    background_image,
    background_annotation,
    path_background_image,
    threshold,
    cur_image_id,
    target_annotations_dir,
    occlusion_name_occlusion_id_dict,
    center_point,
):
    logger = logging.getLogger(__name__)

    (file_name, target_path) = compute_target_path(target_dir, path_background_image, cur_image_id)

    # synthetic image generation and save
    (synthetic_image, point, original_occlusion_image_grayscale_image_mask) = create_synthetic_image(
        background_image,
        occlusion_image,
        threshold,
        center_point
    )

    save_synthetic_image(
        background_image,
        synthetic_image,
        occlusion_image,
        point,
        threshold,
        target_path,
    )

    # image info
    image_info = get_image_info(synthetic_image, file_name, cur_image_id)

    # Generate annotation and save
    image_annotation = compute_annotation(
        background_image,
        synthetic_image,
        occlusion_image,
        point,
        original_occlusion_image_grayscale_image_mask,
    )
    save_image_annotation(
        path_background_image,
        image_annotation,
        target_annotations_dir,
        cur_image_id,
        occlusion_name_occlusion_id_dict,
        occlusion_name,
    )

    logger.debug("image_info, annotation created for %s", image_info["file_name"])

    cur_image_id += 1

    return (image_annotation, image_info, cur_image_id)

def process_original_occlusion_image(
    path_occlusion_image,
    path_background_image,
    background_annotations_file_path,
    threshold,
    target_dir_path_images,
    target_dir_path_annotations,
    cur_image_id,
    occlusion_name_occlusion_id_dict,
    probability_prioritize_objects_of_interest,
):
    logger = logging.getLogger(__name__)

    occlusion_image_filename = get_filename_without_ext(path_occlusion_image)
    occlusion_image = Image.open(path_occlusion_image)
    occlusion_name = get_immediate_parent_folder(path_occlusion_image)

    background_image = Image.open(path_background_image)
    background_annotation = read_background_annotation(background_annotations_file_path)  # Integration with Phil

    image_info_collection = []
    image_annotation_collection = []

    num_runs_per_original_image = 1  # TODO: Move to config

    occlusion_name = get_immediate_parent_folder(path_occlusion_image)

    transformer: ImageTransformer= DecoratorImageTransformer([RandomVerticalFlipper(), RandomHorizontalFlipper(), RandomRotator()])
    affixer: Affixer = OriginalAffixer(probability_prioritize_objects_of_interest)

    for i in range(num_runs_per_original_image):
        logger.debug("For occlusion %s, image %s, run %d of %d", occlusion_name, occlusion_image_filename, i + 1, num_runs_per_original_image)
        occlusion_image_transformed = transformer.transform(occlusion_image)
        (center_point, resize_scale) = affixer.decide_where_and_scale(background_image, background_annotation, occlusion_image_transformed)
        occlusion_image_resized = resize_image(occlusion_image_transformed, background_image, resize_scale)  # specify the gaussian and standard deviation
        # TODO: pass corresponding annotations file
        (image_annotation, image_info, cur_image_id) = process_resized_occlusion_image(
            occlusion_image_resized,
            occlusion_name,
            target_dir_path_images,
            background_image,
            background_annotation,
            path_background_image,
            threshold,
            cur_image_id,
            target_dir_path_annotations,
            occlusion_name_occlusion_id_dict,
            center_point,
        )
        image_info_collection.append(image_info)
        image_annotation_collection.append(image_annotation)

    return (image_annotation_collection, image_info_collection, cur_image_id)

def create_synthetic_images_for_all_images_under_current_folders(
    subdir,
    background_dir_path_images,
    background_dir_path_annotations,
    path_foreground_dir,
    threshold,
    target_dir_path_images,
    target_dir_path_annotations,
    cur_image_id,
    occlusion_name_occlusion_id_dict,
    probability_prioritize_objects_of_interest,
):
    logger = logging.getLogger(__name__)

    if not os.path.isdir(target_dir_path_images):
        os.makedirs(target_dir_path_images)
        logger.info("Created target directory %s", target_dir_path_images)

    if not os.path.isdir(target_dir_path_annotations):
        os.makedirs(target_dir_path_annotations)
        logger.info("Created target directory %s", target_dir_path_annotations)

    logger.info(
        "create_synthetic_images - background images: %s, background annotations: %s, foreground: %s, image output: %s, annotations output: %s",
        background_dir_path_images,
        background_dir_path_annotations,
        path_foreground_dir,
        target_dir_path_images,
        target_dir_path_annotations,
    )

    # TODO: Move to config
    foregound_valid_extensions = [
        "jpg",
        "jpeg",
        "JPEG",
        "JPG",
        "png",
        "PNG",
    ]  # image masking code doesn't work well with PNGs

    foreground_image_paths = []
    for valid_extension in foregound_valid_extensions:
        foreground_search_path = path_foreground_dir + "/" + "*." + valid_extension
        for file_path in glob.glob(foreground_search_path):
            foreground_image_paths.append(file_path)
            logger.debug("Found foreground image at: %s", file_path)

    # TODO: Move to config
    background_valid_extensions = ["jpg", "jpeg", "JPEG", "JPG", "png", "PNG"]
    background_image_paths = []
    for valid_extension in background_valid_extensions:
        background_search_path = (background_dir_path_images + "/" + "*." + valid_extension)
        for file_path in glob.glob(background_search_path):
            background_image_paths.append(file_path)

    image_info_collection = []
    image_annotation_collection = []

    if len(foreground_image_paths) == 0:
        logger.warn("No foreground images found")

    if len(background_image_paths) == 0:
        logger.warn("No background images found")

    for foreground_image_path in tqdm.tqdm(foreground_image_paths, desc=subdir, leave=True):
        logger.debug("Processing foreground image: %s", foreground_image_path)
        for background_image_path in background_image_paths:
            logger.debug("Processing background image: %s", background_image_path)
            background_annotation_file_path = get_background_annotation_file_path(background_image_path, background_dir_path_annotations)
            (image_annotations, image_infos, cur_image_id) = process_original_occlusion_image(
                foreground_image_path,
                background_image_path,
                background_annotation_file_path,
                threshold,
                target_dir_path_images,
                target_dir_path_annotations,
                cur_image_id,
                occlusion_name_occlusion_id_dict,
                probability_prioritize_objects_of_interest,
            )

            image_annotation_collection.append(image_annotations)
            image_info_collection.append(image_infos)

    return (image_annotation_collection, image_info_collection, cur_image_id)

def create_synthetic_images_for_all_direct_subfolders(syntheticConfig, occlusion_name_occlusion_id_dict):
    logger = logging.getLogger(__name__)

    image_annotation_collection = []
    image_info_collection = []
    cur_image_id = syntheticConfig.cur_image_id

    subdirs = get_immediate_subdirectories(syntheticConfig.path_foreground_super_dir)
    for subdir in tqdm.tqdm(subdirs, desc="Overall", position=0):
        logger.info("Processing %s", subdir)
        path_foreground_dir = os.path.join(syntheticConfig.path_foreground_super_dir, subdir)
        target_dir_path_images = os.path.join(syntheticConfig.target_dir, subdir, syntheticConfig.images_dir_name)
        target_dir_path_annotations = os.path.join(syntheticConfig.target_dir, subdir, syntheticConfig.annotations_dir_name)

        logger.debug('path_foreground_dir: %s', path_foreground_dir)
        logger.debug('target_dir_path_images: %s', target_dir_path_images)
        logger.debug('target_dir_path_annotations: %s', target_dir_path_annotations)

        (image_annotations, image_infos, cur_image_id) = create_synthetic_images_for_all_images_under_current_folders( \
                subdir, \
                syntheticConfig.background_dir_path_images, \
                syntheticConfig.background_dir_path_annotations, \
                path_foreground_dir, \
                syntheticConfig.threshold, \
                target_dir_path_images, \
                target_dir_path_annotations, \
                cur_image_id, \
                occlusion_name_occlusion_id_dict, \
                syntheticConfig.probability_prioritize_objects_of_interest)

        logger.info('Subdirectory %s processed', subdir)
        logger.info('Number of images created for %s is %s', subdir, len(image_infos))
        logger.info('Number of annotations created for %s is %s', subdir, len(image_annotations))

        image_annotation_collection.append(image_annotations)
        image_info_collection.append(image_infos)

    logger.info('Total number of images created is %s', len(image_info_collection))
    logger.info('Total number of annotations created is %s', len(image_annotation_collection))

def main():
    startup = Startup.Startup()
    startup.configure()

    logger = logging.getLogger(__name__)
    logger.info('Started')

    syntheticConfig = startup.synthetic_config

    #process_original_occlusion_image(path_foreground_file, path_background_file, threshold, target_dir, cur_image_id, occlusion_name) 
    # TODO: pass background annotations directory
    #create_synthetic_images_for_all_images_under_current_folders(syntheticConfig.background_dir_path_images, syntheticConfig.background_dir_path_annotations, syntheticConfig.path_foreground_dir, syntheticConfig.threshold, syntheticConfig.target_dir_path_images, syntheticConfig.target_dir_path_annotations, syntheticConfig.cur_image_id, startup.occlusion_name_occlusion_id_dict, syntheticConfig.probability_prioritize_objects_of_interest)
    create_synthetic_images_for_all_direct_subfolders(syntheticConfig, startup.occlusion_name_occlusion_id_dict)

    logger.info('Finished')

if __name__ == '__main__':
    main()
