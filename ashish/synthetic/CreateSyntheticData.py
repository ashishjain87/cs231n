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
import yaml
import SyntheticConfig as synthetic_config
import Startup

from Rotator import Rotator
from NoOpRotator import NoOpRotator
from ImagePath import *
from Annotation import *

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

def randomly_sample_point_within_image(pil_image_bg, pil_image_fg):
    logger = logging.getLogger(__name__)

    (backgroundWidth, backgroundHeight) = pil_image_bg.size
    (foregroundWidth, foregroundHeight) = pil_image_fg.size

    if foregroundWidth >= backgroundWidth:
        logger.error('%s is greater or equal to %s specifically %s >= %s', 'foregroundWidth', 'backgroundWidth', foregroundWidth, backgroundWidth)

    if foregroundHeight >= backgroundHeight:
        logger.error('%s is greater or equal to %s specifically %s >= %s', 'foregroundHeight', 'backgroundHeight', foregroundHeight, backgroundHeight)

    (minWidth, minHeight, maxWidth, maxHeight) = get_largest_bounding_box_assuming_using_center_point_to_affix_foreground_image(pil_image_bg, pil_image_fg)

    return randomly_sample_point_within_rectangle_assuming_center_point_to_affix_foreground_image((minWidth, minHeight), (maxWidth, maxHeight), pil_image_bg, pil_image_fg)

def yolo_to_kitti(annotation, pil_background_image):
    x, y, width, height = annotation[1], annotation[2], annotation[3], annotation[4]
    (image_width, image_height) = pil_background_image.size 
    new_width = round(width * image_width)
    new_height = round(height * image_height)
    left = round(x * image_width) - new_width//2
    top = round(y * image_height) - new_height//2
    return (left, top, new_width, new_height)

def get_top_left_bottom_right_coordinates(background_annotations, index, pil_background_image):
    annotation = background_annotations[index]
    (xTopLeft, yTopLeft, width, height) = yolo_to_kitti(annotation, pil_background_image)
    xBottomRight = xTopLeft + width
    yBottomRight = yTopLeft + height
    return ((xTopLeft, yTopLeft), (xBottomRight, yBottomRight))

def randomly_sample_point_within_image_or_object_of_interest(background_image, occlusion_image, background_annotations, p):
    logger = logging.getLogger(__name__)

    (numRows, numCols) = background_annotations.shape 
    useImage = True
    if numRows == 0: # TODO: TEST if no annotations are there for an image
        logger.debug('No annotations found for %s', path)
    else:
        r = random.random()
        useImage = True if r > p else False
        logger.debug('r was %f, p was %s, useImage %s', r, p, useImage)

    if useImage:
        logger.debug('Randomly sampling from whole image')
        return randomly_sample_point_within_image(background_image, occlusion_image)
    else:
        index = randomly_choose_object_of_interest(numRows)
        logger.debug('Randomly sampling from object of interest. Index chosen %i', index)
        (topLeft, bottomRight) = get_top_left_bottom_right_coordinates(background_annotations, index, background_image)

        (xTopLeft, yTopLeft) = topLeft
        (xBottomRight, yBottomRight) = bottomRight
        logger.debug('Rectangle chosen topLeft: (%i, %i), bottomRight: (%i, %i)', xTopLeft, yTopLeft, xBottomRight, yBottomRight)

        return randomly_sample_point_within_rectangle_assuming_center_point_to_affix_foreground_image(topLeft, bottomRight, background_image, occlusion_image)

def randomly_choose_object_of_interest(num_annotations) -> int: # returns index
    return random.randint(0,num_annotations-1)

def get_largest_bounding_box_assuming_using_center_point_to_affix_foreground_image(pil_image_bg, pil_image_fg):
    logger = logging.getLogger(__name__)

    (backgroundWidth, backgroundHeight) = pil_image_bg.size
    (foregroundWidth, foregroundHeight) = pil_image_fg.size

    width = round(foregroundWidth/2.0)
    height = round(foregroundHeight/2.0)

    logger.debug('background image size: (%s, %s)', backgroundWidth, backgroundHeight)
    logger.debug('foreground image size: (%s, %s), half-size-rounded: (%s, %s)', foregroundWidth, foregroundHeight, width, height)

    minWidth = width 
    minHeight = height
    maxWidth = backgroundWidth - width 
    maxHeight = backgroundHeight - height

    return (minWidth, minHeight, maxWidth, maxHeight) 

def correct_bounding_box_coordinates_assuming_center_point_to_affix_foreground_image(backgroundImageTopLeftCoordinates, backgroundImageBottomRightCoordinates, pil_image_bg, pil_image_fg):
    logger = logging.getLogger(__name__)

    (xTopLeft, yTopLeft) = backgroundImageTopLeftCoordinates
    (xBottomRight, yBottomRight) = backgroundImageBottomRightCoordinates

    (minWidth, minHeight, maxWidth, maxHeight) = get_largest_bounding_box_assuming_using_center_point_to_affix_foreground_image(pil_image_bg, pil_image_fg)

    logger.debug('Largest bounding box coordinates: (%s, %s), (%s, %s)', minWidth, minHeight, maxWidth, maxHeight) 
    logger.debug('Original coordinates: (%s, %s), (%s, %s)', xTopLeft, yTopLeft, xBottomRight, yBottomRight) 

    xTopLeft = max(xTopLeft, minWidth)
    yTopLeft = max(yTopLeft, minHeight)
    
    xBottomRight = min(xBottomRight, maxWidth)
    yBottomRight = min(yBottomRight, maxHeight)

    logger.debug('Revised coordinates: (%s, %s), (%s, %s)', xTopLeft, yTopLeft, xBottomRight, yBottomRight) 

    return ((xTopLeft, yTopLeft), (xBottomRight, yBottomRight))

def randomly_sample_point_within_rectangle_assuming_center_point_to_affix_foreground_image(backgroundImageTopLeftCoordinates, backgroundImageBottomRightCoordinates, pil_image_bg, pil_image_fg):
    backgroundImageTopLeftCoordinates, backgroundImageBottomRightCoordinates = correct_bounding_box_coordinates_assuming_center_point_to_affix_foreground_image(backgroundImageTopLeftCoordinates, backgroundImageBottomRightCoordinates, pil_image_bg, pil_image_fg)
    return randomly_sample_point_within_rectangle(backgroundImageTopLeftCoordinates, backgroundImageBottomRightCoordinates)

def randomly_sample_point_within_rectangle(backgroundImageTopLeftCoordinates, backgroundImageBottomRightCoordinates):
    logger = logging.getLogger(__name__)

    (xTopLeft, yTopLeft) = backgroundImageTopLeftCoordinates
    (xBottomRight, yBottomRight) = backgroundImageBottomRightCoordinates

    if xTopLeft >= xBottomRight:
        logger.error('%s is greater than %s specifically %s >= %s', 'xTopLeft', 'xBottomRight', xTopLeft, xBottomRight) 

    if yTopLeft >= yBottomRight:
        logger.error('%s is greater than %s specifically %s >= %s', 'yTopLeft', 'yBottomRight', yTopLeft, yBottomRight) 
    
    x = round(random.uniform(xTopLeft, xBottomRight))
    y = round(random.uniform(yTopLeft, yBottomRight))

    return (x,y)

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

def create_synthetic_image_grayscale_mask(background_pil_image, occlusion_pil_image, occlusion_grayscale_mask, point):
    (background_pil_image_red, background_pil_image_green, background_pil_image_blue) = background_pil_image.split()

    synthetic_grayscale_mask = np.zeros_like(background_pil_image_red, np.uint8) 

    synthetic_grayscale_mask_pil_image = Image.fromarray(synthetic_grayscale_mask)
    occlusion_grayscale_mask_pil_image = Image.fromarray(occlusion_grayscale_mask)

    box = point 
    synthetic_grayscale_mask_pil_image.paste(occlusion_grayscale_mask_pil_image, box) 

    # Note, the numpy array conversion of PIL image is transposed i.e. for an
    # image of size (1920, 1440), you will get back an array of shape (1440,
    # 1920).  If you want to convert from a np array to an image then that's
    # the shape you will need to provide the numpy array in i.e. (1440, 1920)
    # to get back an image of size (1920, 1440). However, if you would like to
    # use other image libraries then you might want to be careful. Note,
    # skimage and opencv save the image in this format.
    #
    # Rationale from scikit website:
    # Link: 
    #
    # Because scikit-image represents images using NumPy arrays, the coordinate
    # conventions must match. Two-dimensional (2D) grayscale images (such as
    # camera above) are indexed by rows and columns (abbreviated to either
    # (row, col) or (r, c)), with the lowest element (0, 0) at the top-left
    # corner. In various parts of the library, you will also see rr and cc
    # refer to lists of row and column coordinates. We distinguish this
    # convention from (x, y), which commonly denote standard Cartesian
    # coordinates, where x is the horizontal coordinate, y - the vertical one,
    # and the origin is at the bottom left (Matplotlib axes, for example, use
    # this convention).
    synthetic_grayscale_mask = np.array(synthetic_grayscale_mask_pil_image)
    return synthetic_grayscale_mask

def create_synthetic_image_point(background_image, occlusion_image, point, threshold):
    logger = logging.getLogger(__name__)

    (x0, y0) = point
    grayscale_image_mask = occlusions.create_grayscale_image_mask(occlusion_image, threshold) 

    (width, height) = occlusion_image.size
    box = (x0, y0, x0 + width, y0 + height) 

    synthetic_image = background_image.copy()
    synthetic_image.paste(occlusion_image, box, grayscale_image_mask)

    return synthetic_image, grayscale_image_mask

def save_synthetic_image(background_image, synthetic_image, occlusion_image, point, threshold, target_path):
    logger = logging.getLogger(__name__)
    synthetic_image.save(target_path)

    (x0, y0) = point

    logger.debug(
        '%s pasted at (%s, %s) and saved in file %s',
        occlusion_image.filename,
        x0,
        y0,
        target_path)

def get_top_left_point(centerPoint, pil_occlusion_image):
    (width, height) = pil_occlusion_image.size
    (xCenter, yCenter) = centerPoint

    xTopLeft = xCenter - width//2 # TODO: Think about this if width is odd
    yTopLeft = yCenter - height//2 # TODO: Think about this if height is odd

    return (xTopLeft, yTopLeft)

def create_synthetic_image(background_image, background_annotation, occlusion_image, threshold, probability_prioritize_objects_of_interest):
    logger = logging.getLogger(__name__)

    # TODO: call the high level method which chooses between whole image versus object of interest
    #point = randomly_sample_point_within_image(background_image, occlusion_image) # TODO: REMOVE
    centerPoint = randomly_sample_point_within_image_or_object_of_interest(background_image, occlusion_image, background_annotation, probability_prioritize_objects_of_interest)
    (x0, y0) = centerPoint
    logger.debug('Randomly sampled point within background image is (%s,%s) which is the center point', x0, y0)

    topLeftPoint = get_top_left_point(centerPoint, occlusion_image)
    (x0, y0) = topLeftPoint 
    logger.debug('Randomly sampled point within background image is (%s,%s) which is the top left point', x0, y0)
    
    synthetic_image, original_occlusion_image_grayscale_image_mask = create_synthetic_image_point(background_image, occlusion_image, topLeftPoint, threshold)
    return synthetic_image, topLeftPoint, original_occlusion_image_grayscale_image_mask 

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
    probability_prioritize_objects_of_interest,
):
    logger = logging.getLogger(__name__)

    (file_name, target_path) = compute_target_path(target_dir, path_background_image, cur_image_id)

    # synthetic image generation and save
    (synthetic_image, point, original_occlusion_image_grayscale_image_mask) = create_synthetic_image(
        background_image,
        background_annotation,
        occlusion_image,
        threshold,
        probability_prioritize_objects_of_interest,
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

    for i in range(num_runs_per_original_image):
        logger.debug("For occlusion %s, image %s, run %d of %d", occlusion_name, occlusion_image_filename, i + 1, num_runs_per_original_image)
        occlusion_image_resized = resize_image_randomly(occlusion_image, background_image)  # specify the gaussian and standard deviation
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
            probability_prioritize_objects_of_interest,
        )
        image_info_collection.append(image_info)
        image_annotation_collection.append(image_annotation)

    return (image_annotation_collection, image_info_collection, cur_image_id)

def create_synthetic_images_for_all_images_under_current_folders(
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

    for foreground_image_path in foreground_image_paths:
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
    for subdir in subdirs:
        logger.info("Processing %s", subdir)
        path_foreground_dir = os.path.join(syntheticConfig.path_foreground_super_dir, subdir)
        target_dir_path_images = os.path.join(syntheticConfig.target_dir, subdir, syntheticConfig.images_dir_name)
        target_dir_path_annotations = os.path.join(syntheticConfig.target_dir, subdir, syntheticConfig.annotations_dir_name)

        logger.debug('path_foreground_dir: %s', path_foreground_dir)
        logger.debug('target_dir_path_images: %s', target_dir_path_images)
        logger.debug('target_dir_path_annotations: %s', target_dir_path_annotations)

        (image_annotations, image_infos, cur_image_id) = create_synthetic_images_for_all_images_under_current_folders( \
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
