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

from PIL import Image, ImageOps
from pathlib import Path

def randomly_sample_new_image_size(image_size):
    (width, height) = image_size

    low = 0.80 # TODO: Move to config
    high = 2.8 # TODO: Move to config
    scale = random.uniform(low, high) # [low, high]
    
    scaled_width = int(scale * width)
    scaled_height = int(scale * height)

    return (scaled_width, scaled_height)

def resize_image_randomly(pil_image_original):
    logger = logging.getLogger(__name__)

    new_size = randomly_sample_new_image_size(pil_image_original.size)
    pil_image_resized = ImageOps.fit(pil_image_original, new_size, Image.ANTIALIAS)

    head, tail = ntpath.split(pil_image_original.filename)
    original_filename_with_extension = tail or ntpath.basename(head)
    original_filename, file_extension = os.path.splitext(original_filename_with_extension)

    (new_width, new_height) = new_size
    pil_image_resized.filename = '%s_width_%d_height_%d%s' % (original_filename, new_width, new_height, file_extension)
    logger.debug('From image %s created new image in memory with name %s', original_filename_with_extension, pil_image_resized.filename) 

    return pil_image_resized

def randomly_sample_point_within_image(pil_image_bg, pil_image_fg):
    (backgroundWidth, backgroundHeight) = pil_image_bg.size
    (foregroundWidth, foregroundHeight) = pil_image_fg.size

    if foregroundWidth >= backgroundWidth:
        logger.error('%s is greater or equal to %s specifically %s >= %s', 'foregroundWidth', 'backgroundWidth', foregroundWidth, backgroundWidth)

    if foregroundHeight >= backgroundHeight:
        logger.error('%s is greater or equal to %s specifically %s >= %s', 'foregroundHeight', 'backgroundHeight', foregroundHeight, backgroundHeight)

    xTopLeft = 0
    yTopLeft = 0

    xBottomRight = backgroundWidth - foregroundWidth
    yBottomRight = backgroundHeight - foregroundHeight

    return randomly_sample_point_within_rectangle((xTopLeft, yTopLeft), (xBottomRight, yBottomRight))

def get_top_left_bottom_right_coordinates(background_annotations, index):
    # TODO:
    return ((xTopLeft, yTopLeft), (xBottomRight, yBottomRight))

def randomly_sample_point_within_image_or_object_of_interest(background_image, occlusion_image, background_annotations, p):
    logger = logging.getLogger(__name__)
    # We decide between image and object of interest

    (numRows, numCols) = background_annotations.shape 
    useImage = True
    if numRows == 0:
        logger.debug('No annotations found for %s', path)
    else:
        r = math.random.uniform()
        useImage = True if r > p else False

    if useImage:
        return randomly_sample_point_within_image(background_image, occlusion_image)
    else:
        index = randomly_choose_object_of_interest(background_annotations)
        points = get_top_left_bottom_right_coordinates(background_annotations, index)
        return randomly_sample_point_within_rectangle(points)

def randomly_choose_object_of_interest(background_annotations):
    # TODO:

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

def get_bounding_box(synthetic_grayscale_mask):
    # Gives a tuple of two arrays where the first array represents
    # the row or the y axis and the second array represents the column for y
    # axis.
    where = np.where(synthetic_grayscale_mask != 0)

    y0 = np.min(where[0]) # row
    x0 = np.min(where[1]) # col

    y1 = np.max(where[0]) # row
    x1 = np.max(where[1]) # col

    width = x1 - x0
    height = y1 - y0

    # We can run into serialization issues with numpy.int64
    x0 = x0.item() # numpy to native value
    y0 = y0.item() # numpy to native value
    width = width.item() # numpy to native value
    height = height.item() # numpy to native value
 
    return (x0, y0, width, height)

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

def create_synthetic_image(background_image, occlusion_image, threshold):
    logger = logging.getLogger(__name__)

    # TODO: call the high level method which chooses between whole image versus object of interest
    point = randomly_sample_point_within_image(background_image, occlusion_image)
    (x0, y0) = point
    logger.debug('Randomly sampled point within background image is (%s,%s)', x0, y0)

    synthetic_image, original_occlusion_image_grayscale_image_mask = create_synthetic_image_point(background_image, occlusion_image, point, threshold)
    return synthetic_image, point, original_occlusion_image_grayscale_image_mask 

def compute_target_path(target_dir, path_background_image, cur_image_id):
    # Compute target path
    (file_name, file_ext) = get_filename_and_extension(path_background_image)
    file_name = '%s.%s%s' % (file_name, cur_image_id, file_ext) 
    target_path = os.path.join(target_dir, file_name)
    return (file_name, target_path)

def compute_annotation(pil_background_image, pil_synthetic_image, pil_occlusion_image, top_left_point, original_occlusion_image_grayscale_image_mask):
    (backgroundImageWidth, backgroundImageHeight) = pil_background_image.size
    (syntheticImageWidth, syntheticImageHeight) = pil_synthetic_image.size
    (occlusionImageTopLeftx, occlusionImageTopLefty) = top_left_point
    (occlusionImageWidth, occlusionImageHeight) = pil_occlusion_image.size

    if syntheticImageWidth != backgroundImageWidth or syntheticImageHeight != backgroundImageHeight:
        logger.error('synthetic image and background image sizes mismatch')

    (topLeftPointMaskWithinForegroundImagex, topLeftPointMaskWithinForegroundImagey, maskWidthWithinForegroundImage, maskHeightWithinForegroundImage) = get_bounding_box(np.array(original_occlusion_image_grayscale_image_mask))

    xCenter = occlusionImageTopLeftx + topLeftPointMaskWithinForegroundImagex + (maskWidthWithinForegroundImage/2.0)
    yCenter = occlusionImageTopLefty + topLeftPointMaskWithinForegroundImagey + (maskHeightWithinForegroundImage/2.0)
    
    xCenterWrtBackgroundImage = xCenter/backgroundImageWidth
    yCenterWrtBackgroundImage = yCenter/backgroundImageHeight

    occlusionWidthWrtBackgroundImage = maskWidthWithinForegroundImage*1.0/backgroundImageWidth
    occlusionHeightWrtBackgroundImage = maskHeightWithinForegroundImage*1.0/backgroundImageHeight

    return (xCenterWrtBackgroundImage, yCenterWrtBackgroundImage, occlusionWidthWrtBackgroundImage, occlusionHeightWrtBackgroundImage)

def save_image_annotation(path_background_image, image_annotation, target_annotations_dir, cur_image_id, occlusion_name_occlusion_id_dict, occlusion_name):
    logger = logging.getLogger(__name__)

    background_file_name = get_filename_without_ext(path_background_image)
    original_path = Path(path_background_image)

    target_annotations_file_name = '%s.annotated.%s.%s' % (background_file_name, cur_image_id, 'txt') 
    target_path_annotations_file = os.path.join(target_annotations_dir, target_annotations_file_name)

    if not os.path.isdir(target_annotations_dir):
        os.mkdir(target_annotations_dir)
        logger.info('Created target annotations directory %s', target_annotations_dir)

    occlusion_id = '8' # Don't care
    if (occlusion_name.upper() not in occlusion_name_occlusion_id_dict):
        logger.error('Occlusion name %s not found in dictionary %s', occlusion_name, 'occlusion_name_occlusion_id_dict')
    else:
        occlusion_id = occlusion_name_occlusion_id_dict[occlusion_name.upper()]

    with open(target_path_annotations_file, 'w') as fileHandle:
        (xCenterWrtBackgroundImage, yCenterWrtBackgroundImage, occlusionWidthWrtBackgroundImage, occlusionHeightWrtBackgroundImage) = image_annotation
        fileHandle.write('%s %s %s %s %s' % (occlusion_id, xCenterWrtBackgroundImage, yCenterWrtBackgroundImage, occlusionWidthWrtBackgroundImage, occlusionHeightWrtBackgroundImage))
        fileHandle.write('\n')

    logger.debug('Wrote the occlusion annotation to %s', target_path_annotations_file) 

def process_resized_occlusion_image(occlusion_image, occlusion_name, target_dir, background_image, path_background_image, threshold, cur_image_id, target_annotations_dir, occlusion_name_occlusion_id_dict): 
    logger = logging.getLogger(__name__)

    (file_name, target_path) = compute_target_path(target_dir, path_background_image, cur_image_id)

    # synthetic image generation and save
    # TODO: Read the annotations file 
    # TODO: pass the background annotations file
    synthetic_image, point, original_occlusion_image_grayscale_image_mask = create_synthetic_image(background_image, occlusion_image, threshold)
    save_synthetic_image(background_image, synthetic_image, occlusion_image, point, threshold, target_path)

    # image info
    image_info = get_image_info(synthetic_image, file_name, cur_image_id)

    # Generate annotation and save
    image_annotation = compute_annotation(background_image, synthetic_image, occlusion_image, point, original_occlusion_image_grayscale_image_mask)
    # TODO: Add support for taking a mapping dictionary which takes as input the name of the occlusion and has a mapping to the integer label for the same.
    save_image_annotation(path_background_image, image_annotation, target_annotations_dir, cur_image_id, occlusion_name_occlusion_id_dict, occlusion_name)

    logger.debug('image_info, annotation created for %s', image_info['file_name'])

    cur_image_id += 1

    return (image_annotation, image_info, cur_image_id)

def get_image_info(synthetic_image, file_name, image_id):
    (width, height) = synthetic_image.size

    dict = {}
    dict['file_name'] = file_name 
    dict['license'] = 1 # hardcoded
    dict['width'] = width 
    dict['height'] = height 
    dict['id'] = image_id 

    return dict

def get_filename_without_ext(path):
    filename_with_ext = os.path.basename(path)
    list = os.path.splitext(filename_with_ext)
    return list[0]

def get_immediate_subdirectories(super_dir):
    return [name for name in os.listdir(super_dir) if os.path.isdir(os.path.join(super_dir, name))]

def get_immediate_subdirectory_paths(super_dir):
    subdirs = get_immediate_subdirectories(super_dir)
    return [os.path.join(super_dir, subdir) for subdir in subdirs]

def get_immediate_parent_folder(abs_path):
    folders = get_all_folders_in_path(abs_path)
    return folders[0] # assuming path is not at root

def get_filename_without_ext(path):
    filename_with_ext = os.path.basename(path)
    list = os.path.splitext(filename_with_ext)
    return list[0]

def get_filename_and_extension(path): 
    path = Path(path)
    return (path.stem, path.suffix)

def get_all_folders_in_path(path): 
    drive, path_and_file = os.path.splitdrive(path)
    path, file = os.path.split(path_and_file)

    folders = []
    while True:
        path, folder = os.path.split(path)

        if folder != "":
            folders.append(folder)
        else:
            if path != "":
                folders.append(path)
            break
    return folders

def process_original_occlusion_image(path_occlusion_image, path_background_image, threshold, target_dir_path_images, target_dir_path_annotations, cur_image_id, occlusion_name_occlusion_id_dict): 
    logger = logging.getLogger(__name__)

    occlusion_image_filename = get_filename_without_ext(path_occlusion_image)
    occlusion_image = Image.open(path_occlusion_image)
    occlusion_name = get_immediate_parent_folder(path_occlusion_image) 

    # TODO: Read the annotations file corresponding to the background image; error if not found
    background_image = Image.open(path_background_image)

    image_info_collection = []
    image_annotation_collection = []

    num_runs_per_original_image = 1 # TODO: Move to config

    occlusion_name = get_immediate_parent_folder(path_occlusion_image)

    for i in range(num_runs_per_original_image): 
        logger.debug('For occlusion %s, image %s, run %d of %d', occlusion_name, occlusion_image_filename, i+1, num_runs_per_original_image) 
        occlusion_image_resized = resize_image_randomly(occlusion_image) # specify the gaussian and standard deviation
        # TODO: pass corresponding annotations file
        (image_annotation, image_info, cur_image_id) = process_resized_occlusion_image(occlusion_image_resized, occlusion_name, target_dir_path_images, background_image, path_background_image, threshold, cur_image_id, target_dir_path_annotations, occlusion_name_occlusion_id_dict)
        image_info_collection.append(image_info)
        image_annotation_collection.append(image_annotation)

    return (image_annotation_collection, image_info_collection, cur_image_id)

def create_synthetic_images_for_all_images_under_current_folders(path_background_dir, path_foreground_dir, threshold, target_dir_path_images, target_dir_path_annotations, cur_image_id, occlusion_name_occlusion_id_dict):
    logger = logging.getLogger(__name__)

    if not os.path.isdir(target_dir_path_images):
        os.mkdir(target_dir_path_images)
        logger.info('Created target directory %s', target_dir_path_images)

    if not os.path.isdir(target_dir_path_annotations):
        os.mkdir(target_dir_path_annotations)
        logger.info('Created target directory %s', target_dir_path_annotations)

    logger.info('create_synthetic_images - background: %s, foreground: %s, image output: %s, annotations output: %s', path_background_dir, path_foreground_dir, target_dir_path_images, target_dir_path_annotations)

    # TODO: Move to config
    foregound_valid_extensions = ["jpg", "jpeg", "JPEG", "JPG"] # image masking code doesn't work well with PNGs

    foreground_image_paths = []
    for valid_extension in foregound_valid_extensions:
        foreground_search_path = path_foreground_dir + "/" + "*." + valid_extension
        for file_path in glob.glob(foreground_search_path):
            foreground_image_paths.append(file_path)

    # TODO: Move to config
    background_valid_extensions = ["jpg", "jpeg", "JPEG", "JPG", "png", "PNG"]
    background_image_paths = []
    for valid_extension in background_valid_extensions:
        background_search_path = path_background_dir + "/" + "*." + valid_extension
        for file_path in glob.glob(background_search_path):
            background_image_paths.append(file_path)

    image_info_collection = []
    image_annotation_collection = []

    for foreground_image_path in foreground_image_paths:
        for background_image_path in background_image_paths:
            #TODO: pass annotations background directory
            (image_annotation_collection, image_info_collection, cur_image_id) = process_original_occlusion_image(foreground_image_path, background_image_path, threshold, target_dir_path_images, target_dir_path_annotations, cur_image_id, occlusion_name_occlusion_id_dict) 

def main():
    startup = Startup.Startup()
    startup.configure()

    logger = logging.getLogger(__name__)
    logger.info('Started')

    occlusion_name = "PlasticBag"
    syntheticConfig = startup.synthetic_config

    #process_original_occlusion_image(path_foreground_file, path_background_file, threshold, target_dir, cur_image_id, occlusion_name) 
    # TODO: pass background annotations directory
    create_synthetic_images_for_all_images_under_current_folders(syntheticConfig.path_background_dir, syntheticConfig.path_foreground_dir, syntheticConfig.threshold, syntheticConfig.target_dir_path_images, syntheticConfig.target_dir_path_annotations, syntheticConfig.cur_image_id, startup.occlusion_name_occlusion_id_dict)

    logger.info('Finished')

if __name__ == '__main__':
    main()
