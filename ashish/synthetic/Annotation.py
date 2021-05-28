import logging
import logging.config
import numpy as np
import os

from pathlib import Path
from ImagePath import *

# TODO: Move to common python file
def clean_yolo(yolo_label):
    if len(yolo_label.shape) == 1:  # if there's only one label, shape will be (5,) which is 1-d, this fixes it.
        yolo_label = np.array([yolo_label])
    return yolo_label

def read_background_annotation(yolo_label_path: str) -> np.ndarray:
    yolo_label = np.genfromtxt(yolo_label_path, delimiter=" ", dtype=float, encoding=None)
    return clean_yolo(yolo_label) 

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

def compute_annotation(pil_background_image, pil_synthetic_image, pil_occlusion_image, top_left_point, original_occlusion_image_grayscale_image_mask):
    logger = logging.getLogger(__name__)

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

