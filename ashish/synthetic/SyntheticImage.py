import logging
import logging.config
import numpy as np
import Occlusions as occlusions
from Affixer import Affixer
from OriginalAffixer import OriginalAffixer
from PIL import Image, ImageOps

def create_synthetic_image_point(background_image, occlusion_image, point, threshold):
    logger = logging.getLogger(__name__)

    (x0, y0) = point
    grayscale_image_mask = occlusions.create_grayscale_image_mask(occlusion_image, threshold) 

    (width, height) = occlusion_image.size
    box = (x0, y0, x0 + width, y0 + height) 

    synthetic_image = background_image.copy()
    synthetic_image.paste(occlusion_image, box, grayscale_image_mask)

    return synthetic_image, grayscale_image_mask

def get_top_left_point(centerPoint, pil_occlusion_image):
    (width, height) = pil_occlusion_image.size
    (xCenter, yCenter) = centerPoint

    xTopLeft = xCenter - width//2 # TODO: Think about this if width is odd
    yTopLeft = yCenter - height//2 # TODO: Think about this if height is odd

    return (xTopLeft, yTopLeft)

def create_synthetic_image(background_image, occlusion_image, threshold, center_point):
    logger = logging.getLogger(__name__)

    (x0, y0) = center_point
    logger.debug('Sampled point within background image is (%s,%s) which is the center point', x0, y0)

    topLeftPoint = get_top_left_point(center_point, occlusion_image)
    (x0, y0) = topLeftPoint 
    logger.debug('Sampled point within background image is (%s,%s) which is the top left point', x0, y0)
    
    synthetic_image, original_occlusion_image_grayscale_image_mask = create_synthetic_image_point(background_image, occlusion_image, topLeftPoint, threshold)
    return synthetic_image, topLeftPoint, original_occlusion_image_grayscale_image_mask 

def create_synthetic_image_randomly(background_image, background_annotation, occlusion_image, threshold, probability_prioritize_objects_of_interest):
    logger = logging.getLogger(__name__)

    # TODO: call the high level method which chooses between whole image versus object of interest
    #point = randomly_sample_point_within_image(background_image, occlusion_image) # TODO: REMOVE
    affixer: Affixer = OriginalAffixer(probability_prioritize_objects_of_interest)
    centerPoint, _ = affixer.decide_where_and_scale(background_image, background_annotation, occlusion_image)
    (x0, y0) = centerPoint
    logger.debug('Randomly sampled point within background image is (%s,%s) which is the center point', x0, y0)

    topLeftPoint = get_top_left_point(centerPoint, occlusion_image)
    (x0, y0) = topLeftPoint 
    logger.debug('Randomly sampled point within background image is (%s,%s) which is the top left point', x0, y0)
    
    synthetic_image, original_occlusion_image_grayscale_image_mask = create_synthetic_image_point(background_image, occlusion_image, topLeftPoint, threshold)
    return synthetic_image, topLeftPoint, original_occlusion_image_grayscale_image_mask 

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
