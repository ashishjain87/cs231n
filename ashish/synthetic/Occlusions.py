import logging
import logging.config

import numpy as np
import skimage
import cv2

from PIL import Image
from skimage.morphology import closing, square
from skimage.segmentation import clear_border
from scipy import ndimage as ndi

def create_grayscale_image_mask(image, threshold):
    logger = logging.getLogger(__name__)
    if (is_transparent(image)):
        logger.debug('Invoking transparent mask generation')
        return create_grayscale_image_mask_transparent(image)
    else:
        logger.debug('Invoking regular mask generation')
        return create_grayscale_image_mask_regular(image, threshold)

def create_grayscale_image_mask_empty(pil_image):
    return np.zeros(pil_image.size, dtype='uint8')

def create_grayscale_image_mask_transparent(pil_image):
    logger = logging.getLogger(__name__)

    # Don't use the regular approach to convert pil image to open cv image because
    # we lose the alpha channel in the process. We are not interested in R,G,B here.
    # Therefore, their relative ordering doesn't matter anyway.
    #
    # Optionally, you can also use the following but it is not necessary. 
    # opencv_image = np.array(pil_image)
    # opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_RGBA2BGRA)
    image = np.array(pil_image)
    (r,g,b,a) = cv2.split(image)

    if a.dtype != 'uint8': 
        logger.error('alpha channel dtype should be uint8 but is %s', a.dtype)
        return create_grayscale_image_mask_empty(pil_image)

    if np.max(a) != 255:
        logger.warn('Maximum value should have been equal to 255. Value is %i', np.max(a))

    image_mask = Image.fromarray(a) 
    return image_mask

def create_grayscale_image_mask_regular(image, threshold):
    """
    There are several approaches to creating a mask based on thresholding.
    1) Convert to '1' or binary image. However, in practice, this does not work well.
    2) Convert to 'L' or gray scale image and then apply thresholding.
    3) Apply thresholding across all three channels and then combine the results.

    Approach 2 can give false positives as the grayscale value might have a high pixel value.
    Therefore, convering on approach 3.
    """
    logger = logging.getLogger(__name__)

    if image.mode == "RBGA":
        logger.error("Regular mask creator can only handle images with three channels but received image with mode RGBA")
        return create_grayscale_image_mask_empty(pil_image)

    # Pillow (PIL) accepts a mask in the image format. Mode 'L' or grayscale works best. '1' or binary mode has a few bugs
    # as per documentation online last updated in 2018.
    #
    # Convert binary mask to grayscale image
    binary_mask = create_binary_mask_threshold(image, threshold)
    #binary_mask = create_binary_mask_grabcut_with_mask_intense(image)
    #binary_mask = create_binary_mask_grabcut(image) # This works well maybe.
    #binary_mask = create_binary_mask_grabcut_with_mask(image)
    image_mask = convert_binary_mask_to_grayscale_image(binary_mask)

    return image_mask

def is_transparent(image):
    if image.mode == "RGB":
        return False 
    elif image.mode == "RGBA":
        extrema = image.getextrema()
        if extrema[3][0] < 255:
            return True
    else:
        return False

def convert_binary_mask_to_grayscale_image(binary_mask):
    mask = binary_mask.astype('uint8') # view('uint8') might be more efficient as it does not make an additional copy
    mask = mask * 255
    image_mask = Image.fromarray(mask)
    return image_mask

def convert_binary_mask_to_grayscale_mask(binary_mask):
    mask = binary_mask.astype('uint8') # view('uint8') might be more efficient as it does not make an additional copy
    return mask

def convert_grayscale_mask_to_grayscale_image(grayscale_mask):
    mask = grayscale_mask
    mask = mask * 255
    image_mask = Image.fromarray(mask)
    return image_mask

def convert_binary_mask_to_rgb_image_red(binary_mask):
    mask = binary_mask.astype('uint8') # view('uint8') might be more efficient as it does not make an additional copy
    mask = mask * 255

    red = mask
    green = np.zeros_like(mask, dtype='uint8')
    blue = np.zeros_like(mask, dtype='uint8')

    image_np = np.zeros((red.shape[0], red.shape[1], 3), dtype='uint8')
    image_np[:,:,0] = red
    image_np[:,:,1] = green
    image_np[:,:,2] = blue

    image = Image.fromarray(image_np)

    return image

def convert_binary_mask_to_rgba_mask(binary_mask, alpha = 0.5):
    mask = binary_mask.astype('uint8') # view('uint8') might be more efficient as it does not make an additional copy
    scale = 255 * alpha
    mask = mask * scale
    mask = mask.astype(np.uint8)

    red = np.zeros_like(mask, dtype='uint8')
    green = np.zeros_like(mask, dtype='uint8')
    blue = np.zeros_like(mask, dtype='uint8')
    alpha = mask 

    image_np = np.zeros((red.shape[0], red.shape[1], 4), dtype='uint8')
    image_np[:,:,0] = red
    image_np[:,:,1] = green 
    image_np[:,:,2] = blue 
    image_np[:,:,3] = alpha 

    image = Image.fromarray(image_np)
    return image

def overlay_mask_on_image(image, create_binary_mask):
    binary_mask = create_binary_mask(image) 
    mask_image = convert_binary_mask_to_rgb_image_red(binary_mask)

def create_binary_mask_background_sure(image, threshold):
    binary_mask_occlusion = create_binary_mask_threshold(image, 254)
    binary_mask_background = np.invert(binary_mask_occlusion)
    return binary_mask_background

def create_binary_mask_threshold(image, threshold):
    image_red, image_green, image_blue = image.split()

    # Convert to numpy arrays to take advantage of vectorized operations
    binary_mask_red = create_binary_mask_grayscale(image_red, threshold)
    binary_mask_green = create_binary_mask_grayscale(image_green, threshold)
    binary_mask_blue = create_binary_mask_grayscale(image_blue, threshold)

    binary_mask = np.logical_and(np.logical_and(binary_mask_red, binary_mask_green), binary_mask_blue)
    binary_mask = np.invert(binary_mask)

    # fill any holes
    binary_mask = ndi.binary_fill_holes(binary_mask)

    return binary_mask

def convert_pil_image_to_opencv_image(pil_image):
    opencv_image = np.array(pil_image)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2BGR)
    return opencv_image

def create_binary_mask_grabcut_with_mask_intense(pil_image):
    image = convert_pil_image_to_opencv_image(pil_image)
    (width, height, numChannels) = image.shape

    # initialize parameters for grabcut
    # initial foreground region
    foreground_bounding_box = (0 , 0, width-1, height-1)
    bgdModel = np.zeros((1,65), np.float64)
    fgdModel = np.zeros((1,65), np.float64)

    # Compute several binary masks to determine where there is a background, foreground and probable foreground
    # background sure
    binary_mask_background_sure = create_binary_mask_background_sure(pil_image)

    # foreground probable
    # TODO: Determine soft threshold using otsu
    soft_threshold = 155 # otsu threshold
    binary_mask_occlusion_probable = create_binary_mask_threshold(pil_image, soft_threshold)

    # foreground sure
    hard_threshold = 100 # visual inspection
    binary_mask_occlusion_sure = create_binary_mask_threshold(pil_image, hard_threshold)

    # initialize grayscale_mask
    grayscale_mask = np.zeros((width, height), np.uint8)
    # apply background sure
    grayscale_mask = np.where(binary_mask_background_sure==True, cv2.GC_BGD, cv2.GC_PR_BGD)
    # apply foreground probable
    grayscale_mask = np.where(binary_mask_occlusion_probable==True, cv2.GC_PR_FGD, grayscale_mask)
    # apply foreground sure
    grayscale_mask = np.where(binary_mask_occlusion_sure==True, cv2.GC_FGD, grayscale_mask)
    # convert to 'uint8'
    grayscale_mask = grayscale_mask.astype(np.uint8)

    numIterations = 5

    cv2.grabCut(image, grayscale_mask, foreground_bounding_box, bgdModel, fgdModel, numIterations, cv2.GC_INIT_WITH_MASK)

    # convert to binary mask
    binary_mask = np.where((grayscale_mask==2) | (grayscale_mask==0), False, True)
    return binary_mask

def create_binary_mask_grabcut_with_mask(pil_image):
    image = convert_pil_image_to_opencv_image(pil_image)
    (width, height, numChannels) = image.shape

    # initialize parameters for grabcut
    # initial foreground region
    foreground_bounding_box = (0 , 0, width-1, height-1)
    bgdModel = np.zeros((1,65), np.float64)
    fgdModel = np.zeros((1,65), np.float64)

    binary_mask_threshold = create_binary_mask_threshold(pil_image, 162)

    grayscale_mask = np.zeros((width, height), np.uint8)
    grayscale_mask = np.where(binary_mask_threshold==False, cv2.GC_PR_BGD, cv2.GC_PR_FGD)
    grayscale_mask = grayscale_mask.astype(np.uint8)

    numIterations = 5

    cv2.grabCut(image, grayscale_mask, foreground_bounding_box, bgdModel, fgdModel, numIterations, cv2.GC_INIT_WITH_MASK)

    # convert to binary mask
    binary_mask = np.where((grayscale_mask==2) | (grayscale_mask==0), False, True)
    return binary_mask

def create_binary_mask_grabcut(pil_image):
    image = convert_pil_image_to_opencv_image(pil_image)
    (width, height, numChannels) = image.shape

    # initialize parameters for grabcut
    # initial foreground region
    foreground_bounding_box = (0 , 0, width-1, height-1)
    bgdModel = np.zeros((1,65), np.float64)
    fgdModel = np.zeros((1,65), np.float64)
    grayscale_mask = np.zeros((width, height), np.uint8)
    numIterations = 5

    cv2.grabCut(image, grayscale_mask, foreground_bounding_box, bgdModel, fgdModel, numIterations, cv2.GC_INIT_WITH_RECT)

    # convert to binary mask
    binary_mask = np.where((grayscale_mask==2) | (grayscale_mask==0), False, True)
    return binary_mask

def create_binary_mask_grayscale(grayscale_image, threshold):
    return np.array(grayscale_image) > threshold
