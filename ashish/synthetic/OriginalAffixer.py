import logging
import logging.config
import math
import numpy as np
import random
from abc import ABC
from typing import Tuple
from PIL import Image
from Affixer import Affixer

class OriginalAffixer(Affixer):
    def  __init__(self, probability_prioritize_objects_of_interest: float, min_percentage_of_background_height: float = 10.0, max_percentage_of_background_height: float = 80.0) -> None:
        super().__init__()
        self.probability_prioritize_objects_of_interest = probability_prioritize_objects_of_interest
        assert min_percentage_of_background_height > 0 and min_percentage_of_background_height < 100, "min_percentage_of_background_height should be between 0 and 100"
        assert max_percentage_of_background_height > 0 and max_percentage_of_background_height < 100, "max_percentage_of_background_height should be between 0 and 100"
        assert min_percentage_of_background_height <= max_percentage_of_background_height, "min_percentage_of_background_height cannot be greater than max_percentage_of_background_height"
        self.min_percentage_of_background_height = min_percentage_of_background_height
        self.max_percentage_of_background_height = max_percentage_of_background_height

    def decide_where_and_scale(
        self,
        backgroundImage: Image,
        backgroundAnnotations: np.ndarray,
        foregroundImage: Image
    ) -> Tuple[Tuple[int, int], float]:
        center_point = self.randomly_sample_point_within_image_or_object_of_interest(
            backgroundImage,
            foregroundImage,
            backgroundAnnotations,
            self.probability_prioritize_objects_of_interest)
        scale = self.randomly_sample_image_resize_scale(foregroundImage.size, backgroundImage.size)
        return (center_point, scale)

    def randomly_sample_image_resize_scale(
        self,
        fg_image_size,
        bg_image_size
    ) -> float:
        logger = logging.getLogger(__name__)

        (fg_width, fg_height) = fg_image_size
        (bg_width, bg_height) = bg_image_size

        logger.debug('fg_size: (%i, %i)', fg_width, fg_height) 
        logger.debug('bg_size: (%i, %i)', bg_width, bg_height) 

        # We are working with wide images. Therefore, worrying about the height is sufficient.
        min_height = math.floor(self.min_percentage_of_background_height*0.01*bg_height)
        max_height = math.floor(self.max_percentage_of_background_height*0.01*bg_height)

        low = min_height*1.0/fg_height
        high = max_height*1.0/fg_height

        logger.debug('min_height: %i, max_height: %i, low: %f, high: %f', min_height, max_height, low, high) 

        # We preserve the aspect ratio for the foreground image. Therefore, scale remains the same across width and height 
        scale = random.uniform(low, high) # [low, high]
    
        scaled_fg_width = int(scale * fg_width)
        scaled_fg_height = int(scale * fg_height)

        logger.debug('scale: %f, scaled_fg_size: (%i, %i)', scale, scaled_fg_width, scaled_fg_height) 

        return scale

    def randomly_sample_point_within_image_or_object_of_interest(
        self,
        background_image,
        occlusion_image,
        background_annotations,
        p
    ):
        logger = logging.getLogger(__name__)

        (numRows, numCols) = background_annotations.shape 
        useImage = True
        if numRows == 0: # TODO: TEST if no annotations are there for an image
            logger.debug('Zero annotations found')
        else:
            r = random.random()
            useImage = True if r > p else False
            logger.debug('r was %f, p was %s, useImage %s', r, p, useImage)

        if useImage:
            logger.debug('Randomly sampling from whole image')
            return OriginalAffixer.randomly_sample_point_within_image(background_image, occlusion_image)
        else:
            index = OriginalAffixer.randomly_choose_object_of_interest(numRows)
            logger.debug('Randomly sampling from object of interest. Index chosen %i', index)
            (topLeft, bottomRight) = OriginalAffixer.get_top_left_bottom_right_coordinates(background_annotations, index, background_image)

            (xTopLeft, yTopLeft) = topLeft
            (xBottomRight, yBottomRight) = bottomRight
            logger.debug('Rectangle chosen topLeft: (%i, %i), bottomRight: (%i, %i)', xTopLeft, yTopLeft, xBottomRight, yBottomRight)

            return OriginalAffixer.randomly_sample_point_within_rectangle_assuming_center_point_to_affix_foreground_image(topLeft, bottomRight, background_image, occlusion_image)

    @staticmethod
    def randomly_sample_point_within_rectangle_assuming_center_point_to_affix_foreground_image(backgroundImageTopLeftCoordinates, backgroundImageBottomRightCoordinates, pil_image_bg, pil_image_fg):
        backgroundImageTopLeftCoordinates, backgroundImageBottomRightCoordinates = OriginalAffixer.correct_bounding_box_coordinates_assuming_center_point_to_affix_foreground_image(backgroundImageTopLeftCoordinates, backgroundImageBottomRightCoordinates, pil_image_bg, pil_image_fg)
        return OriginalAffixer.randomly_sample_point_within_rectangle(backgroundImageTopLeftCoordinates, backgroundImageBottomRightCoordinates)

    @staticmethod
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

    @staticmethod
    def correct_bounding_box_coordinates_assuming_center_point_to_affix_foreground_image(backgroundImageTopLeftCoordinates, backgroundImageBottomRightCoordinates, pil_image_bg, pil_image_fg):
        logger = logging.getLogger(__name__)

        (xTopLeft, yTopLeft) = backgroundImageTopLeftCoordinates
        (xBottomRight, yBottomRight) = backgroundImageBottomRightCoordinates

        (minWidth, minHeight, maxWidth, maxHeight) = OriginalAffixer.get_largest_bounding_box_assuming_using_center_point_to_affix_foreground_image(pil_image_bg, pil_image_fg)

        logger.debug('Largest bounding box coordinates: (%s, %s), (%s, %s)', minWidth, minHeight, maxWidth, maxHeight) 
        logger.debug('Original coordinates: (%s, %s), (%s, %s)', xTopLeft, yTopLeft, xBottomRight, yBottomRight) 

        xTopLeft = max(xTopLeft, minWidth)
        yTopLeft = max(yTopLeft, minHeight)
        
        xBottomRight = min(xBottomRight, maxWidth)
        yBottomRight = min(yBottomRight, maxHeight)

        logger.debug('Revised coordinates: (%s, %s), (%s, %s)', xTopLeft, yTopLeft, xBottomRight, yBottomRight) 

        return ((xTopLeft, yTopLeft), (xBottomRight, yBottomRight))

    @staticmethod
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

    @staticmethod
    def randomly_sample_point_within_image(pil_image_bg, pil_image_fg):
        logger = logging.getLogger(__name__)

        (backgroundWidth, backgroundHeight) = pil_image_bg.size
        (foregroundWidth, foregroundHeight) = pil_image_fg.size

        if foregroundWidth >= backgroundWidth:
            logger.error('%s is greater or equal to %s specifically %s >= %s', 'foregroundWidth', 'backgroundWidth', foregroundWidth, backgroundWidth)

        if foregroundHeight >= backgroundHeight:
            logger.error('%s is greater or equal to %s specifically %s >= %s', 'foregroundHeight', 'backgroundHeight', foregroundHeight, backgroundHeight)

        (minWidth, minHeight, maxWidth, maxHeight) = OriginalAffixer.get_largest_bounding_box_assuming_using_center_point_to_affix_foreground_image(pil_image_bg, pil_image_fg)

        return OriginalAffixer.randomly_sample_point_within_rectangle_assuming_center_point_to_affix_foreground_image((minWidth, minHeight), (maxWidth, maxHeight), pil_image_bg, pil_image_fg)

    @staticmethod
    def randomly_choose_object_of_interest(num_annotations) -> int: # returns index
        return random.randint(0,num_annotations-1)

    @staticmethod
    def yolo_to_kitti(annotation, pil_background_image):
        x, y, width, height = annotation[1], annotation[2], annotation[3], annotation[4]
        (image_width, image_height) = pil_background_image.size 
        new_width = round(width * image_width)
        new_height = round(height * image_height)
        left = round(x * image_width) - new_width//2
        top = round(y * image_height) - new_height//2
        return (left, top, new_width, new_height)

    @staticmethod
    def get_top_left_bottom_right_coordinates(background_annotations, index, pil_background_image):
        annotation = background_annotations[index]
        (xTopLeft, yTopLeft, width, height) = OriginalAffixer.yolo_to_kitti(annotation, pil_background_image)
        xBottomRight = xTopLeft + width
        yBottomRight = yTopLeft + height
        return ((xTopLeft, yTopLeft), (xBottomRight, yBottomRight))

