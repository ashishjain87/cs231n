import logging
import logging.config
import math
import numpy as np
import Annotation
import Painter
from abc import ABC
from typing import Tuple
from PIL import Image
from Affixer import Affixer
from OriginalAffixer import OriginalAffixer

class NoValueRegionAffixer(Affixer):
    def  __init__(self, min_percentage_of_background_height: float = 10.0, max_percentage_of_background_height: float = 50.0, excludeBorders=False) -> None:
        super().__init__()
        assert min_percentage_of_background_height > 0 and min_percentage_of_background_height < 100, "min_percentage_of_background_height should be between 0 and 100"
        assert max_percentage_of_background_height > 0 and max_percentage_of_background_height < 100, "max_percentage_of_background_height should be between 0 and 100"
        assert min_percentage_of_background_height <= max_percentage_of_background_height, "min_percentage_of_background_height cannot be greater than max_percentage_of_background_height"
        self.min_percentage_of_background_height = min_percentage_of_background_height
        self.max_percentage_of_background_height = max_percentage_of_background_height
        self.excludeBorders = excludeBorders

    def decide_where_and_scale(
        self,
        backgroundImage: Image,
        backgroundAnnotations: np.ndarray,
        foregroundImage: Image
    ) -> Tuple[Tuple[int, int], float]:
        logger = logging.getLogger(__name__)

        scale = self.randomly_sample_image_resize_scale(foregroundImage.size, backgroundImage.size)
        scaledForegroundSize = NoValueRegionAffixer.determine_new_image_size(foregroundImage.size, backgroundImage.size, scale)
        mask = NoValueRegionAffixer.generate_mask(backgroundImage.size, scaledForegroundSize, backgroundAnnotations, self.excludeBorders)

        if not NoValueRegionAffixer.is_it_possible_to_sample_from_mask(mask):
            return ((-1,-1), 0.0)
        
        (centerY, centerX) = NoValueRegionAffixer.randomly_choose_from_mask(mask)  
        logger.debug("Center point chosen: (%i, %i)", centerX, centerY)
        logger.debug("Scale chosen: %f)", scale)

        return ((centerX, centerY), scale)

    def randomly_sample_image_resize_scale(
        self,
        fg_image_size,
        bg_image_size
    ) -> float:
        originalAffixer: OriginalAffixer = OriginalAffixer(0.5, self.min_percentage_of_background_height, self.max_percentage_of_background_height)
        return originalAffixer.randomly_sample_image_resize_scale(fg_image_size, bg_image_size)

    @staticmethod
    def is_it_possible_to_sample_from_mask(mask):
        anyOnes = mask == 1
        return np.any(anyOnes)

    @staticmethod
    def randomly_choose_from_mask(mask):
        logger = logging.getLogger(__name__)

        flattenedMask = mask.flatten()
        N = flattenedMask.shape[0]
        sum = np.sum(flattenedMask)
        probabilityFlattenedMask = flattenedMask * 1. / sum 
        rng = np.random.default_rng()

        index = rng.choice(N, p = probabilityFlattenedMask)
        unraveledIndex = np.unravel_index(index, mask.shape)

        (row, col) = unraveledIndex
        logger.debug("FlattenedIndex: %i, UnraveledIndex (row, col): (%i, %i)", index, row, col)

        return unraveledIndex

    @staticmethod
    def generate_mask(backgroundSize, scaledForegroundSize, annotation, excludeBorders=True):
        logger = logging.getLogger(__name__)

        (backgroundWidth, backgroundHeight) = backgroundSize
        (scaledForegroundWidth, scaledForegroundHeight) = scaledForegroundSize

        mask = np.ones((backgroundHeight, backgroundWidth))
        halfScaledForegroundWidth, halfScaledForegroundHeight = math.ceil(scaledForegroundWidth/2), math.ceil(scaledForegroundHeight/2)

        if excludeBorders:
            logger.debug("Borders are being removed as excludeBorders is True")
            mask[:, :halfScaledForegroundWidth] = 0
            mask[:, -halfScaledForegroundWidth:] = 0
            mask[:halfScaledForegroundHeight, :] = 0
            mask[-halfScaledForegroundHeight:, :] = 0

        annotation = Annotation.clean_yolo(annotation)
        (N, _) = annotation.shape

        boundingBoxes = [] 
        for i in range(N):
            boundingBox = NoValueRegionAffixer.rel_to_absolute_label(annotation[i], backgroundSize)
            boundingBoxes.append(boundingBox)

        for boundingBox in boundingBoxes:
            (xTopLeft, yTopLeft, xBottomRight, yBottomRight) = NoValueRegionAffixer.compute_expanded_top_left_and_bottom_right_coordinates(backgroundSize, scaledForegroundSize, boundingBox)
            mask[yTopLeft:yBottomRight+1, xTopLeft:xBottomRight+1] = 0
        
        return mask

    @staticmethod
    def compute_expanded_top_left_and_bottom_right_coordinates(backgroundSize, scaledForegroundSize, boundingBox):
        (xTopLeft, yTopLeft, xBottomRight, yBottomRight) = boundingBox
        (backgroundWidth, backgroundHeight) = backgroundSize
        (scaledForegroundWidth, scaledForegroundHeight) = scaledForegroundSize
        halfScaledForegroundWidth, halfScaledForegroundHeight = math.ceil(scaledForegroundWidth/2), math.ceil(scaledForegroundHeight/2)

        revisedXTopLeft = xTopLeft - halfScaledForegroundWidth
        revisedYTopLeft = yTopLeft - halfScaledForegroundHeight
        revisedXBottomRight = xBottomRight + halfScaledForegroundWidth
        revisedYBottomRight = yBottomRight + halfScaledForegroundHeight

        clippedRevisedXTopLeft = max(0, revisedXTopLeft) 
        clippedRevisedYTopLeft = max(0, revisedYTopLeft) 

        clippedRevisedXBottomRight = min(revisedXBottomRight, backgroundWidth-1)  
        clippedRevisedYBottomRight = min(revisedYBottomRight, backgroundHeight-1)

        return (clippedRevisedXTopLeft, clippedRevisedYTopLeft, clippedRevisedXBottomRight, clippedRevisedYBottomRight)

    @staticmethod
    def rel_to_absolute_label(annotation, imageSize):
        x, y, width, height = annotation[1:5]
        (imageWidth, imageHeight) = imageSize
        new_width = math.floor(width * imageWidth)
        new_height = math.floor(height * imageHeight)
        xTopLeft = math.floor(x * imageWidth) - new_width//2
        yTopLeft = math.floor(y * imageHeight) - new_height//2
        xBottomRight = math.floor(x * imageWidth) + new_width//2
        yBottomRight = math.floor(y * imageHeight) + new_height//2
        return (xTopLeft, yTopLeft, xBottomRight, yBottomRight)

    @staticmethod
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
