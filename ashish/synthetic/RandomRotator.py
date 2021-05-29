import logging
import logging.config
import math
import random
import numpy as np
import Annotation
import Occlusions
import SyntheticImage

from PIL import Image 
from ImageTransformer import ImageTransformer 

class RandomRotator(ImageTransformer):
    def transform(
        self,
        foregroundImage: Image
    ) -> Image:
        return RandomRotator.rotate(foregroundImage)

    @staticmethod
    def rotate(
        foregroundImage: Image
    ) -> Image:
        logger = logging.getLogger(__name__)

        # Create a new image which will not drop positive pixels simply because they go out of the image frame when the image is rotated. 
        expandedImage = RandomRotator.expand_image_as_per_diagonal(foregroundImage) 

        # Decide degree to rotate
        angle = random.randint(0,360) # [a,b] => 0 has two times the chance of any other value
        logger.debug('angle to rotate: %i', angle) 

        # Rotate the image
        rotatedImage = expandedImage.rotate(angle)

        # Crop the image to remove empty space
        croppedImage = RandomRotator.remove_empty_space_around_edges(rotatedImage)

        return croppedImage

    @staticmethod
    def expand_image_as_per_diagonal(image: Image) -> Image:
        logger = logging.getLogger(__name__)

        # Create blank square image of size diagonal*2
        width, height = image.size
        logger.debug('Original image size: (%i, %i)', width, height)

        diagX, diagY= math.ceil(width/2), math.ceil(height/2)
        diagLength = math.ceil(math.sqrt(diagX**2 + diagY**2))
        widthExpanded = heightExpanded = diagLength*2
        logger.debug('Expanded image size: (%i, %i)', widthExpanded, heightExpanded)

        blankImage = Image.new('RGBA', (widthExpanded, heightExpanded), (0, 0, 0, 0))
        centerExpanded = (diagLength, diagLength)
        syntheticImage, topLeftPoint, mask = SyntheticImage.create_synthetic_image(blankImage, image, 100, centerExpanded)
        return syntheticImage

    @staticmethod
    def remove_empty_space_around_edges(expandedImage: Image) -> Image:
        logger = logging.getLogger(__name__)

        mask = Occlusions.create_grayscale_image_mask(expandedImage, 100)
        width, height = expandedImage.size
        logger.debug('Original image size: (%i, %i)', width, height)

        (topLeftPointMaskWithinForegroundImageX, topLeftPointMaskWithinForegroundImageY, maskWidthWithinForegroundImage, maskHeightWithinForegroundImage) = Annotation.get_bounding_box(np.array(mask))
        croppedImage = expandedImage.crop((
            max(0, topLeftPointMaskWithinForegroundImageX-1),
            max(0, topLeftPointMaskWithinForegroundImageY-1),
            min(topLeftPointMaskWithinForegroundImageX+maskWidthWithinForegroundImage+1, width),
            min(topLeftPointMaskWithinForegroundImageY+maskHeightWithinForegroundImage+1, height)))

        widthCropped, heightCropped = croppedImage.size
        logger.debug('Cropped image size: (%i, %i)', widthCropped, heightCropped)

        return croppedImage

