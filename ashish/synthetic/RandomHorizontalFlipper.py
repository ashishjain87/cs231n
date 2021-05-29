import logging
import logging.config
import random
from PIL import Image, ImageOps
from ImageTransformer import ImageTransformer 

class RandomHorizontalFlipper(ImageTransformer):
    def  __init__(self, flipProbability: float = 0.5) -> None:
        super().__init__()
        self.flipProbability = flipProbability

    def transform(
        self,
        foregroundImage: Image
    ) -> Image:
        logger = logging.getLogger(__name__)

        randomNumber = random.random()
        logger.debug('randomNumber: %f', randomNumber)

        if randomNumber < self.flipProbability: 
            logger.debug('Flipping image as %f < %f', randomNumber, self.flipProbability)
            originalFilename = foregroundImage.filename
            transformedImage = ImageOps.mirror(foregroundImage)
            transformedImage.filename = originalFilename # Filenames are necessary for later
            return transformedImage
        else:
            logger.debug('No change to image as %f > %f', randomNumber, self.flipProbability)
            return foregroundImage

