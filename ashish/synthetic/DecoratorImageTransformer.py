import logging
import logging.config
from typing import List
from PIL import Image 
from ImageTransformer import ImageTransformer 

class DecoratorImageTransformer(ImageTransformer):
    def  __init__(self, transformers: List[ImageTransformer]) -> None:
        super().__init__()

        logger = logging.getLogger(__name__)

        if transformers != None and len(transformers) > 0:
            self.transformers = transformers 
        else:
            logger.warn("No image transformers provided")
            self.transformers = List[ImageTransformer]() 

    def transform(
        self,
        foregroundImage: Image
    ) -> Image:
        logger = logging.getLogger(__name__)

        previousImageVersion = foregroundImage
        for transformer in self.transformers:
            transformerClassName = transformer.__class__.__name__
            logger.debug("Calling %s", transformerClassName)
            nextImageVersion = transformer.transform(previousImageVersion)
            previousImageVersion = nextImageVersion

        return previousImageVersion 

