import Annotation
import CreateSyntheticData
import logging
import logging.config
import os
import yaml

from PIL import Image, ImageOps

from Affixer import Affixer
from FixedAffixer import FixedAffixer
from OriginalAffixer import OriginalAffixer # lots of helper static methods which may prove useful inside your Affixer

from ImageTransformer import ImageTransformer
from NoOpImageTransformer import NoOpImageTransformer
from RandomRotator import RandomRotator
from RandomHorizontalFlipper import RandomHorizontalFlipper
from RandomVerticalFlipper import RandomVerticalFlipper
from DecoratorImageTransformer import DecoratorImageTransformer

def setup_logging(logging_config_path: str = 'logging.yaml', default_level: int = logging.INFO) -> None:
    if os.path.exists(logging_config_path):
        with open(logging_config_path, 'rt') as file:
            config = yaml.safe_load(file.read())
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)

def main():
    # Logging Setup
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info('Started')

    # Loading inputs
    foregroundImagePath = "input/foreground/single-shot/train/PlasticBag/PlasticBag04_122x150.png"
    foregroundImage = Image.open(foregroundImagePath)
    
    # Run the Image Transformer 
    transformer: ImageTransformer= DecoratorImageTransformer([RandomVerticalFlipper(), RandomHorizontalFlipper(), RandomRotator()]) # Your code goes here!
    transformedImage = transformer.transform(foregroundImage)

    # Generate the image as per the outputs of the Affixer
    # Save the image
    targetPath = "test-image-transformer.input.png"
    foregroundImage.save(targetPath)

    targetPath = "test-image-transformer.output.png"
    transformedImage.save(targetPath)

    logger.info('Finished')

if __name__ == '__main__':
    main()
