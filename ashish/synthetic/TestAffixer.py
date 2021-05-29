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
    backgroundImagePath = "input/background/images/000016.png" 
    backgroundAnnotationPath = "input/background/labels/000016.txt"
    foregroundImagePath = "input/foreground/single-shot/train/PlasticBag/PlasticBag04_122x150.png"

    backgroundImage = Image.open(backgroundImagePath)
    foregroundImage = Image.open(foregroundImagePath)
    backgroundAnnotation = Annotation.read_background_annotation(backgroundAnnotationPath) 
    
    # Run the affixer
    affixer: Affixer = FixedAffixer() # Your code goes here!
    (centerPoint, scale) = affixer.decide_where_and_scale(backgroundImage, backgroundAnnotation, foregroundImage)
    centerX, centerY = centerPoint
    logger.info("centerX: %s, centerY: %s, scale: %s", centerX, centerY, scale)

    # Generate the image as per the outputs of the Affixer
    foregroundImageResized = CreateSyntheticData.resize_image(foregroundImage, backgroundImage, scale)
    (syntheticImage, point, mask) = CreateSyntheticData.create_synthetic_image(
        backgroundImage,
        foregroundImageResized,
        100,
        centerPoint 
    )

    # Save the image
    targetPath = "test-affixer.png"
    CreateSyntheticData.save_synthetic_image(
        backgroundImage,
        syntheticImage,
        foregroundImage,
        point,
        100,
        targetPath,
    )

    logger.info('Finished')

if __name__ == '__main__':
    main()
