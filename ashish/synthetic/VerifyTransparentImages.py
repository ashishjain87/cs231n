import logging
import logging.config
import os
import glob
import yaml
import ImagePath
import tqdm
import Occlusions as occlusions
import CreateSyntheticData
from PIL import Image, ImageOps

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
    path_foreground_super_dir = "input/foreground/single-shot-original-size/val"
    subdirs = ImagePath.get_immediate_subdirectories(path_foreground_super_dir)
    logger.info("Number of subdirectories found: %i", len(subdirs))

    # TODO: Move to config
    foregound_valid_extensions = [
        "jpg",
        "jpeg",
        "JPEG",
        "JPG",
        "png",
        "PNG",
    ] 

    foreground_image_paths = []
    for subdir in subdirs:
        logger.info("Processing %s", subdir)
        path_foreground_dir = os.path.join(path_foreground_super_dir, subdir)
        for valid_extension in foregound_valid_extensions:
            foreground_search_path = path_foreground_dir + "/" + "*." + valid_extension
            logger.debug("Searching %s using %s", path_foreground_dir, foreground_search_path)  
            for file_path in glob.glob(foreground_search_path):
                foreground_image_paths.append(file_path)
                logger.debug("Found foreground image at: %s", file_path)

    logger.info("Number of images found: %i", len(foreground_image_paths))
    for foreground_image_path in foreground_image_paths:
        image = Image.open(foreground_image_path)
        actualSize = image.size
        isTransparent = occlusions.is_transparent(image)
        intendedSize, intendedSizeDivisor = CreateSyntheticData.determine_scaled_size_as_per_original_propotions(image)
        logger.info("File path: %s", foreground_image_path)
        logger.info("IsTransparent: %s", str(isTransparent))
        logger.info("Original Size: (%i, %i)", actualSize[0], actualSize[1])
        logger.info("Intended Size: (%i, %i)", intendedSize[0], intendedSize[1]) 

    logger.info('Finished')

if __name__ == '__main__':
    main()
