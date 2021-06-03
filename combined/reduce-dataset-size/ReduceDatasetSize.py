import logging
import logging.config
import os
import glob
import yaml
import numpy as np
import random
from tqdm import tqdm
import ImagePath

def setup_logging(logging_config_path: str = 'logging.yaml', default_level: int = logging.INFO) -> None:
    if os.path.exists(logging_config_path):
        with open(logging_config_path, 'rt') as file:
            config = yaml.safe_load(file.read())
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)

def get_immediate_subdirectories(super_dir):
    return [name for name in os.listdir(super_dir) if os.path.isdir(os.path.join(super_dir, name))]

def get_image_paths_dict(folderPath):
    logger = logging.getLogger(__name__)
    valid_extensions = [ "png", ]
    
    dict = {}
    for valid_extension in valid_extensions:
        search_path = folderPath + "/" + "*." + valid_extension
        logger.debug("Searching %s using %s", folderPath, search_path)  
        for file_path in glob.glob(search_path):
            filename_without_ext = ImagePath.get_filename_without_ext(file_path)
            dict[filename_without_ext] = file_path 

    logger.info("Number of images found in %s is %s", folderPath, len(dict))
    return dict

def get_label_paths_dict(folderPath):
    logger = logging.getLogger(__name__)
    valid_extensions = [ "txt", ]
    
    dict = {}
    for valid_extension in valid_extensions:
        search_path = folderPath + "/" + "*." + valid_extension
        logger.debug("Searching %s using %s", folderPath, search_path)  
        for file_path in glob.glob(search_path):
            filename_without_ext = ImagePath.get_filename_without_ext(file_path)
            fileNameParts = filename_without_ext.split(".")
            key = "{originalName}.{id}".format(originalName = fileNameParts[0], id = fileNameParts[1])
            dict[key] = file_path 

    logger.info("Number of labels found in %s is %s", folderPath, len(dict))
    return dict

def main():
    # Logging Setup
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info('Started')

    # Output directory 
    path_dir_amodal_labels_output = "test/labels/combined-train-reduced/amodal" # Specify path to labels folder here
    path_dir_modal_labels_output = "test/labels/combined-train-reduced/modal" # Specify path to labels folder here
    path_dir_images_output = "test/images/combined-train-reduced" # Specify path to images folder here

    # Loading inputs
    path_dir_amodal_labels_input = "test/labels/combined-train/amodal" # Specify path to labels folder here
    path_dir_modal_labels_input = "test/labels/combined-train/modal" # Specify path to labels folder here
    path_dir_images_input = "test/images/combined-train" # Specify path to images folder here

    amodalLabelsPathDict = get_label_paths_dict(path_dir_amodal_labels_input)
    modalLabelsPathDict = get_label_paths_dict(path_dir_modal_labels_input)
    imagesPathDict = get_image_paths_dict(path_dir_images_input)

    skipThreshold = 0.0
    for key in tqdm(imagesPathDict, desc="Processing files"):
        # Assuming all keys are present properly in both dictionaries.

        # Image
        inputImagePath = imagesPathDict[key]
        inputImageFilename = os.path.basename(inputImagePath)
        outputImagePath = os.path.join(path_dir_images_output, inputImageFilename)

        # Modal
        inputModalLabelPath = modalLabelsPathDict[key]
        modalLabelFileName = os.path.basename(inputModalLabelPath)
        outputModalLabelPath = os.path.join(path_dir_modal_labels_output, modalLabelFileName)

        # Amodal
        inputAmodalLabelPath = amodalLabelsPathDict[key]
        amodalLabelFileName = os.path.basename(inputAmodalLabelPath)
        outputAmodalLabelPath = os.path.join(path_dir_amodal_labels_output, amodalLabelFileName)

        randomNumber = random.random()
        logger.debug('randomNumber: %f', randomNumber)

        if randomNumber <= skipThreshold: 
            logger.debug('Skipping image as %f < %f', randomNumber, skipThreshold)
            continue

        logger.debug('Copying image as %f < %f', randomNumber, skipThreshold)
        copyImageCommand = f"cp {inputImagePath} {outputImagePath}"
        os.system(copyImageCommand)
        logger.debug(copyImageCommand)

        copyModalLabelCommand = f"cp {inputModalLabelPath} {outputModalLabelPath}"
        os.system(copyModalLabelCommand)
        logger.debug(copyModalLabelCommand)

        copyAmodalLabelCommand = f"cp {inputAmodalLabelPath} {outputAmodalLabelPath}"
        os.system(copyAmodalLabelCommand)
        logger.debug(copyAmodalLabelCommand)

    logger.info('Finished')

if __name__ == '__main__':
    main()
