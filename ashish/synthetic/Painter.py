import logging
import logging.config
import os
import glob
import yaml
import math
import numpy as np
from tqdm import tqdm
import ImagePath
import Annotation
from PIL import Image, ImageOps, ImageDraw

def setup_logging(logging_config_path: str = 'logging.yaml', default_level: int = logging.INFO) -> None:
    if os.path.exists(logging_config_path):
        with open(logging_config_path, 'rt') as file:
            config = yaml.safe_load(file.read())
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)

def clean_yolo(yolo_label: np.ndarray):
    if len(yolo_label.shape) == 1:  # if there's only one label, shape will be (5,) which is 1-d, this fixes it.
        yolo_label = np.array([yolo_label])
    return yolo_label

def rel_to_absolute_label(annotation, image):
    x, y, width, height = annotation[1:5]
    (imageWidth, imageHeight) = image.size
    new_width = math.floor(width * imageWidth)
    new_height = math.floor(height * imageHeight)
    xTopLeft = math.floor(x * imageWidth) - new_width//2
    yTopLeft = math.floor(y * imageHeight) - new_height//2
    xBottomRight = math.floor(x * imageWidth) + new_width//2
    yBottomRight = math.floor(y * imageHeight) + new_height//2
    return (xTopLeft, yTopLeft, xBottomRight, yBottomRight)

def draw_bounding_box(
    image: Image,
    annotation: np.ndarray
) -> Image:
    # Compute top left and bottom right coordinates
    annotation = clean_yolo(annotation)
    (N, _) = annotation.shape

    boundingBoxes = [] 
    for i in range(N):
        boundingBox = rel_to_absolute_label(annotation[i], image)
        boundingBoxes.append(boundingBox)

    copiedImage = image.copy()
    draw = ImageDraw.Draw(copiedImage)

    for boundingBox in boundingBoxes:
        (xTopLeft, yTopLeft, xBottomRight, yBottomRight) = boundingBox
        draw.rectangle(
            [(xTopLeft, yTopLeft), (xBottomRight, yBottomRight)],
            width=1,
            outline=(0,255,0))

    return copiedImage 

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
            key = "{originalName}.{id}".format(originalName = fileNameParts[0], id = fileNameParts[2])
            dict[key] = file_path 

    logger.info("Number of labels found in %s is %s", folderPath, len(dict))
    return dict

def get_image_label_paths(folderPath, imageFolderName = "images", labelFolderName = "labels"):
    folderPathImages = os.path.join(folderPath, imageFolderName)
    folderPathLabels = os.path.join(folderPath, labelFolderName)

    imagePathDict = get_image_paths_dict(folderPathImages)
    labelPathDict = get_label_paths_dict(folderPathLabels)

    dict = {}
    for key in imagePathDict:
        # Assuming all keys are present properly in both dictionaries.
        dict[key] = (imagePathDict[key], labelPathDict[key])

    return dict

def process_occlusion_class(path_dir, output_path_dir):
    logger = logging.getLogger(__name__)

    dict_image_label_paths = get_image_label_paths(path_dir)

    if not os.path.isdir(output_path_dir):
        os.makedirs(output_path_dir)
        logger.info("Created target directory %s", output_path_dir)

    for key in dict_image_label_paths:
        logger.debug("key: %s, imagePath: %s", key, dict_image_label_paths[key][0])
        logger.debug("key: %s, labelPath: %s", key, dict_image_label_paths[key][1])
        image = Image.open(dict_image_label_paths[key][0])
        annotation = Annotation.read_background_annotation(dict_image_label_paths[key][1]) 
        drawnImage = draw_bounding_box(image, annotation)
        output_filename = "{originalName}.{extension}".format(originalName = key, extension = "png")
        output_path = os.path.join(output_path_dir, output_filename)
        drawnImage.save(output_path)
        logger.debug("Saved drawn image for key %s in %s", key, output_path)

def main():
    # Logging Setup
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info('Started')

    # Output super directory 
    path_super_dir_output = "/Users/ashish/Source/Courses/Stanford/CS231N/project/repo/0/cs231n/ashish/synthetic/debug/train" # Specify path to labels folder here

    # Loading inputs
    path_super_dir = "/Users/ashish/Source/Courses/Stanford/CS231N/project/repo/0/cs231n/ashish/synthetic/output/train" # Specify path to labels folder here
    subdirs = ImagePath.get_immediate_subdirectories(path_super_dir) # train, val, test
    logger.info("Number of subdirectories found: %i", len(subdirs))

    for subdir in tqdm(subdirs, desc="Overall"):
        logger.info("Proessing subdir %s", subdir)
        path_subdir = os.path.join(path_super_dir, subdir)

        path_subdir_output = os.path.join(path_super_dir_output, subdir) 
        if not os.path.isdir(path_subdir_output):
            os.makedirs(path_subdir_output)
            logger.info("Created target directory %s", path_subdir_output)

        process_occlusion_class(path_subdir, path_subdir_output)

    logger.info('Finished')

if __name__ == '__main__':
    main()

