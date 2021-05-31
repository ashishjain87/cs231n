import logging
import logging.config
import os
import glob
import yaml
import numpy as np
from tqdm import tqdm
import argparse

KITTI_CATEGORY_TO_COLLAPSED_CATEGORY = {3: 1, 4: 1, 5: 1, 0: 0, 1: 0, 2: 0, 6: 0, 7: 2, 8: 2}
CLASS_NAMES_MAP = {0: "vehicle", 1: "person", 2: "misc"}

def setup_logging(logging_config_path: str = 'logging.yaml', default_level: int = logging.INFO) -> None:
    if os.path.exists(logging_config_path):
        with open(logging_config_path, 'rt') as file:
            config = yaml.safe_load(file.read())
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--output-labels-dir', type=str, required=True, help="output super directory")
    parser.add_argument('--input-labels-dir', type=str, required=True, help="input super directory")

    return parser.parse_args()

def get_immediate_subdirectories(super_dir):
    return [name for name in os.listdir(super_dir) if os.path.isdir(os.path.join(super_dir, name))]

def read_yolo(path_to_label_file: str) -> np.ndarray:
    yolo_label = np.genfromtxt(path_to_label_file, delimiter=" ", dtype=float, encoding=None)
    return clean_yolo(yolo_label)

def clean_yolo(yolo_label: np.ndarray):
    if len(yolo_label.shape) == 1:  # if there's only one label, shape will be (5,) which is 1-d, this fixes it.
        yolo_label = np.array([yolo_label])
    return yolo_label

def write_label_to_file(labels: np.ndarray, filepath: str):
    np.savetxt(filepath, labels, delimiter=" ")

def collapse_labels(path_to_label_file_in: str, path_to_label_file_out: str):
    yolo_label = read_yolo(path_to_label_file_in)
    for i in range(yolo_label.shape[0]):
        yolo_label[i, 0] = KITTI_CATEGORY_TO_COLLAPSED_CATEGORY[int(yolo_label[i, 0])]
    write_label_to_file(yolo_label, path_to_label_file_out)

def main():
    # Logging Setup
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info('Started')

    args = get_args()

    valid_extensions = [
        "txt",
        "TXT",
    ]

    # Output super directory
    path_super_dir_output = args.output_labels_dir

    # Loading inputs
    path_super_dir = args.input_labels_dir
    subdirs = get_immediate_subdirectories(path_super_dir) # train, val, test
    logger.info("Number of subdirectories found: %i", len(subdirs))

    label_input_ouput_paths_dict = {}
    for subdir in tqdm(subdirs, desc="Traverse folders"):
        logger.info("Proessing subdir %s", subdir)
        path_subdir = os.path.join(path_super_dir, subdir)
        subsubdirs = get_immediate_subdirectories(path_subdir) # modal, amodal
        logger.info("Number of subsubdirectories found: %i", len(subsubdirs))

        path_subdir_output = os.path.join(path_super_dir_output, subdir)
        if not os.path.isdir(path_subdir_output):
            os.makedirs(path_subdir_output)
            logger.info("Created target directory %s", path_subdir_output)

        for subsubdir in subsubdirs:
            logger.info("Processing subsubdir %s", subsubdir)
            path_subsubdir = os.path.join(path_subdir, subsubdir)

            path_subsubdir_output = os.path.join(path_subdir_output, subsubdir)
            if not os.path.isdir(path_subsubdir_output):
                os.makedirs(path_subsubdir_output)
                logger.info("Created target directory %s", path_subsubdir_output)

            for valid_extension in valid_extensions:
                search_path = path_subsubdir + "/" + "*." + valid_extension
                logger.debug("Searching %s using %s", path_subsubdir, search_path)
                for file_path in glob.glob(search_path):
                    filename_with_ext = os.path.basename(file_path)
                    output_file_path = os.path.join(path_super_dir_output, subdir, subsubdir, filename_with_ext)
                    label_input_ouput_paths_dict[file_path] = output_file_path
                    logger.debug("Input: %s, Output: %s", file_path, output_file_path)

    logger.info("Number of labels found: %i", len(label_input_ouput_paths_dict))
    for input_file_path in tqdm(label_input_ouput_paths_dict, desc="Process files"):
        output_file_path = label_input_ouput_paths_dict[input_file_path]
        collapse_labels(input_file_path, output_file_path)
        logger.debug("Processed Input : %s", input_file_path)
        logger.debug("Generated Output: %s", output_file_path)

    logger.info('Finished')


if __name__ == '__main__':
    main()
