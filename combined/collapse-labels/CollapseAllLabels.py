import logging
import logging.config
import os
import glob
import yaml

def setup_logging(logging_config_path: str = 'logging.yaml', default_level: int = logging.INFO) -> None:
    if os.path.exists(logging_config_path):
        with open(logging_config_path, 'rt') as file:
            config = yaml.safe_load(file.read())
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)

def get_immediate_subdirectories(super_dir):
    return [name for name in os.listdir(super_dir) if os.path.isdir(os.path.join(super_dir, name))]

def main():
    # Logging Setup
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info('Started')

    valid_extensions = [
        "txt",
        "TXT",
    ] 

    # Output super directory 
    path_super_dir_output = "collapsed_labels" # Specify path to labels folder here

    # Loading inputs
    path_super_dir = "labels" # Specify path to labels folder here
    subdirs = get_immediate_subdirectories(path_super_dir) # train, val, test
    logger.info("Number of subdirectories found: %i", len(subdirs))

    label_input_ouput_paths_dict = {}
    for subdir in subdirs:
        logger.info("Proessing subdir %s", subdir)
        path_subdir = os.path.join(path_super_dir, subdir)
        subsubdirs = get_immediate_subdirectories(path_subdir) # modal, amodal 
        logger.info("Number of subsubdirectories found: %i", len(subsubdirs))
        for subsubdir in subsubdirs:
            logger.info("Processing subsubdir %s", subsubdir)
            path_subsubdir = os.path.join(path_subdir, subsubdir)
            for valid_extension in valid_extensions:
                search_path = path_subsubdir + "/" + "*." + valid_extension
                logger.debug("Searching %s using %s", path_subsubdir, search_path)  
                for file_path in glob.glob(search_path):
                    filename_with_ext = os.path.basename(file_path)
                    output_file_path = os.path.join(path_super_dir_output, subdir, subsubdir, filename_with_ext)
                    label_input_ouput_paths_dict[file_path] = output_file_path
                    logger.debug("Input: %s, Output: %s", file_path, output_file_path)

    logger.info("Number of labels found: %i", len(label_input_ouput_paths_dict))
    for input_file_path in label_input_ouput_paths_dict:
        output_file_path = label_input_ouput_paths_dict[input_file_path]
        logger.debug("Input: %s, Output: %s", input_file_path, output_file_path)

    logger.info('Finished')

if __name__ == '__main__':
    main()
