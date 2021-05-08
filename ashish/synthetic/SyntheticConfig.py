import confuse
import logging
import os

class SyntheticConfig:
    def __init__(self, configurationName: str) -> None:
        self.configurationName = configurationName 

    def initialize(self):
        logger = logging.getLogger(__name__)
        logger.info('Reading configuration file %s', self.configurationName)

        config = confuse.Configuration('CreateSyntheticData')

        #TODO: Add support for reading images and labels for background images
        self.path_foreground_dir = str(config['pathForegroundDir'])
        self.path_background_dir = str(config['pathBackgroundDir'])
        self.threshold = int(str(config['threshold']))
        self.cur_image_id = int(str(config['StartImageId']))
        self.target_dir = str(config['output']['pathTargetDir'])

        self.images_dir_name = str(config['output']['imagesDirName'])
        self.target_dir_path_images = os.path.join(self.target_dir, self.images_dir_name)

        self.annotations_dir_name = str(config['output']['annotationsDirName'])
        self.target_dir_path_annotations = os.path.join(self.target_dir, self.annotations_dir_name)

        self.path_mapping_file_occlusion_name_to_occlusion_id = str(config['pathOcclusionNameToOcclusionIdMapping'])

        self.probability_prioritize_objects_of_interest = float(str(config['probabilityPrioritizeObjectsOfInterest']))
        assert self.probability_prioritize_objects_of_interest <= 1.0, "probability cannot be greater than 1"
        assert self.probability_prioritize_objects_of_interest >= 0.0, "probability cannot be less than 0"
        
    def log(self):
        logger = logging.getLogger(__name__)

        logger.info('pathForegroundDir %s', self.path_foreground_dir)
        logger.info('pathBackgroundDir %s', self.path_background_dir)
        logger.info('threshold %s', self.threshold)
        logger.info('start image id %s', self.cur_image_id)
        logger.info('target dir %s', self.target_dir)
        logger.info('images dir name %s', self.images_dir_name)
        logger.info('pathOcclusionNameToOcclusionIdMapping %s', self.path_mapping_file_occlusion_name_to_occlusion_id)
        logger.info('probabilityPrioritizeObjectsOfInterest %f', self.probability_prioritize_objects_of_interest)
