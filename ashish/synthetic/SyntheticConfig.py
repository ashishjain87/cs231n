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

        self.path_foreground_dir = str(config['pathForegroundDir'])
        self.path_foreground_super_dir = str(config['pathForegroundSuperDir'])

        #TODO: Add support for reading images and labels for background images
        self.path_background_dir = str(config['background']['pathBackgroundDir'])

        background_images_dir_name = str(config['background']['imagesDirName'])
        self.images_dir_name = background_images_dir_name
        self.background_dir_path_images = os.path.join(self.path_background_dir, background_images_dir_name)

        background_annotations_dir_name = str(config['background']['annotationsDirName'])
        self.annotations_dir_name = background_annotations_dir_name
        self.background_dir_path_annotations = os.path.join(self.path_background_dir, background_annotations_dir_name)

        self.threshold = int(str(config['threshold']))
        self.cur_image_id = int(str(config['StartImageId']))

        self.target_dir = str(config['output']['pathTargetDir'])
        output_images_dir_name = str(config['output']['imagesDirName'])
        self.target_dir_path_images = os.path.join(self.target_dir, output_images_dir_name)
        output_annotations_dir_name = str(config['output']['annotationsDirName'])
        self.target_dir_path_annotations = os.path.join(self.target_dir, output_annotations_dir_name)

        self.path_mapping_file_occlusion_name_to_occlusion_id = str(config['pathOcclusionNameToOcclusionIdMapping'])

        self.affixerType = str(config['affixerType'])

        self.probability_prioritize_objects_of_interest = float(str(config['probabilityPrioritizeObjectsOfInterest']))
        assert self.probability_prioritize_objects_of_interest <= 1.0, "probability cannot be greater than 1"
        assert self.probability_prioritize_objects_of_interest >= 0.0, "probability cannot be less than 0"
        
    def log(self):
        logger = logging.getLogger(__name__)

        logger.info('pathForegroundDir %s', self.path_foreground_dir)
        logger.info('pathForegroundSuperDir %s', self.path_foreground_super_dir)

        logger.info('pathBackgroundDir %s', self.path_background_dir)
        logger.info('background images dir path %s', self.background_dir_path_images)
        logger.info('background annotations dir path %s', self.background_dir_path_annotations)

        logger.info('imagesDirName %s', self.images_dir_name)
        logger.info('annotationsDirName %s', self.annotations_dir_name)

        logger.info('threshold %s', self.threshold)
        logger.info('start image id %s', self.cur_image_id)

        logger.info('target dir %s', self.target_dir)
        logger.info('target images dir path %s', self.target_dir_path_images)
        logger.info('target annotations dir path %s', self.target_dir_path_annotations)

        logger.info('pathOcclusionNameToOcclusionIdMapping %s', self.path_mapping_file_occlusion_name_to_occlusion_id)

        logger.info('affixerType %s', self.affixerType)

        logger.info('probabilityPrioritizeObjectsOfInterest %f', self.probability_prioritize_objects_of_interest)
