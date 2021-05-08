import csv
import logging
import logging.config
import os
import SyntheticConfig as synthetic_config
import yaml

class Startup:
    def configure(self) -> None:
        self.setup_logging()

        logger = logging.getLogger(__name__)
        logger.info('Started')

        self.setup_configuration()
        self.setup_occlusion_name_to_occlusion_id_dict(self.synthetic_config.path_mapping_file_occlusion_name_to_occlusion_id)

        if not os.path.isdir(self.synthetic_config.target_dir):
            os.mkdir(self.synthetic_config.target_dir)
            logger.info('Created target super directory %s', self.synthetic_config.target_dir)

        logger.info('Finished')

    def setup_logging(self, logging_config_path: str = 'logging.yaml', default_level: int = logging.INFO) -> None:
        if os.path.exists(logging_config_path):
            with open(logging_config_path, 'rt') as file:
                config = yaml.safe_load(file.read())
            logging.config.dictConfig(config)
        else:
            logging.basicConfig(level=default_level)

    def setup_occlusion_name_to_occlusion_id_dict(self, path_mapping_file_occlusion_name_to_occlusion_id: str) -> None:
        logger = logging.getLogger(__name__)
        occlusion_name_occlusion_id_dict = {}
        with open(path_mapping_file_occlusion_name_to_occlusion_id) as file_handle:
            reader = csv.reader(file_handle)
            occlusion_name_occlusion_id_dict = { rows[0].upper():int(rows[1]) for rows in reader }

        self.occlusion_name_occlusion_id_dict = occlusion_name_occlusion_id_dict
        logger.info('Mapping dictionary %s was successfully setup from %s', 'occlusion_name_occlusion_id_dict', path_mapping_file_occlusion_name_to_occlusion_id) 

    def setup_configuration(self) -> None:
        self.synthetic_config = synthetic_config.SyntheticConfig('CreateSyntheticData')
        self.synthetic_config.initialize()
        self.synthetic_config.log()
