import json
from os import path


class Config:

    def __init__(self, project_root_path: str, config_local_path: str = ''):
        self.root = project_root_path

        if config_local_path == '':
            config_local_path = 'Scripts/Configs/Config.json'
        config_path = path.join(self.root, config_local_path)

        with open(config_path, 'rt') as cf:
            config_data = json.load(cf)

        self.device = config_data['device'] if 'device' in config_data else 'cpu'

        if 'spacy' in config_data:
            self.spacy: SpacyConfig = SpacyConfig(config_data['spacy'])

        if 'data_root_dir' in config_data:
            self.data_root_dir = config_data['data_root_dir']


class SpacyConfig:
    def __init__(self, json_data: dict):
        if 'pipeline' in json_data:
            self.pipeline: str = json_data['pipeline']

