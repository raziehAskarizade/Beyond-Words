import json


class Config:

    def __init__(self, config_path: str = ''):
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

