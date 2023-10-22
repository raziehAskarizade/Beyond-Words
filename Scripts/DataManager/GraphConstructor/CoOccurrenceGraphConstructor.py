import torch

from GraphConstructor import GraphConstructor
from torch_geometric.data import Data
import spacy
from Scripts.Configs.ConfigClass import Config


class CoOccurrenceGraphConstructor(GraphConstructor):

    def __init__(self, text: str, config: Config):
        super(CoOccurrenceGraphConstructor, self).__init__(text, config)

    def _generate_graph(self):
        nlp = spacy.load(self.config.spacy.pipeline)
        doc = nlp(self.text)

        self.x = torch.tensor()
        self.y = torch.tensor()
        self.edge_index = torch.tensor()

