from abc import ABC, abstractmethod

import torch
from torch_geometric.data import Data
from typing import Tuple, Any

from Scripts.Configs.ConfigClass import Config


class GraphConstructor(ABC):

    def __init__(self, text: str, config: Config):
        self.config: Config = config
        self.text = text
        self.device = config.device
        self.node_attr, self.node_label, self.edge_index, self.edge_attr, self.edge_label = None, None, None, None, None
        self.data = Data(x=self.node_attr, y=self.node_label, edge_index=self.edge_index)

