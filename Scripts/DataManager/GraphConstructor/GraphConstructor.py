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
        self.x, self.y, self.edge_index = self._generate_graph()
        self.data = Data(x=self.x, y=self.y, edge_index=self.edge_index)

    @abstractmethod
    def _generate_graph(self) -> Tuple[torch.Tensor | Any, torch.Tensor | Any, torch.Tensor | Any]:
        pass
