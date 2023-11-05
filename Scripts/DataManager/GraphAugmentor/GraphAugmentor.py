import uuid

from Scripts.DataManager.GraphLoader.GraphLoader import GraphDataModule
from abc import ABC, abstractmethod
import torch_geometric.transforms as T


class GraphAugmentor(ABC):
    def __init__(self, name, inplace: bool = False):
        self.name = name
        self._unique_id = uuid.uuid4()
        self.inplace = inplace

    @abstractmethod
    def augment(self, graph_loader: GraphDataModule):
        pass


