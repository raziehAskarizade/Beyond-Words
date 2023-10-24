from abc import ABC, abstractmethod

from Scripts.DataManager.GraphConstructor.GraphConstructor import GraphConstructor


class GraphCollection(ABC):

    def __init__(self):
        pass


class LazyTextGraphCollection(ABC):

    def __init__(self, graph_constructor: GraphConstructor):
        self.graph_constructor = graph_constructor

    def GetNext(self):
        pass
