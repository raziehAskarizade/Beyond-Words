import copy

from GraphAugmentor import GraphAugmentor
from collections import OrderedDict
from uuid import UUID
from Scripts.DataManager.GraphLoader.GraphLoader import GraphLoader


class GraphAugmentorPipeline(GraphAugmentor):

    def __init__(self):
        super(GraphAugmentorPipeline, self).__init__('GraphAugmentorCollection')
        self._augment_pipeline_dict: OrderedDict[UUID, GraphAugmentor] = OrderedDict()

    def add(self, graph_augmentor: GraphAugmentor):
        self._augment_pipeline_dict[graph_augmentor._unique_id] = graph_augmentor

    def remove(self, graph_augmentor: GraphAugmentor):
        del self._augment_pipeline_dict[graph_augmentor._unique_id]

    def reset_pipeline(self):
        self._augment_pipeline_dict = dict()

    def augment(self, graph_loader: GraphLoader):
        augmented_graph_loader = copy.copy(graph_loader)
        for key in self._augment_pipeline_dict:
            augmentor = self._augment_pipeline_dict[key]
            augmented_graph_loader = augmentor.augment(augmented_graph_loader)
        return augmented_graph_loader
