
from torch.utils.data import Dataset
from Scripts.DataManager.GraphConstructor.GraphConstructor import GraphConstructor

  
class GraphConstructorDataset(Dataset):

    def __init__(self, graph_constructor: GraphConstructor, graph_labels):
        self.graph_constructor = graph_constructor
        self.graph_labels = graph_labels

    def __getitem__(self, index):
        x = self.graph_constructor.get_graph(index)
        y = self.graph_labels[index]
        return x, y

    def __len__(self):
        return len(self.graph_labels)

class GraphConstructorDatasetRanged(Dataset):

    def __init__(self, graph_constructor: GraphConstructor, graph_labels, begin, end):
        self.graph_constructor = graph_constructor
        self.graph_labels = graph_labels
        self.begin = begin
        self.end = end
        self.len = len(self.graph_labels)

    def __getitem__(self, index):
        x = self.graph_constructor.get_graph(index)
        y = self.graph_labels[index]
        return x, y

    def __len__(self):
        return self.len